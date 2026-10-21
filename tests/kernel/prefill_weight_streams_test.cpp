#include "ftlpu/core/bf16.hpp"
#include "ftlpu/icu/fu_3d_codec.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using namespace ftlpu;
constexpr std::size_t T = 32, K = 128, N = 64;
constexpr std::size_t weightStream = 8, activationStream = 0;
constexpr std::size_t firstLoad = 20;
constexpr std::array<std::size_t, 2> activationSlices{40, 41};
constexpr std::array<std::size_t, 4> resultSlices{44, 45, 46, 47};
static_assert(hw::kMxmRows == T && hw::kMxmColumns == T);

std::size_t inputLatency(std::size_t slice) {
    // MEM reads inject at boundary group+1, then traverse to the MXM column.
    return hw::kMxmBoundaryStreamRegisterColumn - slice / hw::kMemSlicesPerGroup;
}
std::size_t outputLatency(std::size_t slice) {
    return hw::kMxmBoundaryStreamRegisterColumn - slice / hw::kMemSlicesPerGroup;
}
float activation(std::size_t m, std::size_t k) {
    return float(int((m * 7 + k * 5 + 3) % 23) - 11) * 0.0625f;
}
std::int8_t weight(std::size_t k, std::size_t n) {
    return std::int8_t(int((k * 13 + n * 7 + k / T * 3) % 31) - 15);
}
float scale(std::size_t block) { return (block % 3 + 1) * 0.03125f; }

// Experimental schedule, directly enqueued into the real distributed ICUs.
// All waits are fixed on-chip transport dependencies; no external-memory
// latency is predicted. Weight data starts resident in MEM for this experiment.
class Schedule {
public:
    explicit Schedule(InstructionControlUnit& value): icu(value) {}
    void mem(std::size_t slice, std::size_t cycle, MemInstruction op) {
        auto queue = InstructionControlUnit::mem_queue(Hemisphere::East, slice);
        auto& next = memCursor[queue];
        if (cycle < next) throw std::logic_error("MEM queue overlap");
        icu.enqueue_mem_nop(queue, cycle - next);
        icu.enqueue_mem(queue, op); next = cycle + 1;
        end = std::max(end, next);
    }
    void load(std::size_t cycle, MxmControlInstruction op, float s) {
        if (cycle < loadCursor) throw std::logic_error("MXM load queue overlap");
        icu.enqueue_mxm_load_nop(0, cycle - loadCursor);
        icu.enqueue_mxm_dequant_nop(0, cycle - loadCursor);
        icu.enqueue_mxm(0, op);
        icu.enqueue_mxm_dequant(0, MxmDequantInstruction::Scale(s));
        loadOps.push_back(op);
        loadCursor = cycle + 1; end = std::max(end, loadCursor);
    }
    void compute(std::size_t cycle, MxmControlInstruction op) {
        if (cycle < computeCursor) throw std::logic_error("MXM compute queue overlap");
        icu.enqueue_mxm_compute_nop(0, cycle - computeCursor);
        icu.enqueue_mxm(0, op); computeCursor = cycle + 1;
        end = std::max(end, computeCursor);
    }
    void columnBlock(std::size_t cycle, std::size_t buffer, float s) {
        if (cycle < loadCursor) throw std::logic_error("column load queue overlap");
        IcuLoop3D loop{};
        loop.wait_cycle = cycle - loadCursor;
        loop.counts = {T, 1, 1};
        loop.cycle_strides = {1, T, T};
        auto instruction = MxmLoadIcuInstruction::Load3D(loop, buffer,
            MxmIcuBufferMode::Fixed, 0, {1, 0, 0}, weightStream);
        instruction.opcode = MxmLoadIcuOpcode::LoadColumn3D;
        const auto decoded = isa::decode_mxm_load_icu_3d_instruction(
            isa::encode_mxm_load_icu_shortest_instruction(instruction));
        icu.enqueue_mxm_load_3d(0, decoded);
        icu.enqueue_mxm_dequant_3d(0,
            MxmDequantIcuInstruction::Dequant3D(loop, MxmDequantInstruction::Scale(s)));
        for (std::size_t i = 0; i < T; ++i)
            loadOps.push_back(MxmControlInstruction::IWColumn(buffer, i / 8, i % 8,
                MxmWeightInputMode::Int8DequantBf16, weightStream));
        loadCursor = cycle + T;
        end = std::max(end, loadCursor);
    }
    std::size_t end = 0;
    std::vector<MxmControlInstruction> loadOps;
private:
    InstructionControlUnit& icu;
    std::array<std::size_t, InstructionControlUnit::kMemQueues> memCursor{};
    std::size_t loadCursor = 0, computeCursor = 0;
};

struct Result {
    std::size_t tokens{}, streams{}, firstCompute{}, lastCompute{}, computeCycles{},
        loadCycles{}, overlapCycles{}, internalBubbles{}, endCycle{}, weightBytes{};
    float maxError = 0;
    std::vector<float> output;
};

Result run(std::size_t tokens, std::size_t streams, const std::filesystem::path& directory) {
    auto system = std::make_unique<TspSliceSystem>();
    Schedule schedule(system->icu());
    const std::size_t blocks = K / T * (N / T);
    const std::size_t loadLength = T / streams;
    const std::size_t interval = std::max(tokens, loadLength);
    const std::size_t computeStart = firstLoad + loadLength;
    if (tokens * (N / T) > hw::kMxmAccumulatorRows)
        throw std::logic_error("test exceeds configured accumulator capacity");

    // One-stream compiler layout: retain eight MEM slices, with descending
    // supercell storage rows; time-multiplex their reads onto one SR stream.
    for (std::size_t block = 0; block < blocks; ++block) {
        const auto nb = block / (K / T), kb = block % (K / T);
        for (std::size_t column = 0; column < T; ++column) {
            const auto slice = column % 8;
            const auto address = block * 4 + (streams == 1 ? 3 - column / 8 : column / 8);
            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
                for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
                    system->initialize_mem_sram_lane_byte(slice, tile, address, lane,
                        std::uint8_t(weight(kb * T + tile * 8 + lane, nb * T + column)));
        }
    }
    for (std::size_t kb = 0; kb < K / T; ++kb)
        for (std::size_t m = 0; m < tokens; ++m)
            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
                for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                    auto bits = Bf16::from_float(activation(m, kb * T + tile * 8 + lane)).bits();
                    for (std::size_t byte = 0; byte < 2; ++byte)
                        system->initialize_mem_sram_lane_byte(activationSlices[byte], tile,
                            kb * tokens + m, lane, std::uint8_t(bits >> (8 * byte)));
                }

    for (std::size_t block = 0; block < blocks; ++block) {
        const auto nb = block / (K / T), kb = block % (K / T);
        const auto compute = computeStart + block * interval;
        const auto load = compute - loadLength;
        // Column-major IW follows the same tile skew as Compute. For seq32
        // the first column of buffer A is rewritten just after its last old
        // use; later column groups are rewritten later. No whole-buffer
        // drain or third weight buffer is assumed.
        for (std::size_t i = 0; i < loadLength; ++i) {
            for (std::size_t stream = 0; stream < streams; ++stream) {
                const auto slice = streams == 1 ? i % 8 : stream;
                const auto address = block * 4 + (streams == 1 ? 3 - i / 8 : i);
                schedule.mem(slice, load + i - inputLatency(slice),
                    MemInstruction::Read(address, StreamId::East(weightStream + stream)));
            }
            const auto op = streams == 1
                ? MxmControlInstruction::IWColumn(block % 2, i / 8, i % 8,
                    MxmWeightInputMode::Int8DequantBf16, weightStream)
                : MxmControlInstruction::IW(block % 2, i,
                    MxmWeightInputMode::Int8DequantBf16, weightStream);
            if (streams != 1) schedule.load(load + i, op, scale(block));
        }
        if (streams == 1) schedule.columnBlock(load, block % 2, scale(block));
        for (std::size_t m = 0; m < tokens; ++m) {
            for (std::size_t byte = 0; byte < 2; ++byte)
                schedule.mem(activationSlices[byte], compute + m - inputLatency(activationSlices[byte]),
                    MemInstruction::Read(kb * tokens + m, StreamId::East(activationStream + byte)));
            schedule.compute(compute + m, MxmControlInstruction::Compute(block % 2,
                activationStream, 0, nb * tokens + (m / T) * T, 1, MxmAccumulatorDestination::Sram,
                MxmDataFormat::BFloat16, false));
        }
    }
    const auto readStart = computeStart + (blocks - 1) * interval + tokens + 16;
    for (std::size_t nb = 0; nb < N / T; ++nb)
        for (std::size_t m = 0; m < tokens; ++m) {
            auto address = nb * tokens + m;
            auto cycle = readStart + address;
            schedule.compute(cycle, MxmControlInstruction::AccumulatorRead(address, 0, true));
            for (std::size_t byte = 0; byte < 4; ++byte)
                schedule.mem(resultSlices[byte], cycle + outputLatency(resultSlices[byte]),
                    MemInstruction::Write(address, StreamId::West(byte)));
        }

    std::ofstream trace;
    if (!directory.empty()) {
        std::filesystem::create_directories(directory);
        trace.open(directory / ("seq" + std::to_string(tokens) + "_s" + std::to_string(streams) + ".pipeline.csv"));
        trace << "start,end,resource,detail\n";
    }
    Result result; result.tokens = tokens; result.streams = streams;
    bool sawCompute = false;
    std::size_t observedWeightBytes = 0;
    for (std::size_t cycle = 0; cycle < schedule.end + 16; ++cycle) {
        TspSliceSystem::LogSinks sinks; sinks.capture_mem_trace = true;
        try { system->tick(sinks); }
        catch (const std::exception& e) {
            throw std::runtime_error("seq=" + std::to_string(tokens) + " streams=" +
                std::to_string(streams) + " cycle=" + std::to_string(cycle) + ": " + e.what());
        }
        const auto& control = system->mxm_unit(0).control();
        const auto loadAction = system->icu().mxm_load_iq(0).last_trace().action;
        const bool loading = loadAction == IcuQueueAction::FunctionalIssue
            || loadAction == IcuQueueAction::MxmLoad3DIssue;
        const bool computing = control.compute_active(0);
        if (loading) {
            const auto dqAction = system->icu().mxm_dequant_iq(0).last_trace().action;
            if (dqAction != IcuQueueAction::FunctionalIssue
                && dqAction != IcuQueueAction::MxmDequant3DIssue)
                throw std::logic_error("load lacks matching real dequant issue");
            const auto& load = schedule.loadOps.at(result.loadCycles);
            ++result.loadCycles;
            if (trace) trace << cycle << ',' << cycle + 1 << ",MXM.E0.Load,IW buffer=" << load.weight_buffer
                << " column=" << load.weight_column << " inner=" << load.weight_inner_column
                << " stream=" << load.weight_stream_base << " count=" << streams << '\n';
            if (trace) trace << cycle << ',' << cycle + 1 << ",MXM.E0.Dequant,INT8 to BF16\n";
        }
        if (computing) {
            if (!sawCompute) result.firstCompute = cycle;
            sawCompute = true; result.lastCompute = cycle; ++result.computeCycles;
            if (trace) trace << cycle << ',' << cycle + 1 << ",MXM.E0.Compute,buffer="
                << control.compute_weight_buffer(0).value() << '\n';
        }
        if (loading && computing) ++result.overlapCycles;
        for (const auto& transfer : system->mem_array(Hemisphere::East).executed_transfers()) {
            const bool read = transfer.kind == MemArrayModel::MemTransfer::Kind::LoadSramToStream;
            if (read && transfer.mem_slice < 8) {
                observedWeightBytes += transfer.bytes.size();
                if (transfer.stream.index() < weightStream || transfer.stream.index() >= weightStream + streams)
                    throw std::logic_error("weight escaped configured stream window");
            }
            if (trace) trace << cycle << ',' << cycle + 1 << ",MEM.E.s" << transfer.mem_slice << ".b" << transfer.bank
                << ".t" << transfer.tile << (read ? ".Read" : ".Write")
                << ",address=" << transfer.address << " stream="
                << (transfer.stream.direction() == StreamDirection::East ? 'E' : 'W')
                << transfer.stream.index() << '\n';
        }
        result.endCycle = cycle + 1;
    }
    result.internalBubbles = result.lastCompute - result.firstCompute + 1 - result.computeCycles;
    result.weightBytes = observedWeightBytes;
    if (result.computeCycles != blocks * tokens || result.loadCycles != blocks * loadLength
        || result.weightBytes != K * N
        || result.internalBubbles != (blocks - 1) * (interval - tokens))
        throw std::runtime_error("observed issue/transfer counts differ from schedule");

    for (std::size_t m = 0; m < tokens; ++m)
        for (std::size_t n = 0; n < N; ++n) {
            std::uint32_t bits = 0;
            const auto address = (n / T) * tokens + m;
            for (std::size_t byte = 0; byte < 4; ++byte)
                bits |= std::uint32_t(system->read_mem_sram_lane_byte(resultSlices[byte],
                    (n % T) / 8, address, n % 8)) << (8 * byte);
            const auto actual = std::bit_cast<float>(bits);
            float expected = 0;
            for (std::size_t k = 0; k < K; ++k) {
                const auto block = (n / T) * (K / T) + k / T;
                expected += activation(m, k) * Bf16::from_float(float(weight(k, n)) * scale(block)).to_float();
            }
            if (!std::isfinite(actual) || std::fabs(actual - expected) > 1e-5f)
                throw std::runtime_error("numeric mismatch seq=" + std::to_string(tokens) +
                    " streams=" + std::to_string(streams) + " m=" + std::to_string(m) +
                    " n=" + std::to_string(n) + " actual=" + std::to_string(actual) +
                    " expected=" + std::to_string(expected));
            result.maxError = std::max(result.maxError, std::fabs(actual - expected));
            result.output.push_back(actual);
        }
    return result;
}
}

int main(int argc, char** argv) try {
    const auto directory = argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::path{};
    std::ofstream summary;
    if (!directory.empty()) {
        std::filesystem::create_directories(directory);
        summary.open(directory / "summary.json");
        summary << "{\"scope\":\"resident-MEM projection; one physical MXM; one-stream LOAD_COLUMN_3D and eight-stream legacy IW; not whole-layer compiler lowering\","
            << "\"k\":" << K << ",\"n\":" << N << ",\"cases\":[\n";
    }
    bool first = true;
    for (const auto tokens : {16u, 32u, 64u}) {
        std::vector<float> baseline;
        for (const auto streams : {8u, 1u}) {
            auto r = run(tokens, streams, directory);
            if (streams == 8) baseline = r.output;
            else if (r.output != baseline) throw std::runtime_error("one/eight stream outputs differ");
            std::cout << "seq=" << tokens << " weight_streams=" << streams
                << " first_compute=" << r.firstCompute << " last_compute=" << r.lastCompute
                << " compute_cycles=" << r.computeCycles << " internal_bubbles=" << r.internalBubbles
                << " load_compute_overlap=" << r.overlapCycles << " max_abs_error=" << r.maxError << '\n';
            if (summary) {
                if (!first) summary << ",\n";
                first = false;
                summary << "{\"tokens\":" << tokens << ",\"weight_streams\":" << streams
                    << ",\"first_compute\":" << r.firstCompute << ",\"last_compute\":" << r.lastCompute
                    << ",\"compute_cycles\":" << r.computeCycles << ",\"internal_bubbles\":" << r.internalBubbles
                    << ",\"load_cycles\":" << r.loadCycles << ",\"load_compute_overlap\":" << r.overlapCycles
                    << ",\"weight_bytes\":" << r.weightBytes << ",\"end_cycle\":" << r.endCycle
                    << ",\"max_abs_error\":" << r.maxError << ",\"numeric_passed\":true}";
            }
        }
    }
    if (summary) summary << "\n],\"status\":\"passed\"}\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "prefill_weight_streams_test: " << e.what() << '\n';
    return 1;
}
