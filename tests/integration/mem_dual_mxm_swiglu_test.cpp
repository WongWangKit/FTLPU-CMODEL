#include "ftlpu/system/tsp_slice_system.hpp"
#include "ftlpu/mxm/performance_monitor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kMxmCount = ftlpu::TspSliceSystem::kMxmCount;
constexpr std::size_t kMatrixCount = 3;
constexpr std::size_t kGateMatrix = 0;
constexpr std::size_t kUpMatrix = 1;
constexpr std::size_t kDownMatrix = 2;
constexpr std::size_t kActivationRows = 160;
constexpr std::size_t kKPerPass = ftlpu::hw::kMxmRows;
constexpr std::size_t kColumns = ftlpu::hw::kMxmColumns;
constexpr std::size_t kHiddenColumns = 640;
constexpr std::size_t kPasses = kHiddenColumns / kColumns;
constexpr std::size_t kBlocks = ftlpu::hw::kMxmSupercellsPerPlane;
constexpr std::size_t kLanes = ftlpu::hw::kLanesPerTile;
constexpr std::size_t kLoadStreams = ftlpu::hw::kMxmLoadStreamsPerCycle;
constexpr std::size_t kActivationMemColumn = 32;
constexpr std::size_t kSwigluMemColumn = 40;
constexpr std::size_t kSwigluMemColumn1 = 41;
constexpr std::size_t kFinalMemColumn = 42;
constexpr std::size_t kActivationStream = 0;
constexpr std::size_t kActivationStream1 = 16;
constexpr std::size_t kActivationEarlyStream = 16;
constexpr std::size_t kActivationBridgeStream = 30;
constexpr std::size_t kGateStreamBase = 32;
constexpr std::size_t kUpStreamBase = 36;
constexpr std::size_t kGateWestStreamBase = kGateStreamBase - ftlpu::hw::kEastStreams;
constexpr std::size_t kUpWestStreamBase = kUpStreamBase - ftlpu::hw::kEastStreams;
constexpr std::size_t kLhsStreamBase = 32;
constexpr std::size_t kRhsStreamBase = 36;
constexpr std::size_t kLhsWestStreamBase = kLhsStreamBase - ftlpu::hw::kEastStreams;
constexpr std::size_t kRhsWestStreamBase = kRhsStreamBase - ftlpu::hw::kEastStreams;
constexpr std::size_t kOutputStream = 0;
constexpr std::size_t kSwigluOutputStream = 31;
constexpr std::size_t kSwigluLatency = 9;
constexpr std::size_t kAddQuantLatency = 5;
constexpr std::size_t kWeightHandoffBaseCycle = 18;
constexpr std::size_t kActivationHandoffBaseCycle = 4;
constexpr std::size_t kLogTile = 0;
constexpr float kProjectionInputScale = 1.0f / 2048.0f;
constexpr float kSwigluOutputScale = 1.0f / 128.0f;
constexpr float kDownPartialScale = 1.0f / 512.0f;
constexpr float kFinalOutputScale = 1.0f / 64.0f;

using MatrixI8 = std::vector<std::int8_t>;
using MatrixI32 = std::vector<std::int32_t>;

std::int8_t weight_value(std::size_t matrix_id, std::size_t k, std::size_t n);

class NullBuffer : public std::streambuf {
public:
    int overflow(int c) override
    {
        return c;
    }
};

struct TestLogs {
    explicit TestLogs(const std::filesystem::path& directory, bool enable)
        : enabled(enable)
        , null_stream(&null_buffer)
        , icu(enabled ? static_cast<std::ostream&>(icu_file) : null_stream)
        , mem(enabled ? static_cast<std::ostream&>(mem_file) : null_stream)
        , mxm(enabled ? static_cast<std::ostream&>(mxm_file) : null_stream)
        , vxm(enabled ? static_cast<std::ostream&>(vxm_file) : null_stream)
    {
        if (!enabled) {
            return;
        }
        icu_file.open(directory / "icu.log");
        mem_file.open(directory / "mem.log");
        mxm_file.open(directory / "mxm.log");
        vxm_file.open(directory / "vxm.log");
    }

    bool good() const
    {
        return !enabled || (icu_file.good() && mem_file.good() && mxm_file.good() && vxm_file.good());
    }

    std::ostream* icu_ptr()
    {
        return enabled ? &icu : nullptr;
    }

    std::ostream* mem_ptr()
    {
        return enabled ? &mem : nullptr;
    }

    std::ostream* mxm_ptr()
    {
        return enabled ? &mxm : nullptr;
    }

    std::ostream* vxm_ptr()
    {
        return enabled ? &vxm : nullptr;
    }

    bool enabled{false};
    NullBuffer null_buffer;
    std::ostream null_stream;
    std::ofstream icu_file;
    std::ofstream mem_file;
    std::ofstream mxm_file;
    std::ofstream vxm_file;
    std::ostream& icu;
    std::ostream& mem;
    std::ostream& mxm;
    std::ostream& vxm;
};

bool ffn_logs_enabled()
{
    const auto* value = std::getenv("FTLPU_FFN_LOG");
    if (value == nullptr) {
        return false;
    }
    const auto text = std::string(value);
    return !(text.empty() || text == "0" || text == "false" || text == "FALSE" || text == "off" || text == "OFF");
}

struct MxmArrayStateSummary {
    bool in_all_compute{false};
    std::size_t all_compute_start{0};
    std::size_t all_compute_last{0};
};

struct MxmOutputEvent {
    std::size_t row{0};
    std::size_t tile{0};
};

struct SwigluStreamConfig {
    const ftlpu::VxmLane::SwigluParams* params{nullptr};
    MatrixI8* output{nullptr};
    std::size_t mem_column{0};
    std::size_t hidden_pass{0};
    std::size_t output_stream{0};
};

struct AddQuantStreamConfig {
    const ftlpu::VxmLane::AddQuantParams* params{nullptr};
    MatrixI8* output{nullptr};
    std::size_t mem_column{0};
    std::size_t pass{0};
    std::size_t output_stream{0};
};

struct VxmStageIssue {
    std::size_t row{0};
    std::size_t stage{0};
};

struct ActivationReadConfig {
    std::size_t column0{0};
    std::size_t pass0{0};
    std::size_t stream0{0};
    std::size_t column1{0};
    std::size_t pass1{0};
    std::size_t stream1{0};
    bool dual_stream{false};
};

void flush_all_compute_summary(
    std::ostream& os,
    const char* label,
    MxmArrayStateSummary& summary);

void log_mxm_array_state(
    std::ostream& os,
    const char* label,
    std::size_t cycle,
    const ftlpu::MxmControlSlice& control,
    const ftlpu::Mxm& mxm,
    MxmArrayStateSummary& summary);

std::vector<MxmOutputEvent> capture_mxm_outputs(
    const ftlpu::Mxm& mxm,
    MatrixI32& output_matrix);

bool require(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

bool verify_loaded_weights(
    const ftlpu::TspSliceSystem& system,
    std::size_t mxm_id,
    std::size_t weight_buffer,
    std::size_t matrix_id,
    std::size_t pass,
    const char* label)
{
    const std::array<std::size_t, 4> sample_tiles {0, 7, 13, 19};
    const std::array<std::size_t, 5> sample_columns {0, 1, 31, 128, 319};
    for (const auto tile : sample_tiles) {
        for (const auto column : sample_columns) {
            const auto column_block = column / kLoadStreams;
            const auto local_column = column % kLoadStreams;
            for (std::size_t lane : {std::size_t {0}, std::size_t {15}}) {
                const auto global_k = matrix_id == kDownMatrix
                    ? pass * kKPerPass + tile * kLanes + lane
                    : tile * kLanes + lane;
                const auto global_column = matrix_id == kDownMatrix
                    ? column
                    : pass * kColumns + column;
                const auto actual = system.mxm_unit(mxm_id).array().weight(
                    weight_buffer,
                    tile,
                    column_block,
                    lane,
                    local_column);
                const auto expected = weight_value(matrix_id, global_k, global_column);
                if (actual != expected) {
                    std::cerr << label
                              << " weight mismatch mxm=" << mxm_id
                              << " tile=" << tile
                              << " lane=" << lane
                              << " column=" << column
                              << " actual=" << static_cast<int>(actual)
                              << " expected=" << static_cast<int>(expected)
                              << '\n';
                    return false;
                }
            }
        }
    }
    return true;
}

std::int8_t weight_value(std::size_t matrix_id, std::size_t k, std::size_t n)
{
    const auto mixed = matrix_id * 37 + k * 11 + n * 17 + (k * n + matrix_id * 5) % 23;
    const auto raw = static_cast<int>(mixed % 31);
    return static_cast<std::int8_t>(raw - 15);
}

std::int8_t activation_value(std::size_t m, std::size_t k)
{
    const auto mixed = m * 13 + k * 7 + ((m + 3) * (k + 5)) % 29;
    const auto raw = static_cast<int>(mixed % 25);
    return static_cast<std::int8_t>(raw - 12);
}

template <typename IssueFn>
void enqueue_swiglu_stage(
    IssueFn issue,
    const ftlpu::VxmLane::SwigluParams& params,
    std::size_t stage,
    std::size_t gate_stream_base,
    std::size_t up_stream_base,
    std::size_t output_stream)
{
    switch (stage) {
    case 0:
        issue(0, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::StreamInt32(gate_stream_base),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            1.0f,
            0,
            ftlpu::VxmCastTarget::Float32,
        });
        issue(1, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::StreamInt32(up_stream_base),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            1.0f,
            0,
            ftlpu::VxmCastTarget::Float32,
        });
        break;
    case 1:
        issue(2, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Alu(0),
            ftlpu::VxmLaneOperand::Imm(params.gate_scale),
        });
        issue(3, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Alu(1),
            ftlpu::VxmLaneOperand::Imm(params.up_scale),
        });
        break;
    case 2:
        issue(4, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Alu(2),
            ftlpu::VxmLaneOperand::Alu(3),
        });
        issue(5, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Negate,
            ftlpu::VxmLaneOperand::Alu(2),
        });
        break;
    case 3:
        issue(6, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Exp,
            ftlpu::VxmLaneOperand::Alu(5),
        });
        issue(9, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Pass,
            ftlpu::VxmLaneOperand::Alu(4),
        });
        break;
    case 4:
        issue(7, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Add,
            ftlpu::VxmLaneOperand::Alu(6),
            ftlpu::VxmLaneOperand::Imm(1.0f),
        });
        issue(10, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Pass,
            ftlpu::VxmLaneOperand::Alu(9),
        });
        break;
    case 5:
        issue(8, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Divide,
            ftlpu::VxmLaneOperand::Imm(1.0f),
            ftlpu::VxmLaneOperand::Alu(7),
        });
        issue(11, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Pass,
            ftlpu::VxmLaneOperand::Alu(10),
        });
        break;
    case 6:
        issue(12, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Alu(11),
            ftlpu::VxmLaneOperand::Alu(8),
        });
        break;
    case 7:
        issue(13, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Alu(12),
            ftlpu::VxmLaneOperand::Imm(1.0f / params.output_scale),
        });
        break;
    case 8:
        issue(14, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Add,
            ftlpu::VxmLaneOperand::Alu(13),
            ftlpu::VxmLaneOperand::Imm(static_cast<float>(params.output_zero_point)),
        });
        break;
    case 9:
        issue(15, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::Alu(14),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            1.0f,
            0,
            ftlpu::VxmCastTarget::Int8,
            output_stream,
        });
        break;
    default:
        throw std::out_of_range("SwiGLU VXM stage is outside the ALU pipeline");
    }
}

template <typename IssueFn>
void enqueue_add_quant_stage(
    IssueFn issue,
    const ftlpu::VxmLane::AddQuantParams& params,
    std::size_t stage,
    std::size_t lhs_stream_base,
    std::size_t rhs_stream_base,
    std::size_t output_stream)
{
    switch (stage) {
    case 0:
        issue(0, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::StreamInt32(lhs_stream_base),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            1.0f,
            0,
            ftlpu::VxmCastTarget::Float32,
        });
        issue(1, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::StreamInt32(rhs_stream_base),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            1.0f,
            0,
            ftlpu::VxmCastTarget::Float32,
        });
        break;
    case 1:
        issue(2, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Alu(0),
            ftlpu::VxmLaneOperand::Imm(params.lhs_scale),
        });
        issue(3, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Alu(1),
            ftlpu::VxmLaneOperand::Imm(params.rhs_scale),
        });
        break;
    case 2:
        issue(4, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Add,
            ftlpu::VxmLaneOperand::Alu(2),
            ftlpu::VxmLaneOperand::Alu(3),
        });
        break;
    case 3:
        issue(5, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Alu(4),
            ftlpu::VxmLaneOperand::Imm(1.0f / params.output_scale),
        });
        break;
    case 4:
        issue(6, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Add,
            ftlpu::VxmLaneOperand::Alu(5),
            ftlpu::VxmLaneOperand::Imm(static_cast<float>(params.output_zero_point)),
        });
        break;
    case 5:
        issue(7, ftlpu::VxmLaneAluInstruction {
            ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::Alu(6),
            ftlpu::VxmLaneOperand::Imm(0.0f),
            1.0f,
            0,
            ftlpu::VxmCastTarget::Int8,
            output_stream,
        });
        break;
    default:
        throw std::out_of_range("add-quant VXM stage is outside the ALU pipeline");
    }
}

std::size_t matrix_address(std::size_t local_row, std::size_t column)
{
    return local_row * kColumns + column;
}

std::size_t vector_row_address(std::size_t row, std::size_t lane)
{
    return row * kLanes + lane;
}

std::size_t weight_address(
    std::size_t matrix_id,
    std::size_t pass,
    std::size_t column_block)
{
    return matrix_id * kPasses * kBlocks * kLanes
        + pass * kBlocks * kLanes
        + column_block * kLanes;
}

std::size_t matrix_index(std::size_t row, std::size_t column)
{
    return row * kColumns + column;
}

std::size_t hidden_matrix_index(std::size_t row, std::size_t column)
{
    return row * kHiddenColumns + column;
}

std::size_t swiglu_address(std::size_t pass, std::size_t row, std::size_t lane)
{
    return pass * kActivationRows * kLanes + row * kLanes + lane;
}

void stage_activation_matrix(ftlpu::TileArrayModel& mem)
{
    for (std::size_t row = 0; row < kActivationRows; ++row) {
        for (std::size_t k = 0; k < kKPerPass; ++k) {
            const auto tile = k / kLanes;
            const auto lane = k % kLanes;
            mem.set_sram_lane_byte(
                kActivationMemColumn,
                tile,
                vector_row_address(row, 0),
                lane,
                static_cast<std::uint8_t>(activation_value(row, k)));
        }
    }
}

void stage_weight_matrices(ftlpu::TileArrayModel& mem)
{
    for (std::size_t matrix_id = 0; matrix_id < kMatrixCount; ++matrix_id) {
        for (std::size_t pass = 0; pass < kPasses; ++pass) {
            for (std::size_t tile = 0; tile < kBlocks; ++tile) {
                for (std::size_t column_block = 0; column_block < kBlocks; ++column_block) {
                    for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
                        const auto address = weight_address(matrix_id, pass, column_block);
                        for (std::size_t lane = 0; lane < kLanes; ++lane) {
                            const auto global_k = matrix_id == kDownMatrix
                                ? pass * kKPerPass + tile * kLanes + lane
                                : tile * kLanes + lane;
                            const auto column = matrix_id == kDownMatrix
                                ? column_block * kLoadStreams + stream
                                : pass * kColumns + column_block * kLoadStreams + stream;
                            mem.set_sram_lane_byte(
                                stream,
                                tile,
                                address,
                                lane,
                                static_cast<std::uint8_t>(weight_value(matrix_id, global_k, column)));
                            mem.set_sram_lane_byte(
                                kLoadStreams + stream,
                                tile,
                                address,
                                lane,
                                static_cast<std::uint8_t>(weight_value(matrix_id, global_k, column)));
                        }
                    }
                }
            }
        }
    }
}

std::size_t east_stream_cycles_to_sreg11(std::size_t column)
{
    constexpr auto kTargetSreg = ftlpu::hw::kStreamRegisterColumns - 1;
    return kTargetSreg - column / ftlpu::hw::kSlicesPerGroup;
}

std::size_t activation_read_address(std::size_t column, std::size_t pass, std::size_t row)
{
    return column == kSwigluMemColumn || column == kSwigluMemColumn1
        ? swiglu_address(pass, row, 0)
        : vector_row_address(row, 0);
}

void enqueue_mem_read_sequence(
    ftlpu::InstructionControlUnit& icu,
    std::size_t column,
    std::size_t first_cycle,
    std::size_t address,
    std::size_t stream,
    std::size_t count,
    std::int64_t address_stride)
{
    if (count == 0) {
        return;
    }
    icu.enqueue_mem_nop(column, first_cycle);
    icu.enqueue_mem(column, ftlpu::MemInstruction::Read(address, stream));
    icu.enqueue_mem_repeat(column, count - 1, 1, address_stride);
}

void enqueue_mxm_sequence(
    ftlpu::InstructionControlUnit& icu,
    std::size_t mxm,
    std::size_t first_cycle,
    ftlpu::MxmControlInstruction instruction,
    std::size_t count)
{
    if (count == 0) {
        return;
    }
    icu.enqueue_mxm_nop(mxm, first_cycle);
    icu.enqueue_mxm(mxm, instruction);
    if (instruction.opcode == ftlpu::MxmControlOpcode::Compute) {
        icu.enqueue_mxm_compute_repeat(mxm, count - 1);
    } else {
        icu.enqueue_mxm_load_repeat(mxm, count - 1);
    }
}

void load_dual_mxm_from_mem(
    ftlpu::TspSliceSystem& system,
    std::size_t matrix0,
    std::size_t pass0,
    std::size_t matrix1,
    std::size_t pass1,
    TestLogs& logs,
    const ActivationReadConfig* activation_prefetch = nullptr)
{
    const auto sinks = ftlpu::TspSliceSystem::LogSinks {
        logs.icu_ptr(),
        logs.mem_ptr(),
        logs.mxm_ptr(),
        logs.vxm_ptr(),
        nullptr,
        kLogTile,
        kLogTile,
        kLogTile,
    };
    constexpr auto kTotalCycles = kWeightHandoffBaseCycle + 2 * kBlocks;

    for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
        const auto mxm0_column = stream;
        const auto mxm1_column = kLoadStreams + stream;
        const auto mxm0_first = kWeightHandoffBaseCycle - east_stream_cycles_to_sreg11(mxm0_column) - 1;
        const auto mxm1_first = kWeightHandoffBaseCycle - east_stream_cycles_to_sreg11(mxm1_column) - 1;
        enqueue_mem_read_sequence(
            system.icu(),
            mxm0_column,
            mxm0_first,
            weight_address(matrix0, pass0, kBlocks - 1),
            stream,
            kBlocks,
            -static_cast<std::int64_t>(kLanes));
        enqueue_mem_read_sequence(
            system.icu(),
            mxm1_column,
            mxm1_first,
            weight_address(matrix1, pass1, kBlocks - 1),
            kLoadStreams + stream,
            kBlocks,
            -static_cast<std::int64_t>(kLanes));
    }

    if (activation_prefetch != nullptr) {
        const auto latency0 = east_stream_cycles_to_sreg11(activation_prefetch->column0);
        if (latency0 < kTotalCycles) {
            const auto first_cycle = kTotalCycles - latency0;
            enqueue_mem_read_sequence(
                system.icu(),
                activation_prefetch->column0,
                first_cycle,
                activation_read_address(activation_prefetch->column0, activation_prefetch->pass0, 0),
                activation_prefetch->stream0,
                kTotalCycles - first_cycle,
                kLanes);
        }

        if (activation_prefetch->dual_stream) {
            const auto latency1 = east_stream_cycles_to_sreg11(activation_prefetch->column1);
            if (latency1 < kTotalCycles) {
                const auto first_cycle = kTotalCycles - latency1;
                enqueue_mem_read_sequence(
                    system.icu(),
                    activation_prefetch->column1,
                    first_cycle,
                    activation_read_address(activation_prefetch->column1, activation_prefetch->pass1, 0),
                    activation_prefetch->stream1,
                    kTotalCycles - first_cycle,
                    kLanes);
            }
        }
    }

    for (std::size_t cycle = 0; cycle < kTotalCycles; ++cycle) {
        if (logs.enabled) {
            logs.mxm << "dual_mxm_load cycle " << cycle << '\n'
                     << "  mxm0 matrix=" << matrix0 << " pass=" << pass0 << " streams=E0..E15\n"
                     << "  mxm1 matrix=" << matrix1 << " pass=" << pass1 << " streams=E16..E31\n";
        }

        for (std::size_t column_block = 0; column_block < kBlocks; ++column_block) {
            const auto handoff_cycle = kWeightHandoffBaseCycle + column_block;
            if (cycle == handoff_cycle) {
                system.icu().enqueue_mxm(0, ftlpu::MxmControlInstruction::IW());
                system.icu().enqueue_mxm(1, ftlpu::MxmControlInstruction::IW());
            }
        }
        system.tick(sinks);
    }
}

std::array<MatrixI32, kMxmCount> run_dual_mxm_gemm(
    ftlpu::TspSliceSystem& system,
    std::size_t activation_column0,
    std::size_t activation_pass0,
    std::size_t activation_column1,
    std::size_t activation_pass1,
    TestLogs& logs,
    SwigluStreamConfig* swiglu_stream = nullptr,
    AddQuantStreamConfig* add_quant_stream = nullptr,
    bool activation_prefetched = false)
{
    if (swiglu_stream != nullptr && add_quant_stream != nullptr) {
        throw std::logic_error("dual MXM GEMM can feed only one VXM post-op stream at a time");
    }
    std::array<MatrixI32, kMxmCount> outputs {
        MatrixI32(kActivationRows * kColumns, 0),
        MatrixI32(kActivationRows * kColumns, 0),
    };
    MxmArrayStateSummary mxm0_summary {};
    MxmArrayStateSummary mxm1_summary {};
    ftlpu::MxmPerformanceMonitor mxm0_perf {};
    ftlpu::MxmPerformanceMonitor mxm1_perf {};

    constexpr auto kComputeTicks = kActivationRows + 2 * kBlocks;
    const auto activation_read_latency0 = east_stream_cycles_to_sreg11(activation_column0);
    const auto activation_read_latency1 = east_stream_cycles_to_sreg11(activation_column1);
    const auto activation_handoff_base = activation_prefetched
        ? std::size_t {0}
        : std::max(activation_read_latency0, activation_read_latency1) + 1;
    const auto vxm_mem_column = swiglu_stream != nullptr
        ? swiglu_stream->mem_column
        : add_quant_stream != nullptr ? add_quant_stream->mem_column : 0;
    const auto vxm_latency = swiglu_stream != nullptr
        ? kSwigluLatency
        : add_quant_stream != nullptr ? kAddQuantLatency : 0;
    const auto vxm_write_latency = swiglu_stream != nullptr || add_quant_stream != nullptr
        ? vxm_mem_column / ftlpu::hw::kSlicesPerGroup + 2
        : 0;
    const auto total_cycles = activation_handoff_base + kComputeTicks
        + (swiglu_stream != nullptr || add_quant_stream != nullptr
                ? ftlpu::hw::kStreamRegisterColumns + vxm_latency + vxm_write_latency + kBlocks + 8
                : 0);
    std::vector<std::vector<MxmOutputEvent>> vxm_feed_rows(total_cycles + ftlpu::hw::kStreamRegisterColumns + 1);
    std::vector<std::vector<MxmOutputEvent>> vxm_output_rows(total_cycles + vxm_latency + 1);
    std::vector<std::vector<MxmOutputEvent>> vxm_write_rows(total_cycles + vxm_write_latency + kBlocks + 1);
    std::vector<std::vector<VxmStageIssue>> vxm_issue_stages(
        total_cycles + ftlpu::hw::kStreamRegisterColumns + vxm_latency + kBlocks + 1);
    std::array<std::deque<std::size_t>, kBlocks> vxm_ready_rows {};
    std::vector<bool> vxm_write_scheduled(kActivationRows, false);
    std::vector<bool> vxm_token_scheduled(kActivationRows, false);

    if (swiglu_stream != nullptr) {
        if (swiglu_stream->params == nullptr || swiglu_stream->output == nullptr) {
            throw std::logic_error("SwiGLU stream config is incomplete");
        }
        swiglu_stream->output->assign(kActivationRows * kColumns, 0);
    }
    if (add_quant_stream != nullptr) {
        if (add_quant_stream->params == nullptr || add_quant_stream->output == nullptr) {
            throw std::logic_error("add-quant stream config is incomplete");
        }
        add_quant_stream->output->assign(kActivationRows * kColumns, 0);
    }

    auto enqueue_activation_reads = [&](std::size_t column, std::size_t pass, std::size_t stream, std::size_t latency) {
        const auto first_cycle = activation_handoff_base > latency ? activation_handoff_base - latency : std::size_t {0};
        const auto first_row = first_cycle + latency - activation_handoff_base;
        if (first_row >= kActivationRows) {
            return;
        }
        enqueue_mem_read_sequence(
            system.icu(),
            column,
            first_cycle,
            activation_read_address(column, pass, first_row),
            stream,
            kActivationRows - first_row,
            kLanes);
    };

    enqueue_activation_reads(activation_column0, activation_pass0, kActivationStream, activation_read_latency0);
    if (activation_column1 != activation_column0 || activation_pass1 != activation_pass0) {
        enqueue_activation_reads(activation_column1, activation_pass1, kActivationStream1, activation_read_latency1);
    }
    enqueue_mxm_sequence(
        system.icu(),
        0,
        activation_handoff_base,
        ftlpu::MxmControlInstruction::Compute(0, kActivationStream, kGateWestStreamBase),
        kActivationRows);
    enqueue_mxm_sequence(
        system.icu(),
        1,
        activation_handoff_base,
        ftlpu::MxmControlInstruction::Compute(0, kActivationStream1, kUpWestStreamBase),
        kActivationRows);

    for (std::size_t cycle = 0; cycle < total_cycles; ++cycle) {
        if ((swiglu_stream != nullptr || add_quant_stream != nullptr) && cycle < vxm_issue_stages.size()) {
            for (const auto& event : vxm_issue_stages[cycle]) {
                auto issue = [&](std::size_t alu, ftlpu::VxmLaneAluInstruction instruction) {
                    system.icu().enqueue_vxm(alu, instruction);
                };
                if (swiglu_stream != nullptr) {
                    enqueue_swiglu_stage(
                        issue,
                        *swiglu_stream->params,
                        event.stage,
                        kGateStreamBase,
                        kUpStreamBase,
                        swiglu_stream->output_stream);
                } else {
                    enqueue_add_quant_stage(
                        issue,
                        *add_quant_stream->params,
                        event.stage,
                        kLhsStreamBase,
                        kRhsStreamBase,
                        add_quant_stream->output_stream);
                }
            }
        }

        if ((swiglu_stream != nullptr || add_quant_stream != nullptr) && cycle < vxm_write_rows.size()) {
            for (const auto& event : vxm_write_rows[cycle]) {
                const auto mem_column = swiglu_stream != nullptr ? swiglu_stream->mem_column : add_quant_stream->mem_column;
                const auto pass = swiglu_stream != nullptr ? swiglu_stream->hidden_pass : add_quant_stream->pass;
                const auto output_stream = swiglu_stream != nullptr ? swiglu_stream->output_stream : add_quant_stream->output_stream;
                system.icu().enqueue_mem(
                    mem_column,
                    ftlpu::MemInstruction::Write(
                        swiglu_address(pass, event.row, 0),
                        output_stream));
            }
        }

        system.dispatch_icu_only(logs.icu_ptr());
        const auto mxm_control_sinks = ftlpu::TspSliceSystem::LogSinks {
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            std::nullopt,
            kLogTile,
            std::nullopt,
        };
        system.tick_mxm_controls_only(mxm_control_sinks);
        if (logs.enabled) {
            system.mem().tick(logs.mem, kLogTile);
        } else {
            system.mem().tick();
        }
        system.tick_mxm_datapaths_only(ftlpu::TspSliceSystem::LogSinks {
            nullptr,
            nullptr,
            logs.mxm_ptr(),
            nullptr,
            nullptr,
            std::nullopt,
            kLogTile,
            std::nullopt,
        });

        auto compute_cycle = std::size_t {0};
        const auto in_compute_window = cycle >= activation_handoff_base
            && (compute_cycle = cycle - activation_handoff_base) < kComputeTicks;

        if (in_compute_window) {
            mxm0_perf.sample(system.mxm_unit(0));
            mxm1_perf.sample(system.mxm_unit(1));
            const auto gate_events = capture_mxm_outputs(system.mxm_unit(0), outputs[0]);
            capture_mxm_outputs(system.mxm_unit(1), outputs[1]);
            if (swiglu_stream != nullptr || add_quant_stream != nullptr) {
                for (const auto& event : gate_events) {
                    const auto feed_cycle = cycle + ftlpu::hw::kStreamRegisterColumns;
                    if (feed_cycle < vxm_feed_rows.size()) {
                        vxm_feed_rows[feed_cycle].push_back(event);
                    }
                    const auto issue_cycle = feed_cycle > event.tile ? feed_cycle - event.tile : cycle;
                    if (!vxm_token_scheduled[event.row]) {
                        for (std::size_t stage = 0; stage <= vxm_latency; ++stage) {
                            const auto stage_cycle = issue_cycle + stage;
                            if (stage_cycle < vxm_issue_stages.size()) {
                                vxm_issue_stages[stage_cycle].push_back(VxmStageIssue {event.row, stage});
                            }
                        }
                        vxm_token_scheduled[event.row] = true;
                    }
                }
            }
            if (logs.enabled) {
                log_mxm_array_state(
                    logs.mxm,
                    "mxm0",
                    compute_cycle,
                    system.mxm_unit(0).control(),
                    system.mxm_unit(0),
                    mxm0_summary);
                log_mxm_array_state(
                    logs.mxm,
                    "mxm1",
                    compute_cycle,
                    system.mxm_unit(1).control(),
                    system.mxm_unit(1),
                    mxm1_summary);
            }
        }

        if (swiglu_stream != nullptr || add_quant_stream != nullptr) {
            if (cycle < vxm_feed_rows.size()) {
                for (const auto& event : vxm_feed_rows[cycle]) {
                    const auto output_cycle = cycle + vxm_latency;
                    if (output_cycle < vxm_output_rows.size()) {
                        vxm_output_rows[output_cycle].push_back(event);
                    }
                }
            }
            if (cycle < vxm_output_rows.size()) {
                for (const auto& event : vxm_output_rows[cycle]) {
                    vxm_ready_rows[event.tile].push_back(event.row);
                }
            }

            const auto bridge_sinks = ftlpu::TspSliceSystem::LogSinks {
                nullptr,
                logs.mem_ptr(),
                nullptr,
                logs.vxm_ptr(),
                nullptr,
                kLogTile,
                std::nullopt,
                kLogTile,
            };
            system.tick_vxm_stream_bridge(bridge_sinks, 0);

            for (std::size_t tile = 0; tile < kBlocks; ++tile) {
                const auto& tile_output = system.vxm().output_at(tile);
                if (!tile_output.has_value()) {
                    continue;
                }
                if (vxm_ready_rows[tile].empty()) {
                    continue;
                }
                const auto row = vxm_ready_rows[tile].front();
                vxm_ready_rows[tile].pop_front();
                auto& output_matrix = swiglu_stream != nullptr ? *swiglu_stream->output : *add_quant_stream->output;
                for (std::size_t lane = 0; lane < kLanes; ++lane) {
                    output_matrix[matrix_index(row, tile * kLanes + lane)] = tile_output->values[lane];
                }

                const auto write_cycle = cycle + vxm_write_latency - tile;
                if (!vxm_write_scheduled[row] && write_cycle < vxm_write_rows.size()) {
                    vxm_write_rows[write_cycle].push_back(MxmOutputEvent {row, tile});
                    vxm_write_scheduled[row] = true;
                }
            }
        }
    }
    if (logs.enabled) {
        flush_all_compute_summary(logs.mxm, "mxm0", mxm0_summary);
        flush_all_compute_summary(logs.mxm, "mxm1", mxm1_summary);
        mxm0_perf.print(logs.mxm, "mxm0");
        mxm1_perf.print(logs.mxm, "mxm1");
    }

    if (logs.enabled) {
        logs.mxm << "dual_mxm_compute shared_activation_streams=E0..E15 rows=160 k=320\n";
    }
    if (swiglu_stream != nullptr) {
        logs.vxm << "vxm_swiglu source=MEM west edge gate_streams=W0..W3 up_streams=W4..W7 out=E"
                 << swiglu_stream->output_stream << '\n';
        logs.mem << "stored SwiGLU stream output column=" << swiglu_stream->mem_column
                 << " hidden_pass=" << swiglu_stream->hidden_pass
                 << " via VXM E" << swiglu_stream->output_stream << " -> MEM Write\n";
    }
    if (add_quant_stream != nullptr) {
        logs.vxm << "vxm_add_quant source=MEM west edge lhs_streams=W0..W3 rhs_streams=W4..W7 out=E"
                 << add_quant_stream->output_stream << '\n';
        logs.mem << "stored final add-quant stream output column=" << add_quant_stream->mem_column
                 << " pass=" << add_quant_stream->pass
                 << " via VXM E" << add_quant_stream->output_stream << " -> MEM Write\n";
    }
    return outputs;
}

char mxm_state_char(
    const ftlpu::MxmControlSlice& control,
    const ftlpu::Mxm& mxm,
    std::size_t tile,
    std::size_t column_block)
{
    if (mxm.computing_cell(tile, column_block)) {
        return 'C';
    }
    if (control.loaded_cell(tile, column_block)) {
        return 'L';
    }
    return '.';
}

void flush_all_compute_summary(
    std::ostream& os,
    const char* label,
    MxmArrayStateSummary& summary)
{
    if (!summary.in_all_compute) {
        return;
    }

    os << label << " array_state all_C cycles "
       << summary.all_compute_start << ".." << summary.all_compute_last << '\n';
    summary.in_all_compute = false;
}

void log_mxm_array_state(
    std::ostream& os,
    const char* label,
    std::size_t cycle,
    const ftlpu::MxmControlSlice& control,
    const ftlpu::Mxm& mxm,
    MxmArrayStateSummary& summary)
{
    std::array<std::array<char, kBlocks>, kBlocks> state {};
    bool all_compute = true;
    bool has_loaded_idle = false;
    for (std::size_t tile = 0; tile < kBlocks; ++tile) {
        for (std::size_t column_block = 0; column_block < kBlocks; ++column_block) {
            const auto value = mxm_state_char(control, mxm, tile, column_block);
            state[tile][column_block] = value;
            all_compute = all_compute && value == 'C';
            has_loaded_idle = has_loaded_idle || value == 'L';
        }
    }

    if (all_compute) {
        if (!summary.in_all_compute) {
            summary.in_all_compute = true;
            summary.all_compute_start = cycle;
        }
        summary.all_compute_last = cycle;
        return;
    }

    flush_all_compute_summary(os, label, summary);
    if (!has_loaded_idle) {
        return;
    }

    os << label << " array_state cycle " << cycle << '\n';
    for (std::size_t tile = kLogTile; tile <= kLogTile; ++tile) {
        os << "  row " << tile << ": ";
        for (std::size_t column_block = 0; column_block < kBlocks; ++column_block) {
            os << state[tile][column_block];
        }
        os << '\n';
    }
}

std::vector<MxmOutputEvent> capture_mxm_outputs(
    const ftlpu::Mxm& mxm,
    MatrixI32& output_matrix)
{
    std::vector<MxmOutputEvent> events;
    for (const auto& output : mxm.last_outputs()) {
        for (std::size_t lane = 0; lane < kLanes; ++lane) {
            const auto column = output.column_block * kLanes + lane;
            const auto value = output.values[lane];
            output_matrix[matrix_index(output.row, column)] = value;
        }
        events.push_back(MxmOutputEvent {output.row, output.column_block});
    }
    return events;
}

std::int32_t expected_projection(
    std::size_t matrix_id,
    std::size_t row,
    std::size_t column,
    std::size_t pass)
{
    std::int32_t sum = 0;
    for (std::size_t k = 0; k < kKPerPass; ++k) {
        const auto weight_k = matrix_id == kDownMatrix ? pass * kKPerPass + k : k;
        const auto weight_column = matrix_id == kDownMatrix ? column : pass * kColumns + column;
        sum += static_cast<std::int32_t>(activation_value(row, k))
            * static_cast<std::int32_t>(weight_value(matrix_id, weight_k, weight_column));
    }
    return sum;
}

std::int8_t expected_swiglu(
    std::int32_t gate,
    std::int32_t up,
    const ftlpu::VxmLane::SwigluParams& params)
{
    const auto gate_fp32 = static_cast<float>(gate) * params.gate_scale;
    const auto up_fp32 = static_cast<float>(up) * params.up_scale;
    const auto sigmoid = 1.0f / (1.0f + std::exp(-gate_fp32));
    return ftlpu::VxmAlu::quantize_scalar(gate_fp32 * sigmoid * up_fp32, params.output_scale, params.output_zero_point);
}

std::int32_t expected_down_partial(
    const MatrixI8& swiglu,
    std::size_t pass,
    std::size_t row,
    std::size_t column)
{
    std::int32_t sum = 0;
    for (std::size_t k = 0; k < kKPerPass; ++k) {
        sum += static_cast<std::int32_t>(swiglu[hidden_matrix_index(row, pass * kKPerPass + k)])
            * static_cast<std::int32_t>(weight_value(kDownMatrix, pass * kKPerPass + k, column));
    }
    return sum;
}

std::size_t count_nonzero(const MatrixI8& matrix)
{
    return static_cast<std::size_t>(
        std::count_if(matrix.begin(), matrix.end(), [](std::int8_t value) {
            return value != 0;
        }));
}

template <typename Instruction>
struct ScheduledInstruction {
    std::size_t cycle{0};
    Instruction instruction{};
};

class OfflineIcuProgram {
public:
    void emit_mem(std::size_t cycle, std::size_t column, ftlpu::MemInstruction instruction)
    {
        mem_[column].push_back(ScheduledInstruction<ftlpu::MemInstruction> {cycle, instruction});
        last_cycle_ = std::max(last_cycle_, cycle);
    }

    void emit_mxm(std::size_t cycle, std::size_t mxm, ftlpu::MxmControlInstruction instruction)
    {
        if (instruction.opcode == ftlpu::MxmControlOpcode::Compute) {
            mxm_compute_[mxm].push_back(ScheduledInstruction<ftlpu::MxmControlInstruction> {cycle, instruction});
        } else {
            mxm_load_[mxm].push_back(ScheduledInstruction<ftlpu::MxmControlInstruction> {cycle, instruction});
        }
        last_cycle_ = std::max(last_cycle_, cycle);
    }

    void emit_vxm(std::size_t cycle, std::size_t alu, ftlpu::VxmLaneAluInstruction instruction)
    {
        vxm_[alu].push_back(ScheduledInstruction<ftlpu::VxmLaneAluInstruction> {cycle, instruction});
        last_cycle_ = std::max(last_cycle_, cycle);
    }

    void load_into(ftlpu::InstructionControlUnit& icu)
    {
        for (std::size_t column = 0; column < mem_.size(); ++column) {
            load_queue(mem_[column], "mem" + std::to_string(column), [&](std::size_t nop) { icu.enqueue_mem_nop(column, nop); }, [&](auto instruction) {
                icu.enqueue_mem(column, instruction);
            });
        }
        for (std::size_t mxm = 0; mxm < mxm_load_.size(); ++mxm) {
            load_queue(
                mxm_load_[mxm],
                "mxm" + std::to_string(mxm) + ".load",
                [&](std::size_t nop) { icu.enqueue_mxm_load_nop(mxm, nop); },
                [&](auto instruction) { icu.enqueue_mxm(mxm, instruction); });
            load_queue(
                mxm_compute_[mxm],
                "mxm" + std::to_string(mxm) + ".compute",
                [&](std::size_t nop) { icu.enqueue_mxm_compute_nop(mxm, nop); },
                [&](auto instruction) { icu.enqueue_mxm(mxm, instruction); });
        }
        for (std::size_t alu = 0; alu < vxm_.size(); ++alu) {
            load_queue(vxm_[alu], "vxm" + std::to_string(alu), [&](std::size_t nop) { icu.enqueue_vxm_nop(alu, nop); }, [&](auto instruction) {
                icu.enqueue_vxm(alu, instruction);
            });
        }
    }

    std::size_t last_cycle() const
    {
        return last_cycle_;
    }

private:
    template <typename Instruction, typename NopFn, typename EmitFn>
    static void load_queue(std::vector<ScheduledInstruction<Instruction>>& events, const std::string& name, NopFn nop, EmitFn emit)
    {
        std::sort(events.begin(), events.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.cycle < rhs.cycle;
        });
        auto cursor = std::size_t {0};
        for (const auto& event : events) {
            if (event.cycle < cursor) {
                std::ostringstream os;
                os << "offline ICU program has two instructions in one queue cycle"
                   << " queue=" << name
                   << " event_cycle=" << event.cycle
                   << " cursor=" << cursor;
                throw std::logic_error(os.str());
            }
            nop(event.cycle - cursor);
            emit(event.instruction);
            cursor = event.cycle + 1;
        }
    }

    std::array<std::vector<ScheduledInstruction<ftlpu::MemInstruction>>, ftlpu::InstructionControlUnit::kMemQueues> mem_{};
    std::array<std::vector<ScheduledInstruction<ftlpu::MxmControlInstruction>>, ftlpu::InstructionControlUnit::kMxmUnitCount> mxm_load_{};
    std::array<std::vector<ScheduledInstruction<ftlpu::MxmControlInstruction>>, ftlpu::InstructionControlUnit::kMxmUnitCount> mxm_compute_{};
    std::array<std::vector<ScheduledInstruction<ftlpu::VxmLaneAluInstruction>>, ftlpu::InstructionControlUnit::kVxmQueues> vxm_{};
    std::size_t last_cycle_{0};
};

enum class OfflinePostOp {
    Swiglu,
    AddQuant,
};

struct OfflineComputePhase {
    std::size_t start{0};
    std::size_t output_start{0};
    std::size_t activation_column0{0};
    std::size_t activation_pass0{0};
    std::size_t activation_column1{0};
    std::size_t activation_pass1{0};
    std::size_t mxm0_activation_stream_base{0};
    std::size_t mxm1_activation_stream_base{0};
    OfflinePostOp post_op{OfflinePostOp::Swiglu};
    std::size_t vxm_latency{0};
    std::size_t mem_column{0};
    std::size_t output_pass{0};
    std::size_t weight_buffer{0};
    int vxm_start_adjust{0};
    bool stagger_shared_activation_stream{false};
    MatrixI8* output_i8{nullptr};
    MatrixI32* mxm0_output{nullptr};
    MatrixI32* mxm1_output{nullptr};
};

std::size_t vxm_post_op_start_offset();
std::size_t phase_vxm_post_op_start_offset(const OfflineComputePhase& phase);
std::size_t mxm_output_cycles();

std::size_t shared_activation_stream_for_row(const OfflineComputePhase& phase, std::size_t row)
{
    if (!phase.stagger_shared_activation_stream) {
        return kActivationStream;
    }
    if (row < 15) {
        return kActivationEarlyStream;
    }
    if (row < 19) {
        return kActivationBridgeStream;
    }
    return kActivationStream;
}

void emit_offline_mem_read_sequence(
    OfflineIcuProgram& program,
    std::size_t cycle,
    std::size_t column,
    std::size_t address,
    std::size_t stream,
    std::size_t count,
    std::int64_t address_stride)
{
    for (std::size_t index = 0; index < count; ++index) {
        const auto offset = address_stride * static_cast<std::int64_t>(index);
        program.emit_mem(
            cycle + index,
            column,
            ftlpu::MemInstruction::Read(static_cast<std::size_t>(static_cast<std::int64_t>(address) + offset), stream));
    }
}

void emit_offline_mem_write_sequence(
    OfflineIcuProgram& program,
    std::size_t cycle,
    std::size_t column,
    std::size_t pass,
    std::size_t stream,
    std::size_t count)
{
    for (std::size_t row = 0; row < count; ++row) {
        program.emit_mem(
            cycle + row,
            column,
            ftlpu::MemInstruction::Write(swiglu_address(pass, row, 0), stream));
    }
}

void emit_offline_weight_iw_mxm(
    OfflineIcuProgram& program,
    std::size_t iw_start,
    std::size_t mxm,
    std::size_t weight_buffer,
    std::size_t matrix,
    std::size_t pass)
{
    const auto stream_base = mxm * kLoadStreams;
    for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
        const auto column = stream_base + stream;
        emit_offline_mem_read_sequence(
            program,
            iw_start - east_stream_cycles_to_sreg11(column) - 1,
            column,
            weight_address(matrix, pass, kBlocks - 1),
            stream_base + stream,
            kBlocks,
            -static_cast<std::int64_t>(kLanes));
    }

    for (std::size_t column_block = 0; column_block < kBlocks; ++column_block) {
        const auto handoff_cycle = iw_start + column_block;
        program.emit_mxm(handoff_cycle, mxm, ftlpu::MxmControlInstruction::IW(weight_buffer));
    }
}

void emit_offline_activation_reads(OfflineIcuProgram& program, const OfflineComputePhase& phase)
{
    const auto latency0 = east_stream_cycles_to_sreg11(phase.activation_column0);
    for (std::size_t row = 0; row < kActivationRows; ++row) {
        program.emit_mem(
            phase.start + row - latency0,
            phase.activation_column0,
            ftlpu::MemInstruction::Read(
                activation_read_address(phase.activation_column0, phase.activation_pass0, row),
                shared_activation_stream_for_row(phase, row)));
    }

    if (phase.activation_column1 == phase.activation_column0 && phase.activation_pass1 == phase.activation_pass0) {
        return;
    }

    const auto latency1 = east_stream_cycles_to_sreg11(phase.activation_column1);
    emit_offline_mem_read_sequence(
        program,
        phase.start - latency1,
        phase.activation_column1,
        activation_read_address(phase.activation_column1, phase.activation_pass1, 0),
        kActivationStream1,
        kActivationRows,
        kLanes);
}

void emit_offline_compute_phase(
    OfflineIcuProgram& program,
    const OfflineComputePhase& phase,
    const ftlpu::VxmLane::SwigluParams& swiglu_params,
    const ftlpu::VxmLane::AddQuantParams& add_quant_params)
{
    emit_offline_activation_reads(program, phase);

    for (std::size_t row = 0; row < kActivationRows; ++row) {
        const auto compute_cycle = phase.start + row;
        const auto mxm0_stream = phase.stagger_shared_activation_stream
            ? shared_activation_stream_for_row(phase, row)
            : phase.mxm0_activation_stream_base;
        const auto mxm1_stream = phase.activation_column1 == phase.activation_column0
                && phase.activation_pass1 == phase.activation_pass0
            ? mxm0_stream
            : phase.mxm1_activation_stream_base;
        program.emit_mxm(
            compute_cycle,
            0,
            ftlpu::MxmControlInstruction::Compute(phase.weight_buffer, mxm0_stream, kGateWestStreamBase));
        program.emit_mxm(
            compute_cycle,
            1,
            ftlpu::MxmControlInstruction::Compute(phase.weight_buffer, mxm1_stream, kUpWestStreamBase));
    }

    for (std::size_t row = 0; row < kActivationRows; ++row) {
        const auto issue_start = phase.output_start + row + phase_vxm_post_op_start_offset(phase);
        for (std::size_t stage = 0; stage <= phase.vxm_latency; ++stage) {
            const auto cycle = issue_start + stage;
            auto issue = [&](std::size_t alu, ftlpu::VxmLaneAluInstruction instruction) {
                program.emit_vxm(cycle, alu, instruction);
            };
            if (phase.post_op == OfflinePostOp::Swiglu) {
                enqueue_swiglu_stage(issue, swiglu_params, stage, kGateStreamBase, kUpStreamBase, kSwigluOutputStream);
            } else {
                enqueue_add_quant_stage(issue, add_quant_params, stage, kLhsStreamBase, kRhsStreamBase, kSwigluOutputStream);
            }
        }
    }

    const auto vxm_write_latency = phase.mem_column / ftlpu::hw::kSlicesPerGroup + 2;
    emit_offline_mem_write_sequence(
        program,
        phase.output_start + phase_vxm_post_op_start_offset(phase) + phase.vxm_latency + vxm_write_latency,
        phase.mem_column,
        phase.output_pass,
        kSwigluOutputStream,
        kActivationRows);
}

struct PipelineBlock {
    std::size_t row{0};
    std::size_t start{0};
    std::size_t duration{0};
    std::string label{};
    std::string fill{};
};

struct PipelineDiagram {
    std::vector<PipelineBlock> blocks{};
    std::size_t total_cycles{0};
};

std::size_t load_phase_cycles()
{
    return kBlocks;
}

std::size_t compute_issue_cycles()
{
    return kActivationRows;
}

std::size_t compute_engine_ticks()
{
    return kActivationRows + 2 * kBlocks;
}

std::size_t mxm_output_cycles()
{
    return kActivationRows + kBlocks - 1;
}

std::size_t iw_issue_window_end(std::size_t iw_start)
{
    return iw_start + kBlocks;
}

struct DualMxmLoadSchedule {
    std::size_t mxm0_iw_start{0};
    std::size_t mxm1_iw_start{0};
    std::size_t done_cycle{0};

    std::size_t done() const
    {
        return done_cycle;
    }
};

std::size_t post_op_write_start(std::size_t gemm_start, std::size_t vxm_latency, std::size_t vxm_mem_column);

std::size_t tile0_weight_read_start(std::size_t iw_start, std::size_t mxm)
{
    const auto first_column = mxm * kLoadStreams;
    return iw_start - east_stream_cycles_to_sreg11(first_column) - 1;
}

std::size_t mxm1_e31_weight_read_first_cycle(std::size_t iw_start)
{
    constexpr auto kMxm1LastColumn = 2 * kLoadStreams - 1;
    return iw_start - east_stream_cycles_to_sreg11(kMxm1LastColumn) - 1;
}

std::size_t mxm1_e31_weight_read_last_cycle(std::size_t iw_start)
{
    return mxm1_e31_weight_read_first_cycle(iw_start) + kBlocks - 1;
}

std::size_t vxm_output_enters_mem_cycle(std::size_t gemm_start, std::size_t vxm_latency, std::size_t vxm_mem_column)
{
    const auto vxm_write_latency = vxm_mem_column / ftlpu::hw::kSlicesPerGroup + 2;
    return post_op_write_start(gemm_start, vxm_latency, vxm_mem_column) - vxm_write_latency;
}

std::size_t vxm_output_leaves_shared_stream_cycle(
    std::size_t gemm_start,
    std::size_t vxm_latency,
    std::size_t vxm_mem_column)
{
    return vxm_output_enters_mem_cycle(gemm_start, vxm_latency, vxm_mem_column)
        + kActivationRows
        + ftlpu::hw::kStreamRegisterColumns
        - 1;
}

std::size_t avoid_mxm1_e31_vxm_output_conflict(
    std::size_t candidate_iw_start,
    std::size_t previous_gemm_start,
    std::size_t vxm_latency,
    std::size_t vxm_mem_column)
{
    const auto output_first = vxm_output_enters_mem_cycle(previous_gemm_start, vxm_latency, vxm_mem_column);
    const auto output_last = vxm_output_leaves_shared_stream_cycle(previous_gemm_start, vxm_latency, vxm_mem_column);
    const auto read_first = mxm1_e31_weight_read_first_cycle(candidate_iw_start);
    const auto read_last = mxm1_e31_weight_read_last_cycle(candidate_iw_start);
    if (read_last < output_first || read_first > output_last) {
        return candidate_iw_start;
    }
    return output_last + 1 + east_stream_cycles_to_sreg11(2 * kLoadStreams - 1) + 1;
}

DualMxmLoadSchedule make_pingpong_load_schedule(
    std::size_t mxm0_iw_start,
    std::size_t mxm1_iw_start,
    std::size_t compute_done_cycle)
{
    const auto done_cycle = std::max({
        compute_done_cycle,
        iw_issue_window_end(mxm0_iw_start),
        iw_issue_window_end(mxm1_iw_start),
    });
    return DualMxmLoadSchedule {mxm0_iw_start, mxm1_iw_start, done_cycle};
}

DualMxmLoadSchedule initial_weight_load_schedule(std::size_t start)
{
    const auto iw_start = start + kWeightHandoffBaseCycle;
    return DualMxmLoadSchedule {iw_start, iw_start, iw_issue_window_end(iw_start)};
}

std::size_t pingpong_candidate_load_start(std::size_t gemm_start)
{
    return gemm_start;
}

DualMxmLoadSchedule pingpong_buffer_load_after_gemm_start(std::size_t gemm_start)
{
    const auto mxm0_iw_start = pingpong_candidate_load_start(gemm_start);
    const auto mxm1_iw_start = mxm0_iw_start + kBlocks;
    return make_pingpong_load_schedule(mxm0_iw_start, mxm1_iw_start, gemm_start);
}

std::size_t mxm0_weight_iw_after_activation_e0_clear(std::size_t gemm_start)
{
    return gemm_start + compute_issue_cycles() + 9;
}

DualMxmLoadSchedule down_weight_load_after_gemm_start(
    std::size_t gemm0_output_start,
    std::size_t gemm1_start,
    std::size_t gemm1_output_start)
{
#ifdef FTLPU_EARLY_MXM_COMPUTE_TEST
    auto mxm1_iw_start = pingpong_candidate_load_start(gemm1_start) + kBlocks;
    mxm1_iw_start = avoid_mxm1_e31_vxm_output_conflict(
        mxm1_iw_start,
        gemm0_output_start,
        kSwigluLatency,
        kSwigluMemColumn);
    mxm1_iw_start = avoid_mxm1_e31_vxm_output_conflict(
        mxm1_iw_start,
        gemm1_output_start,
        kSwigluLatency,
        kSwigluMemColumn1);

    const auto mxm0_iw_start = mxm0_weight_iw_after_activation_e0_clear(gemm1_start);
#else
    const auto mxm0_iw_start = mxm0_weight_iw_after_activation_e0_clear(gemm1_start);
    auto mxm1_iw_start = mxm0_iw_start + kBlocks;
    mxm1_iw_start = avoid_mxm1_e31_vxm_output_conflict(
        mxm1_iw_start,
        gemm0_output_start,
        kSwigluLatency,
        kSwigluMemColumn);
    mxm1_iw_start = avoid_mxm1_e31_vxm_output_conflict(
        mxm1_iw_start,
        gemm1_output_start,
        kSwigluLatency,
        kSwigluMemColumn1);
#endif
    return make_pingpong_load_schedule(mxm0_iw_start, mxm1_iw_start, gemm1_start);
}

std::size_t post_op_gemm_phase_cycles(std::size_t vxm_latency, std::size_t vxm_mem_column)
{
    const auto vxm_write_latency = vxm_mem_column / ftlpu::hw::kSlicesPerGroup + 2;
    return compute_issue_cycles()
        + ftlpu::hw::kStreamRegisterColumns
        + vxm_latency
        + vxm_write_latency
        + kBlocks
        + 8;
}

std::size_t vxm_post_op_start_offset()
{
    return ftlpu::hw::kStreamRegisterColumns;
}

std::size_t phase_vxm_post_op_start_offset(const OfflineComputePhase& phase)
{
    const auto base = static_cast<int>(vxm_post_op_start_offset()) + phase.vxm_start_adjust;
    if (base < 0) {
        throw std::logic_error("VXM phase start offset became negative");
    }
    return static_cast<std::size_t>(base);
}

std::size_t vxm_post_op_cycles(std::size_t vxm_latency, std::size_t vxm_mem_column)
{
    const auto vxm_write_latency = vxm_mem_column / ftlpu::hw::kSlicesPerGroup + 2;
    return kActivationRows + vxm_latency + vxm_write_latency + kBlocks;
}

std::size_t compute_queues_ready(std::size_t compute_start)
{
    return compute_start + compute_issue_cycles();
}

std::size_t scheduled_gemm1_compute_start(
    const DualMxmLoadSchedule& load1,
    std::size_t gemm0_start,
    std::size_t gemm0_output_start)
{
    (void)gemm0_output_start;
    const auto data_ready = std::max(load1.done(), compute_queues_ready(gemm0_start));
    return data_ready;
}

std::size_t post_op_write_start(std::size_t output_start, std::size_t vxm_latency, std::size_t vxm_mem_column)
{
    const auto vxm_write_latency = vxm_mem_column / ftlpu::hw::kSlicesPerGroup + 2;
    return output_start + vxm_post_op_start_offset() + vxm_latency + vxm_write_latency;
}

std::size_t swiglu_outputs_ready_for_down_gemm(std::size_t gemm1_output_start)
{
    const auto last_write_done = post_op_write_start(gemm1_output_start, kSwigluLatency, kSwigluMemColumn1) + kActivationRows;
    return last_write_done + east_stream_cycles_to_sreg11(kSwigluMemColumn1);
}

void add_block(
    PipelineDiagram& diagram,
    std::size_t row,
    std::size_t start,
    std::size_t duration,
    std::string label,
    std::string fill)
{
    diagram.blocks.push_back(PipelineBlock {
        row,
        start,
        duration,
        std::move(label),
        std::move(fill),
    });
    diagram.total_cycles = std::max(diagram.total_cycles, start + duration);
}

void add_load_phase(PipelineDiagram& diagram, std::size_t start, const std::string& label)
{
    const auto iw_start = start + kWeightHandoffBaseCycle;
    add_block(diagram, 0, tile0_weight_read_start(iw_start, 0), load_phase_cycles(), "W " + label + " M0", "#78b957");
    add_block(diagram, 0, tile0_weight_read_start(iw_start, 1), load_phase_cycles(), "W " + label + " M1", "#78b957");
    add_block(diagram, 3, start + kWeightHandoffBaseCycle, kBlocks, "IW " + label, "#e8e8e8");
    add_block(diagram, 5, start + kWeightHandoffBaseCycle, kBlocks, "IW " + label, "#e8e8e8");
}

void add_load_phase(PipelineDiagram& diagram, DualMxmLoadSchedule schedule, const std::string& label)
{
    if (schedule.mxm0_iw_start == schedule.mxm1_iw_start) {
        add_block(
            diagram,
            0,
            tile0_weight_read_start(schedule.mxm0_iw_start, 0),
            load_phase_cycles(),
            "W " + label + " M0",
            "#78b957");
        add_block(
            diagram,
            0,
            tile0_weight_read_start(schedule.mxm1_iw_start, 1),
            load_phase_cycles(),
            "W " + label + " M1",
            "#78b957");
        add_block(diagram, 3, schedule.mxm0_iw_start, kBlocks, "IW " + label, "#e8e8e8");
        add_block(diagram, 5, schedule.mxm1_iw_start, kBlocks, "IW " + label, "#e8e8e8");
        return;
    }
    add_block(
        diagram,
        0,
        tile0_weight_read_start(schedule.mxm0_iw_start, 0),
        load_phase_cycles(),
        "W " + label + " M0",
        "#78b957");
    add_block(
        diagram,
        0,
        tile0_weight_read_start(schedule.mxm1_iw_start, 1),
        load_phase_cycles(),
        "W " + label + " M1",
        "#78b957");
    add_block(
        diagram,
        5,
        schedule.mxm1_iw_start,
        kBlocks,
        "IW " + label + " M1",
        "#e8e8e8");
    add_block(
        diagram,
        3,
        schedule.mxm0_iw_start,
        kBlocks,
        "IW " + label + " M0",
        "#e8e8e8");
}

void add_gemm_phase(
    PipelineDiagram& diagram,
    std::size_t compute_start,
    std::size_t output_start,
    const std::string& mxm_label,
    const std::string& vxm_label,
    std::size_t vxm_latency,
    std::size_t vxm_mem_column)
{
    const auto total = post_op_gemm_phase_cycles(vxm_latency, vxm_mem_column);
    const auto vxm_start = output_start + vxm_post_op_start_offset();
    const auto vxm_write_latency = vxm_mem_column / ftlpu::hw::kSlicesPerGroup + 2;
    const auto mem_write_start = vxm_start + vxm_latency;
    add_block(diagram, 1, compute_start, kActivationRows, "A", "#d8ead2");
    add_block(diagram, 2, mem_write_start, kActivationRows + vxm_write_latency + kBlocks, "out", "#f5d28a");
    add_block(diagram, 4, compute_start, compute_issue_cycles(), mxm_label + " M0", "#eeeeee");
    add_block(diagram, 6, compute_start, compute_issue_cycles(), mxm_label + " M1", "#eeeeee");
    add_block(
        diagram,
        7,
        vxm_start,
        vxm_post_op_cycles(vxm_latency, vxm_mem_column),
        vxm_label,
        "#6f9fe8");
}

PipelineDiagram build_mem_dual_mxm_swiglu_diagram()
{
    PipelineDiagram diagram {};
    const auto load0 = initial_weight_load_schedule(0);
    const auto gemm0_start = load0.done();
    const auto gemm0_output_start = gemm0_start + kBlocks - 1;
    const auto load1 = pingpong_buffer_load_after_gemm_start(gemm0_start);
    const auto gemm1_start = scheduled_gemm1_compute_start(load1, gemm0_start, gemm0_output_start);
    const auto gemm1_output_start = gemm1_start + kBlocks - 1;
    const auto down_load = down_weight_load_after_gemm_start(gemm0_output_start, gemm1_start, gemm1_output_start);
    const auto down_gemm_start = std::max(down_load.done(), swiglu_outputs_ready_for_down_gemm(gemm1_output_start));
    const auto down_output_start = down_gemm_start + kBlocks - 1;

    add_load_phase(diagram, load0, "p0");
    add_gemm_phase(diagram, gemm0_start, gemm0_output_start, "GEMM p0", "SwiGLU p0", kSwigluLatency, kSwigluMemColumn);

    add_load_phase(diagram, load1, "p1");
    add_gemm_phase(diagram, gemm1_start, gemm1_output_start, "GEMM p1", "SwiGLU p1", kSwigluLatency, kSwigluMemColumn1);

    add_load_phase(diagram, down_load, "d");
    add_gemm_phase(diagram, down_gemm_start, down_output_start, "GEMM d", "add+q", kAddQuantLatency, kFinalMemColumn);

    return diagram;
}

std::string svg_escape(const std::string& text)
{
    std::string escaped;
    escaped.reserve(text.size());
    for (const auto ch : text) {
        switch (ch) {
        case '&':
            escaped += "&amp;";
            break;
        case '<':
            escaped += "&lt;";
            break;
        case '>':
            escaped += "&gt;";
            break;
        default:
            escaped.push_back(ch);
            break;
        }
    }
    return escaped;
}

void write_svg_lines(
    std::ostream& os,
    const std::string& label,
    double center_x,
    double center_y,
    double line_height,
    double font_size)
{
    std::size_t line_start = 0;
    std::vector<std::string> lines;
    while (line_start <= label.size()) {
        const auto line_end = label.find('\n', line_start);
        if (line_end == std::string::npos) {
            lines.push_back(label.substr(line_start));
            break;
        }
        lines.push_back(label.substr(line_start, line_end - line_start));
        line_start = line_end + 1;
    }

    const auto first_y = center_y - (static_cast<double>(lines.size() - 1) * line_height / 2.0);
    os << "<text x=\"" << center_x << "\" y=\"" << first_y
       << "\" text-anchor=\"middle\" dominant-baseline=\"middle\" font-size=\"" << font_size << "\">";
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (i == 0) {
            os << svg_escape(lines[i]);
        } else {
            os << "<tspan x=\"" << center_x << "\" dy=\"" << line_height << "\">"
               << svg_escape(lines[i]) << "</tspan>";
        }
    }
    os << "</text>\n";
}

void write_pipeline_svg(const std::filesystem::path& path)
{
    const auto diagram = build_mem_dual_mxm_swiglu_diagram();
    constexpr double kLeft = 150.0;
    constexpr double kRight = 38.0;
    constexpr double kTop = 58.0;
    constexpr double kRowGap = 54.0;
    constexpr double kBlockHeight = 36.0;
    constexpr double kWidth = 1280.0;
    constexpr double kHeight = 540.0;
    const auto scale = (kWidth - kLeft - kRight) / static_cast<double>(diagram.total_cycles);
    const char* row_labels[] {
        "MEM W read",
        "MEM A read",
        "MEM write",
        "MXM0 load",
        "MXM0 compute",
        "MXM1 load",
        "MXM1 compute",
        "VXM",
    };

    std::ofstream os(path);
    if (!os.good()) {
        throw std::runtime_error("failed to write pipeline SVG");
    }

    os << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << kWidth
       << "\" height=\"" << kHeight << "\" viewBox=\"0 0 " << kWidth << " " << kHeight << "\">\n";
    os << "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n";
    os << "<defs><marker id=\"arrow\" markerWidth=\"10\" markerHeight=\"8\" refX=\"9\" refY=\"4\" orient=\"auto\">"
          "<path d=\"M0,0 L10,4 L0,8 Z\" fill=\"#555\"/></marker></defs>\n";
    os << "<g font-family=\"Arial, Helvetica, sans-serif\" fill=\"#000\">\n";
    os << "<line x1=\"" << kLeft << "\" y1=\"28\" x2=\"" << (kWidth - kRight - 120)
       << "\" y2=\"28\" stroke=\"#555\" stroke-width=\"3\" stroke-dasharray=\"12 10\" marker-end=\"url(#arrow)\"/>\n";
    os << "<text x=\"" << (kWidth - kRight - 100) << "\" y=\"34\" font-size=\"28\">Time</text>\n";

    for (std::size_t row = 0; row < 8; ++row) {
        const auto y = kTop + row * kRowGap;
        os << "<text x=\"26\" y=\"" << (y + kBlockHeight / 2.0 + 6.0)
           << "\" font-size=\"18\">" << row_labels[row] << "</text>\n";
    }

    for (const auto& block : diagram.blocks) {
        const auto x = kLeft + static_cast<double>(block.start) * scale;
        const auto y = kTop + static_cast<double>(block.row) * kRowGap;
        const auto width = std::max(2.0, static_cast<double>(block.duration) * scale);
        os << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << width
           << "\" height=\"" << kBlockHeight << "\" fill=\"" << block.fill
           << "\" stroke=\"#111\" stroke-width=\"1.5\"/>\n";
        if (width >= 18.0) {
            write_svg_lines(os, block.label, x + width / 2.0, y + kBlockHeight / 2.0, 12.0, 12.0);
        }
    }

    os << "<text x=\"" << kLeft << "\" y=\"522\" font-size=\"16\" fill=\"#555\">0</text>\n";
    os << "<text x=\"" << (kWidth - kRight - 92) << "\" y=\"522\" font-size=\"16\" fill=\"#555\">"
       << diagram.total_cycles << " cycles</text>\n";
    os << "</g>\n</svg>\n";
}

int run_offline_icu_ffn_test()
{
    const ftlpu::VxmLane::SwigluParams swiglu_params {
        kProjectionInputScale,
        kProjectionInputScale,
        kSwigluOutputScale,
        0,
    };
    const ftlpu::VxmLane::AddQuantParams down_params {
        kDownPartialScale,
        kDownPartialScale,
        kFinalOutputScale,
        0,
    };

    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    const auto enable_logs = ffn_logs_enabled();
    const auto log_dir = std::filesystem::path("logs")
#ifdef FTLPU_EARLY_MXM_COMPUTE_TEST
        / "mem_dual_mxm_swiglu_early_compute_icu";
#else
        / "mem_dual_mxm_swiglu_offline_icu";
#endif
    if (enable_logs) {
        std::filesystem::create_directories(log_dir);
    }
    auto logs = TestLogs(log_dir, enable_logs);
    if (!logs.good()) {
        std::cerr << "failed to open " << log_dir.string() << " log files\n";
        return 1;
    }

    stage_weight_matrices(system->mem());
    stage_activation_matrix(system->mem());

    const auto load0 = initial_weight_load_schedule(0);
    const auto gemm0_start = load0.done();
    const auto gemm0_output_start = gemm0_start + kBlocks - 1;
    const auto load1 = pingpong_buffer_load_after_gemm_start(gemm0_start);
    const auto gemm1_start = scheduled_gemm1_compute_start(load1, gemm0_start, gemm0_output_start);
    const auto gemm1_output_start = gemm1_start + kBlocks - 1;
    const auto down_load = down_weight_load_after_gemm_start(gemm0_output_start, gemm1_start, gemm1_output_start);
    const auto down_gemm_start = std::max(down_load.done(), swiglu_outputs_ready_for_down_gemm(gemm1_output_start));
    const auto down_output_start = down_gemm_start + kBlocks - 1;

    auto swiglu = MatrixI8(kActivationRows * kHiddenColumns, 0);
    auto swiglu_chunk0 = MatrixI8(kActivationRows * kColumns, 0);
    auto swiglu_chunk1 = MatrixI8(kActivationRows * kColumns, 0);
    auto final = MatrixI8(kActivationRows * kColumns, 0);
    auto gate0 = MatrixI32(kActivationRows * kColumns, 0);
    auto up0 = MatrixI32(kActivationRows * kColumns, 0);
    auto gate1 = MatrixI32(kActivationRows * kColumns, 0);
    auto up1 = MatrixI32(kActivationRows * kColumns, 0);
    auto down0 = MatrixI32(kActivationRows * kColumns, 0);
    auto down1 = MatrixI32(kActivationRows * kColumns, 0);

    std::array<OfflineComputePhase, 3> phases {
        OfflineComputePhase {
            gemm0_start,
            gemm0_output_start,
            kActivationMemColumn,
            0,
            kActivationMemColumn,
            0,
            kActivationStream,
            kActivationStream,
            OfflinePostOp::Swiglu,
            kSwigluLatency,
            kSwigluMemColumn,
            0,
            0,
            0,
            true,
            &swiglu_chunk0,
            &gate0,
            &up0,
        },
        OfflineComputePhase {
            gemm1_start,
            gemm1_output_start,
            kActivationMemColumn,
            0,
            kActivationMemColumn,
            0,
            kActivationStream,
            kActivationStream,
            OfflinePostOp::Swiglu,
            kSwigluLatency,
            kSwigluMemColumn1,
            1,
            1,
            0,
            true,
            &swiglu_chunk1,
            &gate1,
            &up1,
        },
        OfflineComputePhase {
            down_gemm_start,
            down_output_start,
            kSwigluMemColumn,
            0,
            kSwigluMemColumn1,
            1,
            kActivationStream,
            kActivationStream1,
            OfflinePostOp::AddQuant,
            kAddQuantLatency,
            kFinalMemColumn,
            0,
            0,
            0,
            false,
            &final,
            &down0,
            &down1,
        },
    };

    auto program = OfflineIcuProgram {};
    emit_offline_weight_iw_mxm(program, load0.mxm0_iw_start, 0, phases[0].weight_buffer, kGateMatrix, 0);
    emit_offline_weight_iw_mxm(program, load0.mxm1_iw_start, 1, phases[0].weight_buffer, kUpMatrix, 0);
    emit_offline_compute_phase(program, phases[0], swiglu_params, down_params);
    emit_offline_weight_iw_mxm(program, load1.mxm1_iw_start, 1, phases[1].weight_buffer, kUpMatrix, 1);
    emit_offline_weight_iw_mxm(program, load1.mxm0_iw_start, 0, phases[1].weight_buffer, kGateMatrix, 1);
    emit_offline_compute_phase(program, phases[1], swiglu_params, down_params);
    emit_offline_weight_iw_mxm(program, down_load.mxm1_iw_start, 1, phases[2].weight_buffer, kDownMatrix, 1);
    emit_offline_weight_iw_mxm(program, down_load.mxm0_iw_start, 0, phases[2].weight_buffer, kDownMatrix, 0);
    emit_offline_compute_phase(program, phases[2], swiglu_params, down_params);
    program.load_into(system->icu());

    if (logs.enabled) {
        logs.icu << "offline ICU FFN program loaded before cycle 0\n";
#ifdef FTLPU_EARLY_MXM_COMPUTE_TEST
        logs.icu << "  schedule=early_mxm_compute p1 compute starts as soon as the compute queue is free\n";
#else
        logs.icu << "  schedule=baseline p1 compute uses conservative post-p0 spacing\n";
#endif
        logs.icu << "  load0.mxm0_iw=" << load0.mxm0_iw_start << " load0.mxm1_iw=" << load0.mxm1_iw_start
                 << " load0.done=" << load0.done()
                 << " gemm0=" << gemm0_start
                 << " gemm0_output=" << gemm0_output_start
                 << " load1.mxm1_iw=" << load1.mxm1_iw_start << " load1.mxm0_iw=" << load1.mxm0_iw_start
                 << " load1.done=" << load1.done()
                 << " gemm1=" << gemm1_start
                 << " gemm1_output=" << gemm1_output_start
                 << " down_load.mxm1_iw=" << down_load.mxm1_iw_start
                 << " down_load.mxm0_iw=" << down_load.mxm0_iw_start
                 << " down_load.done=" << down_load.done()
                 << " down_gemm=" << down_gemm_start
                 << " down_output=" << down_output_start << '\n';
        logs.icu << "  pingpong_candidate load1=" << pingpong_candidate_load_start(gemm0_start)
                 << " down_load=" << pingpong_candidate_load_start(gemm1_start)
                 << " dual_weight_buffers: IW fills one buffer while Compute consumes the other;"
                 << " VXM E" << kSwigluOutputStream << " writeback stream pressure can still delay MXM1 IW\n";
    }

    struct RuntimePhase {
        const OfflineComputePhase* config{nullptr};
        const char* name{nullptr};
        bool started{false};
        MxmArrayStateSummary mxm0_summary{};
        MxmArrayStateSummary mxm1_summary{};
        ftlpu::MxmPerformanceMonitor mxm0_perf{};
        ftlpu::MxmPerformanceMonitor mxm1_perf{};
    };

    std::array<RuntimePhase, 3> runtime_phases {
        RuntimePhase {&phases[0], "gate_up_p0"},
        RuntimePhase {&phases[1], "gate_up_p1"},
        RuntimePhase {&phases[2], "down"},
    };
    auto total_mxm0_perf = ftlpu::MxmPerformanceMonitor {};
    auto total_mxm1_perf = ftlpu::MxmPerformanceMonitor {};

    const auto final_write_latency = kFinalMemColumn / ftlpu::hw::kSlicesPerGroup + 2;
    const auto total_cycles = down_output_start
        + compute_issue_cycles()
        + vxm_post_op_start_offset()
        + kAddQuantLatency
        + final_write_latency
        + kBlocks
        + 8;
    const auto perf_total_cycles = down_gemm_start + compute_issue_cycles();

    for (std::size_t cycle = 0; cycle < total_cycles; ++cycle) {
        auto mxm0_sampled = false;
        auto mxm1_sampled = false;
        for (auto& phase : runtime_phases) {
            if (!phase.started && cycle == phase.config->start) {
                phase.started = true;
            }
        }

        system->dispatch_icu_only(logs.icu_ptr());
        const auto mxm_control_sinks = ftlpu::TspSliceSystem::LogSinks {
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            std::nullopt,
            kLogTile,
            std::nullopt,
        };
        system->tick_mxm_controls_only(mxm_control_sinks);
        if (logs.enabled) {
            system->mem().tick(logs.mem, kLogTile);
        } else {
            system->mem().tick();
        }
        system->tick_mxm_datapaths_only(ftlpu::TspSliceSystem::LogSinks {
            nullptr,
            nullptr,
            logs.mxm_ptr(),
            nullptr,
            nullptr,
            std::nullopt,
            kLogTile,
            std::nullopt,
        });

        for (auto& phase : runtime_phases) {
            if (!phase.started || cycle < phase.config->start) {
                continue;
            }
            const auto compute_cycle = cycle - phase.config->start;
            if (compute_cycle >= compute_engine_ticks()) {
                continue;
            }
            if (compute_cycle < compute_issue_cycles()) {
                phase.mxm0_perf.sample(system->mxm_unit(0));
                phase.mxm1_perf.sample(system->mxm_unit(1));
                if (cycle < perf_total_cycles) {
                    total_mxm0_perf.sample(system->mxm_unit(0));
                    total_mxm1_perf.sample(system->mxm_unit(1));
                    mxm0_sampled = true;
                    mxm1_sampled = true;
                }
            }
            if (cycle >= phase.config->output_start
                && cycle < phase.config->output_start + mxm_output_cycles()) {
                capture_mxm_outputs(system->mxm_unit(0), *phase.config->mxm0_output);
                capture_mxm_outputs(system->mxm_unit(1), *phase.config->mxm1_output);
            }
            if (logs.enabled) {
                log_mxm_array_state(
                    logs.mxm,
                    "offline_mxm0",
                    compute_cycle,
                    system->mxm_unit(0).control(),
                    system->mxm_unit(0),
                    phase.mxm0_summary);
                log_mxm_array_state(
                    logs.mxm,
                    "offline_mxm1",
                    compute_cycle,
                    system->mxm_unit(1).control(),
                    system->mxm_unit(1),
                    phase.mxm1_summary);
            }
        }
        if (cycle < perf_total_cycles && !mxm0_sampled) {
            total_mxm0_perf.sample_idle();
        }
        if (cycle < perf_total_cycles && !mxm1_sampled) {
            total_mxm1_perf.sample_idle();
        }

        const auto bridge_sinks = ftlpu::TspSliceSystem::LogSinks {
            nullptr,
            logs.mem_ptr(),
            nullptr,
            logs.vxm_ptr(),
            nullptr,
            kLogTile,
            std::nullopt,
            kLogTile,
        };
        system->tick_vxm_stream_bridge(bridge_sinks, 0);

        for (const auto& phase : phases) {
            const auto first_output = phase.output_start + phase_vxm_post_op_start_offset(phase) + phase.vxm_latency;
            if (cycle < first_output) {
                continue;
            }
            for (std::size_t tile = 0; tile < kBlocks; ++tile) {
                const auto row = cycle - first_output - tile;
                if (row >= kActivationRows) {
                    continue;
                }
                const auto& tile_output = system->vxm().output_at(tile);
                if (!tile_output.has_value()) {
                    continue;
                }
                for (std::size_t lane = 0; lane < kLanes; ++lane) {
                    (*phase.output_i8)[matrix_index(row, tile * kLanes + lane)] = tile_output->values[lane];
                }
            }
        }
    }

    if (logs.enabled) {
        for (auto& phase : runtime_phases) {
            const auto mxm0_label = std::string("offline_") + phase.name + "_mxm0";
            const auto mxm1_label = std::string("offline_") + phase.name + "_mxm1";
            flush_all_compute_summary(logs.mxm, mxm0_label.c_str(), phase.mxm0_summary);
            flush_all_compute_summary(logs.mxm, mxm1_label.c_str(), phase.mxm1_summary);
            phase.mxm0_perf.print(logs.mxm, mxm0_label);
            phase.mxm1_perf.print(logs.mxm, mxm1_label);
        }
        total_mxm0_perf.print(logs.mxm, "offline_total_mxm0");
        total_mxm1_perf.print(logs.mxm, "offline_total_mxm1");
    }

    if (!verify_loaded_weights(*system, 0, phases[1].weight_buffer, kGateMatrix, 1, "gate_p1")) {
        return 1;
    }
    if (!verify_loaded_weights(*system, 1, phases[1].weight_buffer, kUpMatrix, 1, "up_p1")) {
        return 1;
    }

    for (std::size_t row = 0; row < kActivationRows; ++row) {
        for (std::size_t column = 0; column < kColumns; ++column) {
            swiglu[hidden_matrix_index(row, column)] = swiglu_chunk0[matrix_index(row, column)];
            swiglu[hidden_matrix_index(row, kColumns + column)] = swiglu_chunk1[matrix_index(row, column)];
        }
    }

    auto reference_swiglu = MatrixI8(kActivationRows * kHiddenColumns, 0);
    for (std::size_t row = 0; row < kActivationRows; ++row) {
        for (std::size_t column = 0; column < kHiddenColumns; ++column) {
            const auto hidden_pass = column / kColumns;
            const auto local_column = column % kColumns;
            const auto gate = expected_projection(kGateMatrix, row, local_column, hidden_pass);
            const auto up = expected_projection(kUpMatrix, row, local_column, hidden_pass);
            const auto expected_hidden = expected_swiglu(gate, up, swiglu_params);
            reference_swiglu[hidden_matrix_index(row, column)] = expected_hidden;
            const auto hidden_tile = local_column / kLanes;
            const auto hidden_lane = column % kLanes;
            const auto hidden_mem_column = hidden_pass == 0 ? kSwigluMemColumn : kSwigluMemColumn1;
            const auto actual_hidden = static_cast<std::int8_t>(
                system->mem().sram_lane_byte(hidden_mem_column, hidden_tile, swiglu_address(hidden_pass, row, 0), hidden_lane));
            if (actual_hidden != expected_hidden) {
                std::cerr << "offline SwiGLU MEM output mismatch"
                          << " row=" << row
                          << " column=" << column
                          << " actual=" << static_cast<int>(actual_hidden)
                          << " expected=" << static_cast<int>(expected_hidden)
                          << " gate_actual=" << gate0[matrix_index(row, local_column)]
                          << " up_actual=" << up0[matrix_index(row, local_column)]
                          << " gate1_actual=" << gate1[matrix_index(row, local_column)]
                          << " up1_actual=" << up1[matrix_index(row, local_column)]
                          << " gate_expected=" << gate
                          << " up_expected=" << up
                          << '\n';
                return 1;
            }
        }

        for (std::size_t column = 0; column < kColumns; ++column) {
            const auto lhs = expected_down_partial(reference_swiglu, 0, row, column);
            const auto rhs = expected_down_partial(reference_swiglu, 1, row, column);
            const auto expected_final = ftlpu::VxmAlu::quantize_scalar(
                static_cast<float>(lhs) * down_params.lhs_scale + static_cast<float>(rhs) * down_params.rhs_scale,
                down_params.output_scale,
                down_params.output_zero_point);
            const auto actual_final = static_cast<std::int8_t>(
                system->mem().sram_lane_byte(kFinalMemColumn, column / kLanes, swiglu_address(0, row, 0), column % kLanes));
            if (actual_final != expected_final) {
                std::cerr << "offline down AddQuant MEM output mismatch"
                          << " row=" << row
                          << " column=" << column
                          << " actual=" << static_cast<int>(actual_final)
                          << " expected=" << static_cast<int>(expected_final)
                          << '\n';
                return 1;
            }
        }
    }

    const auto swiglu_nonzero = count_nonzero(reference_swiglu);
    const auto final_nonzero = count_nonzero(final);
    if (swiglu_nonzero < reference_swiglu.size() / 20 || final_nonzero < final.size() / 20) {
        std::cerr << "offline FFN output is unexpectedly sparse\n";
        return 1;
    }

    if (logs.enabled) {
        write_pipeline_svg(log_dir / "pipeline.svg");
        logs.icu << "pipeline diagram: " << (log_dir / "pipeline.svg").string() << '\n';
    }
    return 0;
}

} // namespace

#ifndef FTLPU_OFFLINE_ICU_FFN_TEST
int main()
try
{
    const ftlpu::VxmLane::SwigluParams swiglu_params {
        kProjectionInputScale,
        kProjectionInputScale,
        kSwigluOutputScale,
        0,
    };
    const ftlpu::VxmLane::AddQuantParams down_params {
        kDownPartialScale,
        kDownPartialScale,
        kFinalOutputScale,
        0,
    };

    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    const auto enable_logs = ffn_logs_enabled();
    const auto log_dir = std::filesystem::path("logs") / "mem_dual_mxm_swiglu";
    if (enable_logs) {
        std::filesystem::create_directories(log_dir);
    }
    auto logs = TestLogs(log_dir, enable_logs);
    if (!logs.good()) {
        std::cerr << "failed to open mem_dual_mxm_swiglu log files\n";
        return 1;
    }

    stage_weight_matrices(system->mem());
    stage_activation_matrix(system->mem());
    if (logs.enabled) {
        logs.mem << "mem initialized activation=160x320 gate/up/down weights=640x320\n";
        logs.mem << "  activation matrix column=" << kActivationMemColumn << '\n';
        logs.mem << "  swiglu output column=" << kSwigluMemColumn << '\n';
        logs.mem << "  final output column=" << kFinalMemColumn << '\n';
        logs.mem << "  weight matrices staged across MEM columns 0..31\n";
        logs.mem << "  external initialization writes SRAM directly; all movement between MEM/MXM/VXM uses ICU MEM Read/Write\n";
    }

    MatrixI8 swiglu(kActivationRows * kHiddenColumns, 0);

    for (std::size_t pass = 0; pass < kPasses; ++pass) {
        const auto activation_prefetch = ActivationReadConfig {
            kActivationMemColumn,
            0,
            kActivationStream,
            kActivationMemColumn,
            0,
            kActivationStream1,
            false,
        };
        load_dual_mxm_from_mem(*system, kGateMatrix, pass, kUpMatrix, pass, logs, &activation_prefetch);
        auto swiglu_chunk = MatrixI8(kActivationRows * kColumns, 0);
        auto swiglu_stream = SwigluStreamConfig {
            &swiglu_params,
            &swiglu_chunk,
            pass == 0 ? kSwigluMemColumn : kSwigluMemColumn1,
            pass,
            kSwigluOutputStream,
        };
        run_dual_mxm_gemm(*system, kActivationMemColumn, 0, kActivationMemColumn, 0, logs, &swiglu_stream, nullptr, true);
        const auto buffer = pass % ftlpu::MxmSupercell::kWeightBuffers;
        if (!verify_loaded_weights(*system, 0, buffer, kGateMatrix, pass, "gate")) {
            return 1;
        }
        if (!verify_loaded_weights(*system, 1, buffer, kUpMatrix, pass, "up")) {
            return 1;
        }
        for (std::size_t row = 0; row < kActivationRows; ++row) {
            for (std::size_t column = 0; column < kColumns; ++column) {
                swiglu[hidden_matrix_index(row, pass * kColumns + column)] = swiglu_chunk[matrix_index(row, column)];
            }
        }
    }

    const auto down_activation_prefetch = ActivationReadConfig {
        kSwigluMemColumn,
        0,
        kActivationStream,
        kSwigluMemColumn1,
        1,
        kActivationStream1,
        true,
    };
    load_dual_mxm_from_mem(*system, kDownMatrix, 0, kDownMatrix, 1, logs, &down_activation_prefetch);
    auto final = MatrixI8(kActivationRows * kColumns, 0);
    auto down_add_quant_stream = AddQuantStreamConfig {
        &down_params,
        &final,
        kFinalMemColumn,
        0,
        kSwigluOutputStream,
    };
    const auto down_partials = run_dual_mxm_gemm(
        *system,
        kSwigluMemColumn,
        0,
        kSwigluMemColumn1,
        1,
        logs,
        nullptr,
        &down_add_quant_stream,
        true);
    if (!verify_loaded_weights(*system, 0, 0, kDownMatrix, 0, "down0")) {
        return 1;
    }
    if (!verify_loaded_weights(*system, 1, 0, kDownMatrix, 1, "down1")) {
        return 1;
    }

    auto reference_swiglu = MatrixI8(kActivationRows * kHiddenColumns, 0);
    for (std::size_t row = 0; row < kActivationRows; ++row) {
        for (std::size_t column = 0; column < kHiddenColumns; ++column) {
            const auto hidden_pass = column / kColumns;
            const auto local_column = column % kColumns;
            const auto gate = expected_projection(kGateMatrix, row, local_column, hidden_pass);
            const auto up = expected_projection(kUpMatrix, row, local_column, hidden_pass);
            const auto expected_hidden = expected_swiglu(gate, up, swiglu_params);
            reference_swiglu[hidden_matrix_index(row, column)] = expected_hidden;
            const auto hidden_tile = local_column / kLanes;
            const auto hidden_lane = column % kLanes;
            const auto hidden_mem_column = hidden_pass == 0 ? kSwigluMemColumn : kSwigluMemColumn1;
            const auto actual_hidden = static_cast<std::int8_t>(
                system->mem().sram_lane_byte(hidden_mem_column, hidden_tile, swiglu_address(hidden_pass, row, 0), hidden_lane));
            if (actual_hidden != expected_hidden) {
                std::cerr << "SwiGLU MEM output mismatch"
                          << " row=" << row
                          << " column=" << column
                          << " hidden_pass=" << hidden_pass
                          << " local_column=" << local_column
                          << " tile=" << hidden_tile
                          << " lane=" << hidden_lane
                          << " actual=" << static_cast<int>(actual_hidden)
                          << " expected=" << static_cast<int>(expected_hidden)
                          << '\n';
                return 1;
            }
        }

        for (std::size_t column = 0; column < kColumns; ++column) {
            const auto lhs = expected_down_partial(reference_swiglu, 0, row, column);
            const auto rhs = expected_down_partial(reference_swiglu, 1, row, column);
            const auto expected_final = ftlpu::VxmAlu::quantize_scalar(
                static_cast<float>(lhs) * down_params.lhs_scale + static_cast<float>(rhs) * down_params.rhs_scale,
                down_params.output_scale,
                down_params.output_zero_point);
            const auto actual_final = static_cast<std::int8_t>(
                system->mem().sram_lane_byte(kFinalMemColumn, column / kLanes, swiglu_address(0, row, 0), column % kLanes));
            if (actual_final != expected_final) {
                const auto actual_lhs = down_partials[0][matrix_index(row, column)];
                const auto actual_rhs = down_partials[1][matrix_index(row, column)];
                std::cerr << "down AddQuant MEM output mismatch"
                          << " row=" << row
                          << " column=" << column
                          << " lhs=" << lhs
                          << " rhs=" << rhs
                          << " actual_lhs=" << actual_lhs
                          << " actual_rhs=" << actual_rhs
                          << " actual=" << static_cast<int>(actual_final)
                          << " expected=" << static_cast<int>(expected_final)
                          << '\n';
                return 1;
            }
        }
    }

    const auto swiglu_nonzero = count_nonzero(reference_swiglu);
    const auto final_nonzero = count_nonzero(final);
    if (logs.enabled) {
        logs.vxm << "symmetric_quant swiglu_scale=" << swiglu_params.output_scale
                 << " final_scale=" << down_params.output_scale
                 << " swiglu_nonzero=" << swiglu_nonzero << "/" << reference_swiglu.size()
                 << " final_nonzero=" << final_nonzero << "/" << final.size() << '\n';
    }
    if (swiglu_nonzero < reference_swiglu.size() / 20) {
        std::cerr << "SwiGLU output is too sparse after symmetric quantization: "
                  << swiglu_nonzero << "/" << reference_swiglu.size() << " nonzero\n";
        return 1;
    }
    if (final_nonzero < final.size() / 20) {
        std::cerr << "final output is too sparse after symmetric quantization: "
                  << final_nonzero << "/" << final.size() << " nonzero\n";
        return 1;
    }

    if (logs.enabled) {
        write_pipeline_svg(log_dir / "pipeline.svg");
        logs.icu << "pipeline diagram: " << (log_dir / "pipeline.svg").string() << '\n';
    }

    return 0;
}
catch (const std::exception& ex) {
    std::cerr << "mem_dual_mxm_swiglu_test failed: " << ex.what() << '\n';
    return 1;
}
#else
int main()
try
{
    return run_offline_icu_ffn_test();
}
catch (const std::exception& ex) {
#ifdef FTLPU_EARLY_MXM_COMPUTE_TEST
    std::cerr << "mem_dual_mxm_swiglu_early_compute_icu_test failed: " << ex.what() << '\n';
#else
    std::cerr << "mem_dual_mxm_swiglu_offline_icu_test failed: " << ex.what() << '\n';
#endif
    return 1;
}
#endif
