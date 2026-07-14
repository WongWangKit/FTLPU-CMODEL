#include "ftlpu/system/tsp_slice_system.hpp"

#include <algorithm>
#include <array>
#include <cmath>
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
    explicit TestLogs(const std::filesystem::path& directory)
        : icu(directory / "icu.log")
        , mem(directory / "mem.log")
        , mxm(directory / "mxm.log")
        , vxm(directory / "vxm.log")
    {
    }

    bool good() const
    {
        return icu.good() && mem.good() && mxm.good() && vxm.good();
    }

    std::ofstream icu;
    std::ofstream mem;
    std::ofstream mxm;
    std::ofstream vxm;
};

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
    const ftlpu::MxmGemmEngine& gemm,
    MxmArrayStateSummary& summary);

std::vector<MxmOutputEvent> emit_completed_mxm_outputs_to_mem(
    ftlpu::TspSliceSystem& system,
    std::size_t mxm_id,
    const ftlpu::MxmGemmEngine& gemm,
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
    std::size_t matrix_id,
    std::size_t pass,
    const char* label)
{
    const std::array<std::size_t, 4> sample_tiles {0, 7, 13, 19};
    const std::array<std::size_t, 4> sample_columns {0, 31, 128, 319};
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
                const auto actual = system.mxm_unit(mxm_id).array().weight(tile, column_block, lane, local_column);
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

ftlpu::MxmGemmEngine::ActivationVector activation_input_from_streams(
    const ftlpu::TileArrayModel& mem,
    std::size_t tile,
    std::size_t stream)
{
    constexpr auto kTargetSreg = ftlpu::hw::kStreamRegisterColumns - 1;
    ftlpu::MxmGemmEngine::ActivationVector input {};
    for (std::size_t lane = 0; lane < kLanes; ++lane) {
        const auto& slot = mem.east_register(tile, lane, kTargetSreg, stream);
        if (!slot.has_value()) {
            throw std::logic_error("activation MEM read did not reach MXM handoff stream register");
        }
        input[lane] = ftlpu::MxmGemmEngine::ActivationWord {
            static_cast<std::int8_t>(slot->data),
            lane + 1 == kLanes,
        };
    }
    return input;
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
    icu.enqueue_mxm_repeat(mxm, count - 1);
}

void enqueue_mxm_output_sequence(
    ftlpu::InstructionControlUnit& icu,
    std::size_t mxm,
    std::size_t first_cycle,
    ftlpu::MxmControlInstruction instruction,
    std::size_t count)
{
    if (count == 0) {
        return;
    }
    icu.enqueue_mxm_output_nop(mxm, first_cycle);
    icu.enqueue_mxm(mxm, instruction);
    icu.enqueue_mxm_output_repeat(mxm, count - 1);
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
        &logs.icu,
        &logs.mem,
        &logs.mxm,
        &logs.vxm,
        nullptr,
        kLogTile,
        kLogTile,
        kLogTile,
    };
    constexpr auto kTotalCycles = kWeightHandoffBaseCycle + kBlocks;

    for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
        const auto mxm0_column = stream;
        const auto mxm1_column = kLoadStreams + stream;
        const auto mxm0_first = kWeightHandoffBaseCycle - east_stream_cycles_to_sreg11(mxm0_column) - 1;
        const auto mxm1_first = kWeightHandoffBaseCycle - east_stream_cycles_to_sreg11(mxm1_column) - 1;
        enqueue_mem_read_sequence(
            system.icu(),
            mxm0_column,
            mxm0_first,
            weight_address(matrix0, pass0, 0),
            stream,
            kBlocks,
            kLanes);
        enqueue_mem_read_sequence(
            system.icu(),
            mxm1_column,
            mxm1_first,
            weight_address(matrix1, pass1, 0),
            kLoadStreams + stream,
            kBlocks,
            kLanes);
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
        logs.mxm << "dual_mxm_load cycle " << cycle << '\n'
                 << "  mxm0 matrix=" << matrix0 << " pass=" << pass0 << " streams=E0..E15\n"
                 << "  mxm1 matrix=" << matrix1 << " pass=" << pass1 << " streams=E16..E31\n";

        for (std::size_t column_block = 0; column_block < kBlocks; ++column_block) {
            const auto handoff_cycle = kWeightHandoffBaseCycle + column_block;
            if (cycle == handoff_cycle) {
                system.icu().enqueue_mxm(0, ftlpu::MxmControlInstruction::IW(column_block));
                system.icu().enqueue_mxm(1, ftlpu::MxmControlInstruction::IW(column_block));
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
    auto mxm0_gemm = std::make_unique<ftlpu::MxmGemmEngine>(system.mxm_unit(0).array());
    auto mxm1_gemm = std::make_unique<ftlpu::MxmGemmEngine>(system.mxm_unit(1).array());
    mxm0_gemm->start_compute(kActivationRows);
    mxm1_gemm->start_compute(kActivationRows);
    NullBuffer null_buffer;
    std::ostream null_stream(&null_buffer);
    MxmArrayStateSummary mxm0_summary {};
    MxmArrayStateSummary mxm1_summary {};

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
        ftlpu::MxmControlInstruction::Compute(),
        kActivationRows);
    enqueue_mxm_sequence(
        system.icu(),
        1,
        activation_handoff_base,
        ftlpu::MxmControlInstruction::Compute(),
        kActivationRows);
    enqueue_mxm_output_sequence(
        system.icu(),
        0,
        activation_handoff_base + kBlocks - 1,
        ftlpu::MxmControlInstruction::Output(kGateWestStreamBase),
        kActivationRows + kBlocks - 1);
    enqueue_mxm_output_sequence(
        system.icu(),
        1,
        activation_handoff_base + kBlocks - 1,
        ftlpu::MxmControlInstruction::Output(kUpWestStreamBase),
        kActivationRows + kBlocks - 1);

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

        system.dispatch_icu_only(&logs.icu);
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
        system.mem().tick(logs.mem, kLogTile);

        auto compute_cycle = std::size_t {0};
        const auto in_compute_window = cycle >= activation_handoff_base
            && (compute_cycle = cycle - activation_handoff_base) < kComputeTicks;

        if (in_compute_window) {
            for (std::size_t tile = 0; tile < kBlocks; ++tile) {
                if (compute_cycle < tile || compute_cycle - tile >= kActivationRows) {
                    continue;
                }
                const auto activation0 = activation_input_from_streams(system.mem(), tile, kActivationStream);
                const auto activation1 = (activation_column1 != activation_column0 || activation_pass1 != activation_pass0)
                    ? activation_input_from_streams(system.mem(), tile, kActivationStream1)
                    : activation0;
                mxm0_gemm->set_activation_input(tile, activation0);
                mxm1_gemm->set_activation_input(tile, activation1);
            }

            mxm0_gemm->tick(null_stream, false, false);
            mxm1_gemm->tick(null_stream, false, false);
            const auto gate_events = emit_completed_mxm_outputs_to_mem(system, 0, *mxm0_gemm, outputs[0]);
            emit_completed_mxm_outputs_to_mem(system, 1, *mxm1_gemm, outputs[1]);
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
            log_mxm_array_state(
                logs.mxm,
                "mxm0",
                compute_cycle,
                system.mxm_unit(0).control(),
                *mxm0_gemm,
                mxm0_summary);
            log_mxm_array_state(
                logs.mxm,
                "mxm1",
                compute_cycle,
                system.mxm_unit(1).control(),
                *mxm1_gemm,
                mxm1_summary);
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
                &logs.mem,
                nullptr,
                &logs.vxm,
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
    flush_all_compute_summary(logs.mxm, "mxm0", mxm0_summary);
    flush_all_compute_summary(logs.mxm, "mxm1", mxm1_summary);

    logs.mxm << "dual_mxm_compute shared_activation_streams=E0..E15 rows=160 k=320\n";
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
    const ftlpu::MxmGemmEngine& gemm,
    std::size_t tile,
    std::size_t column_block)
{
    if (gemm.computing_cell(tile, column_block)) {
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
    const ftlpu::MxmGemmEngine& gemm,
    MxmArrayStateSummary& summary)
{
    std::array<std::array<char, kBlocks>, kBlocks> state {};
    bool all_compute = true;
    bool has_loaded_idle = false;
    for (std::size_t tile = 0; tile < kBlocks; ++tile) {
        for (std::size_t column_block = 0; column_block < kBlocks; ++column_block) {
            const auto value = mxm_state_char(control, gemm, tile, column_block);
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

std::vector<MxmOutputEvent> emit_completed_mxm_outputs_to_mem(
    ftlpu::TspSliceSystem& system,
    std::size_t mxm_id,
    const ftlpu::MxmGemmEngine& gemm,
    MatrixI32& output_matrix)
{
    std::vector<MxmOutputEvent> events;
    for (const auto& output : gemm.completed_column_outputs()) {
        const auto stream_base = system.mxm_unit(mxm_id).control().output_stream_base(output.column_block);
        if (!stream_base.has_value()) {
            continue;
        }

        for (std::size_t lane = 0; lane < kLanes; ++lane) {
            const auto column = output.column_block * kLanes + lane;
            const auto value = output.values[lane];
            output_matrix[matrix_index(output.row, column)] = value;
            const auto bytes = ftlpu::VxmLane::pack_int32(value);
            for (std::size_t byte = 0; byte < bytes.size(); ++byte) {
                system.mem().set_west_stream_input(
                    output.column_block,
                    lane,
                    *stream_base + byte,
                    ftlpu::TileArrayModel::DataWord {
                        bytes[byte],
                        lane + 1 == kLanes,
                    });
            }
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
        mxm_[mxm].push_back(ScheduledInstruction<ftlpu::MxmControlInstruction> {cycle, instruction});
        last_cycle_ = std::max(last_cycle_, cycle);
    }

    void emit_mxm_output(std::size_t cycle, std::size_t mxm, ftlpu::MxmControlInstruction instruction)
    {
        mxm_output_[mxm].push_back(ScheduledInstruction<ftlpu::MxmControlInstruction> {cycle, instruction});
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
            load_queue(mem_[column], [&](std::size_t nop) { icu.enqueue_mem_nop(column, nop); }, [&](auto instruction) {
                icu.enqueue_mem(column, instruction);
            });
        }
        for (std::size_t mxm = 0; mxm < mxm_.size(); ++mxm) {
            load_queue(mxm_[mxm], [&](std::size_t nop) { icu.enqueue_mxm_nop(mxm, nop); }, [&](auto instruction) {
                icu.enqueue_mxm(mxm, instruction);
            });
            load_queue(
                mxm_output_[mxm],
                [&](std::size_t nop) { icu.enqueue_mxm_output_nop(mxm, nop); },
                [&](auto instruction) { icu.enqueue_mxm(mxm, instruction); });
        }
        for (std::size_t alu = 0; alu < vxm_.size(); ++alu) {
            load_queue(vxm_[alu], [&](std::size_t nop) { icu.enqueue_vxm_nop(alu, nop); }, [&](auto instruction) {
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
    static void load_queue(std::vector<ScheduledInstruction<Instruction>>& events, NopFn nop, EmitFn emit)
    {
        std::sort(events.begin(), events.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.cycle < rhs.cycle;
        });
        auto cursor = std::size_t {0};
        for (const auto& event : events) {
            if (event.cycle < cursor) {
                throw std::logic_error("offline ICU program has two instructions in one queue cycle");
            }
            nop(event.cycle - cursor);
            emit(event.instruction);
            cursor = event.cycle + 1;
        }
    }

    std::array<std::vector<ScheduledInstruction<ftlpu::MemInstruction>>, ftlpu::InstructionControlUnit::kMemQueues> mem_{};
    std::array<std::vector<ScheduledInstruction<ftlpu::MxmControlInstruction>>, ftlpu::InstructionControlUnit::kMxmQueues> mxm_{};
    std::array<std::vector<ScheduledInstruction<ftlpu::MxmControlInstruction>>, ftlpu::InstructionControlUnit::kMxmQueues> mxm_output_{};
    std::array<std::vector<ScheduledInstruction<ftlpu::VxmLaneAluInstruction>>, ftlpu::InstructionControlUnit::kVxmQueues> vxm_{};
    std::size_t last_cycle_{0};
};

enum class OfflinePostOp {
    Swiglu,
    AddQuant,
};

struct OfflineComputePhase {
    std::size_t start{0};
    std::size_t activation_column0{0};
    std::size_t activation_pass0{0};
    std::size_t activation_column1{0};
    std::size_t activation_pass1{0};
    OfflinePostOp post_op{OfflinePostOp::Swiglu};
    std::size_t vxm_latency{0};
    std::size_t mem_column{0};
    std::size_t output_pass{0};
    MatrixI8* output_i8{nullptr};
    MatrixI32* mxm0_output{nullptr};
    MatrixI32* mxm1_output{nullptr};
};

void emit_offline_mem_read_sequence(
    OfflineIcuProgram& program,
    std::size_t cycle,
    std::size_t column,
    std::size_t address,
    std::size_t stream,
    std::size_t count,
    std::size_t address_stride)
{
    for (std::size_t index = 0; index < count; ++index) {
        program.emit_mem(
            cycle + index,
            column,
            ftlpu::MemInstruction::Read(address + index * address_stride, stream));
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

void emit_offline_weight_load(
    OfflineIcuProgram& program,
    std::size_t start,
    std::size_t matrix0,
    std::size_t pass0,
    std::size_t matrix1,
    std::size_t pass1)
{
    for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
        const auto mxm0_column = stream;
        const auto mxm1_column = kLoadStreams + stream;
        emit_offline_mem_read_sequence(
            program,
            start + kWeightHandoffBaseCycle - east_stream_cycles_to_sreg11(mxm0_column) - 1,
            mxm0_column,
            weight_address(matrix0, pass0, 0),
            stream,
            kBlocks,
            kLanes);
        emit_offline_mem_read_sequence(
            program,
            start + kWeightHandoffBaseCycle - east_stream_cycles_to_sreg11(mxm1_column) - 1,
            mxm1_column,
            weight_address(matrix1, pass1, 0),
            kLoadStreams + stream,
            kBlocks,
            kLanes);
    }

    for (std::size_t column_block = 0; column_block < kBlocks; ++column_block) {
        const auto handoff_cycle = start + kWeightHandoffBaseCycle + column_block;
        program.emit_mxm(handoff_cycle, 0, ftlpu::MxmControlInstruction::IW(column_block));
        program.emit_mxm(handoff_cycle, 1, ftlpu::MxmControlInstruction::IW(column_block));
    }
}

void emit_offline_compute_phase(
    OfflineIcuProgram& program,
    const OfflineComputePhase& phase,
    const ftlpu::VxmLane::SwigluParams& swiglu_params,
    const ftlpu::VxmLane::AddQuantParams& add_quant_params)
{
    const auto latency0 = east_stream_cycles_to_sreg11(phase.activation_column0);
    const auto latency1 = east_stream_cycles_to_sreg11(phase.activation_column1);
    emit_offline_mem_read_sequence(
        program,
        phase.start - latency0,
        phase.activation_column0,
        activation_read_address(phase.activation_column0, phase.activation_pass0, 0),
        kActivationStream,
        kActivationRows,
        kLanes);
    if (phase.activation_column1 != phase.activation_column0 || phase.activation_pass1 != phase.activation_pass0) {
        emit_offline_mem_read_sequence(
            program,
            phase.start - latency1,
            phase.activation_column1,
            activation_read_address(phase.activation_column1, phase.activation_pass1, 0),
            kActivationStream1,
            kActivationRows,
            kLanes);
    }

    for (std::size_t row = 0; row < kActivationRows; ++row) {
        const auto compute_cycle = phase.start + row;
        program.emit_mxm(compute_cycle, 0, ftlpu::MxmControlInstruction::Compute());
        program.emit_mxm(compute_cycle, 1, ftlpu::MxmControlInstruction::Compute());
    }
    for (std::size_t output = 0; output < kActivationRows + kBlocks - 1; ++output) {
        const auto output_cycle = phase.start + kBlocks - 1 + output;
        program.emit_mxm_output(output_cycle, 0, ftlpu::MxmControlInstruction::Output(kGateWestStreamBase));
        program.emit_mxm_output(output_cycle, 1, ftlpu::MxmControlInstruction::Output(kUpWestStreamBase));
    }

    for (std::size_t row = 0; row < kActivationRows; ++row) {
        const auto issue_start = phase.start + row + kBlocks - 1 + ftlpu::hw::kStreamRegisterColumns;
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
        phase.start + kBlocks - 1 + ftlpu::hw::kStreamRegisterColumns + phase.vxm_latency + vxm_write_latency,
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
    return kWeightHandoffBaseCycle + kBlocks;
}

std::size_t compute_ticks()
{
    return kActivationRows + 2 * kBlocks;
}

std::size_t next_weight_load_start_after_gemm_start(std::size_t gemm_start)
{
    return gemm_start + compute_ticks() + 1;
}

std::size_t post_op_gemm_phase_cycles(std::size_t vxm_latency, std::size_t vxm_mem_column)
{
    const auto vxm_write_latency = vxm_mem_column / ftlpu::hw::kSlicesPerGroup + 2;
    return compute_ticks()
        + ftlpu::hw::kStreamRegisterColumns
        + vxm_latency
        + vxm_write_latency
        + kBlocks
        + 8;
}

std::size_t vxm_post_op_start_offset()
{
    return kBlocks - 1 + ftlpu::hw::kStreamRegisterColumns;
}

std::size_t vxm_post_op_cycles(std::size_t vxm_latency, std::size_t vxm_mem_column)
{
    const auto vxm_write_latency = vxm_mem_column / ftlpu::hw::kSlicesPerGroup + 2;
    return kActivationRows + vxm_latency + vxm_write_latency + kBlocks;
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
    const auto duration = load_phase_cycles();
    add_block(diagram, 0, start, duration, "weight read\n" + label, "#78b957");
    add_block(diagram, 2, start + kWeightHandoffBaseCycle, kBlocks, "IW\n" + label, "#e8e8e8");
}

void add_gemm_phase(
    PipelineDiagram& diagram,
    std::size_t start,
    const std::string& mxm_label,
    const std::string& vxm_label,
    std::size_t vxm_latency,
    std::size_t vxm_mem_column)
{
    const auto total = post_op_gemm_phase_cycles(vxm_latency, vxm_mem_column);
    const auto vxm_start = start + vxm_post_op_start_offset();
    const auto vxm_write_latency = vxm_mem_column / ftlpu::hw::kSlicesPerGroup + 2;
    const auto mem_write_start = vxm_start + vxm_latency;
    add_block(diagram, 0, start, kActivationRows, "activation\nread", "#d8ead2");
    add_block(diagram, 1, mem_write_start, kActivationRows + vxm_write_latency + kBlocks, "result\nwrite", "#f5d28a");
    add_block(diagram, 2, start, compute_ticks(), mxm_label, "#eeeeee");
    add_block(
        diagram,
        3,
        vxm_start,
        vxm_post_op_cycles(vxm_latency, vxm_mem_column),
        vxm_label,
        "#6f9fe8");
}

PipelineDiagram build_mem_dual_mxm_swiglu_diagram()
{
    PipelineDiagram diagram {};
    const auto load0_start = std::size_t {0};
    const auto gemm0_start = load0_start + load_phase_cycles();
    const auto load1_start = next_weight_load_start_after_gemm_start(gemm0_start);
    const auto gemm1_start = load1_start + load_phase_cycles();
    const auto down_load_start = next_weight_load_start_after_gemm_start(gemm1_start);
    const auto down_gemm_start = down_load_start + load_phase_cycles();

    add_load_phase(diagram, load0_start, "gate/up p0");
    add_gemm_phase(diagram, gemm0_start, "GEMM gate/up\np0", "SwiGLU\np0", kSwigluLatency, kSwigluMemColumn);

    add_load_phase(diagram, load1_start, "gate/up p1");
    add_gemm_phase(diagram, gemm1_start, "GEMM gate/up\np1", "SwiGLU\np1", kSwigluLatency, kSwigluMemColumn1);

    add_load_phase(diagram, down_load_start, "down p0/p1");
    add_gemm_phase(diagram, down_gemm_start, "GEMM down\np0+p1", "add + quant", kAddQuantLatency, kFinalMemColumn);

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
    constexpr double kLeft = 118.0;
    constexpr double kRight = 38.0;
    constexpr double kTop = 58.0;
    constexpr double kRowGap = 72.0;
    constexpr double kBlockHeight = 48.0;
    constexpr double kWidth = 1280.0;
    constexpr double kHeight = 372.0;
    const auto scale = (kWidth - kLeft - kRight) / static_cast<double>(diagram.total_cycles);
    const char* row_labels[] {"MEM read", "MEM write", "MXM", "VXM"};

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

    for (std::size_t row = 0; row < 4; ++row) {
        const auto y = kTop + row * kRowGap;
        os << "<text x=\"42\" y=\"" << (y + kBlockHeight / 2.0 + 7.0)
           << "\" font-size=\"30\">" << row_labels[row] << "</text>\n";
    }

    for (const auto& block : diagram.blocks) {
        const auto x = kLeft + static_cast<double>(block.start) * scale;
        const auto y = kTop + static_cast<double>(block.row) * kRowGap;
        const auto width = std::max(2.0, static_cast<double>(block.duration) * scale);
        os << "<rect x=\"" << x << "\" y=\"" << y << "\" width=\"" << width
           << "\" height=\"" << kBlockHeight << "\" fill=\"" << block.fill
           << "\" stroke=\"#111\" stroke-width=\"1.5\"/>\n";
        write_svg_lines(os, block.label, x + width / 2.0, y + kBlockHeight / 2.0, 20.0, 24.0);
    }

    os << "<text x=\"" << kLeft << "\" y=\"356\" font-size=\"16\" fill=\"#555\">0</text>\n";
    os << "<text x=\"" << (kWidth - kRight - 92) << "\" y=\"356\" font-size=\"16\" fill=\"#555\">"
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
    const auto log_dir = std::filesystem::path("logs") / "mem_dual_mxm_swiglu_offline_icu";
    std::filesystem::create_directories(log_dir);
    auto logs = TestLogs(log_dir);
    if (!logs.good()) {
        std::cerr << "failed to open mem_dual_mxm_swiglu_offline_icu log files\n";
        return 1;
    }

    stage_weight_matrices(system->mem());
    stage_activation_matrix(system->mem());

    const auto load0_start = std::size_t {0};
    const auto gemm0_start = load0_start + load_phase_cycles();
    const auto load1_start = next_weight_load_start_after_gemm_start(gemm0_start);
    const auto gemm1_start = load1_start + load_phase_cycles();
    const auto down_load_start = next_weight_load_start_after_gemm_start(gemm1_start);
    const auto down_gemm_start = down_load_start + load_phase_cycles();

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
            kActivationMemColumn,
            0,
            kActivationMemColumn,
            0,
            OfflinePostOp::Swiglu,
            kSwigluLatency,
            kSwigluMemColumn,
            0,
            &swiglu_chunk0,
            &gate0,
            &up0,
        },
        OfflineComputePhase {
            gemm1_start,
            kActivationMemColumn,
            0,
            kActivationMemColumn,
            0,
            OfflinePostOp::Swiglu,
            kSwigluLatency,
            kSwigluMemColumn1,
            1,
            &swiglu_chunk1,
            &gate1,
            &up1,
        },
        OfflineComputePhase {
            down_gemm_start,
            kSwigluMemColumn,
            0,
            kSwigluMemColumn1,
            1,
            OfflinePostOp::AddQuant,
            kAddQuantLatency,
            kFinalMemColumn,
            0,
            &final,
            &down0,
            &down1,
        },
    };

    auto program = OfflineIcuProgram {};
    emit_offline_weight_load(program, load0_start, kGateMatrix, 0, kUpMatrix, 0);
    emit_offline_compute_phase(program, phases[0], swiglu_params, down_params);
    emit_offline_weight_load(program, load1_start, kGateMatrix, 1, kUpMatrix, 1);
    emit_offline_compute_phase(program, phases[1], swiglu_params, down_params);
    emit_offline_weight_load(program, down_load_start, kDownMatrix, 0, kDownMatrix, 1);
    emit_offline_compute_phase(program, phases[2], swiglu_params, down_params);
    program.load_into(system->icu());

    logs.icu << "offline ICU FFN program loaded before cycle 0\n";
    logs.icu << "  load0=" << load0_start << " gemm0=" << gemm0_start
             << " load1=" << load1_start << " gemm1=" << gemm1_start
             << " down_load=" << down_load_start << " down_gemm=" << down_gemm_start << '\n';

    struct RuntimePhase {
        const OfflineComputePhase* config{nullptr};
        std::unique_ptr<ftlpu::MxmGemmEngine> mxm0{};
        std::unique_ptr<ftlpu::MxmGemmEngine> mxm1{};
        bool started{false};
        MxmArrayStateSummary mxm0_summary{};
        MxmArrayStateSummary mxm1_summary{};
    };

    std::array<RuntimePhase, 3> runtime_phases {
        RuntimePhase {&phases[0]},
        RuntimePhase {&phases[1]},
        RuntimePhase {&phases[2]},
    };

    NullBuffer null_buffer;
    std::ostream null_stream(&null_buffer);
    const auto final_write_latency = kFinalMemColumn / ftlpu::hw::kSlicesPerGroup + 2;
    const auto total_cycles = down_gemm_start
        + compute_ticks()
        + ftlpu::hw::kStreamRegisterColumns
        + kAddQuantLatency
        + final_write_latency
        + kBlocks
        + 8;

    for (std::size_t cycle = 0; cycle < total_cycles; ++cycle) {
        for (auto& phase : runtime_phases) {
            if (!phase.started && cycle == phase.config->start) {
                phase.mxm0 = std::make_unique<ftlpu::MxmGemmEngine>(system->mxm_unit(0).array());
                phase.mxm1 = std::make_unique<ftlpu::MxmGemmEngine>(system->mxm_unit(1).array());
                phase.mxm0->start_compute(kActivationRows);
                phase.mxm1->start_compute(kActivationRows);
                phase.started = true;
            }
        }

        system->dispatch_icu_only(&logs.icu);
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
        system->mem().tick(logs.mem, kLogTile);

        for (auto& phase : runtime_phases) {
            if (!phase.started || cycle < phase.config->start) {
                continue;
            }
            const auto compute_cycle = cycle - phase.config->start;
            if (compute_cycle >= compute_ticks()) {
                continue;
            }
            for (std::size_t tile = 0; tile < kBlocks; ++tile) {
                if (compute_cycle < tile || compute_cycle - tile >= kActivationRows) {
                    continue;
                }
                const auto activation0 = activation_input_from_streams(system->mem(), tile, kActivationStream);
                const auto activation1 = (phase.config->activation_column1 != phase.config->activation_column0
                                             || phase.config->activation_pass1 != phase.config->activation_pass0)
                    ? activation_input_from_streams(system->mem(), tile, kActivationStream1)
                    : activation0;
                phase.mxm0->set_activation_input(tile, activation0);
                phase.mxm1->set_activation_input(tile, activation1);
            }

            phase.mxm0->tick(null_stream, false, false);
            phase.mxm1->tick(null_stream, false, false);
            emit_completed_mxm_outputs_to_mem(system.operator*(), 0, *phase.mxm0, *phase.config->mxm0_output);
            emit_completed_mxm_outputs_to_mem(system.operator*(), 1, *phase.mxm1, *phase.config->mxm1_output);
            log_mxm_array_state(
                logs.mxm,
                "offline_mxm0",
                compute_cycle,
                system->mxm_unit(0).control(),
                *phase.mxm0,
                phase.mxm0_summary);
            log_mxm_array_state(
                logs.mxm,
                "offline_mxm1",
                compute_cycle,
                system->mxm_unit(1).control(),
                *phase.mxm1,
                phase.mxm1_summary);
        }

        const auto bridge_sinks = ftlpu::TspSliceSystem::LogSinks {
            nullptr,
            &logs.mem,
            nullptr,
            &logs.vxm,
            nullptr,
            kLogTile,
            std::nullopt,
            kLogTile,
        };
        system->tick_vxm_stream_bridge(bridge_sinks, 0);

        for (const auto& phase : phases) {
            const auto first_output = phase.start + kBlocks - 1 + ftlpu::hw::kStreamRegisterColumns + phase.vxm_latency;
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

    for (auto& phase : runtime_phases) {
        flush_all_compute_summary(logs.mxm, "offline_mxm0", phase.mxm0_summary);
        flush_all_compute_summary(logs.mxm, "offline_mxm1", phase.mxm1_summary);
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

    write_pipeline_svg(log_dir / "pipeline.svg");
    logs.icu << "pipeline diagram: " << (log_dir / "pipeline.svg").string() << '\n';
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
    const auto log_dir = std::filesystem::path("logs") / "mem_dual_mxm_swiglu";
    std::filesystem::create_directories(log_dir);
    auto logs = TestLogs(log_dir);
    if (!logs.good()) {
        std::cerr << "failed to open mem_dual_mxm_swiglu log files\n";
        return 1;
    }

    stage_weight_matrices(system->mem());
    stage_activation_matrix(system->mem());
    logs.mem << "mem initialized activation=160x320 gate/up/down weights=640x320\n";
    logs.mem << "  activation matrix column=" << kActivationMemColumn << '\n';
    logs.mem << "  swiglu output column=" << kSwigluMemColumn << '\n';
    logs.mem << "  final output column=" << kFinalMemColumn << '\n';
    logs.mem << "  weight matrices staged across MEM columns 0..31\n";
    logs.mem << "  external initialization writes SRAM directly; all movement between MEM/MXM/VXM uses ICU MEM Read/Write\n";

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
        if (!verify_loaded_weights(*system, 0, kGateMatrix, pass, "gate")) {
            return 1;
        }
        if (!verify_loaded_weights(*system, 1, kUpMatrix, pass, "up")) {
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
    if (!verify_loaded_weights(*system, 0, kDownMatrix, 0, "down0")) {
        return 1;
    }
    if (!verify_loaded_weights(*system, 1, kDownMatrix, 1, "down1")) {
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
    logs.vxm << "symmetric_quant swiglu_scale=" << swiglu_params.output_scale
             << " final_scale=" << down_params.output_scale
             << " swiglu_nonzero=" << swiglu_nonzero << "/" << reference_swiglu.size()
             << " final_nonzero=" << final_nonzero << "/" << final.size() << '\n';
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

    write_pipeline_svg(log_dir / "pipeline.svg");
    logs.icu << "pipeline diagram: " << (log_dir / "pipeline.svg").string() << '\n';

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
    std::cerr << "mem_dual_mxm_swiglu_offline_icu_test failed: " << ex.what() << '\n';
    return 1;
}
#endif
