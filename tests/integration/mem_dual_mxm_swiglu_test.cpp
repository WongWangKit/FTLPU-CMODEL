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
#include <streambuf>
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
    const auto raw = static_cast<int>((matrix_id * 19 + k * 3 + n * 5) % 13);
    return static_cast<std::int8_t>(raw - 6);
}

std::int8_t activation_value(std::size_t m, std::size_t k)
{
    const auto raw = static_cast<int>((m * 7 + k * 3) % 9);
    return static_cast<std::int8_t>(raw - 4);
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
            mem.set_sram_byte(
                kActivationMemColumn,
                tile,
                vector_row_address(row, lane),
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
                            mem.set_sram_byte(
                                stream,
                                tile,
                                address + lane,
                                static_cast<std::uint8_t>(weight_value(matrix_id, global_k, column)));
                            mem.set_sram_byte(
                                kLoadStreams + stream,
                                tile,
                                address + lane,
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
    for (std::size_t cycle = 0; cycle < kTotalCycles; ++cycle) {
        logs.mxm << "dual_mxm_load cycle " << cycle << '\n'
                 << "  mxm0 matrix=" << matrix0 << " pass=" << pass0 << " streams=E0..E15\n"
                 << "  mxm1 matrix=" << matrix1 << " pass=" << pass1 << " streams=E16..E31\n";

        for (std::size_t column_block = 0; column_block < kBlocks; ++column_block) {
            const auto handoff_cycle = kWeightHandoffBaseCycle + column_block;
            for (std::size_t stream = 0; stream < kLoadStreams; ++stream) {
                const auto mxm0_column = stream;
                const auto mxm1_column = kLoadStreams + stream;
                const auto mxm0_issue = handoff_cycle - east_stream_cycles_to_sreg11(mxm0_column) - 1;
                const auto mxm1_issue = handoff_cycle - east_stream_cycles_to_sreg11(mxm1_column) - 1;
                if (cycle == mxm0_issue) {
                    system.icu().enqueue_mem(
                        mxm0_column,
                        ftlpu::MemInstruction::Read(weight_address(matrix0, pass0, column_block), stream));
                }
                if (cycle == mxm1_issue) {
                    system.icu().enqueue_mem(
                        mxm1_column,
                        ftlpu::MemInstruction::Read(weight_address(matrix1, pass1, column_block), kLoadStreams + stream));
                }
            }
            if (cycle == handoff_cycle) {
                system.icu().enqueue_mxm(0, ftlpu::MxmControlInstruction::IW(column_block));
                system.icu().enqueue_mxm(1, ftlpu::MxmControlInstruction::IW(column_block));
            }
        }

        if (activation_prefetch != nullptr) {
            const auto latency0 = east_stream_cycles_to_sreg11(activation_prefetch->column0);
            if (cycle + latency0 >= kTotalCycles) {
                const auto row = cycle + latency0 - kTotalCycles;
                if (row < kActivationRows) {
                    system.icu().enqueue_mem(
                        activation_prefetch->column0,
                        ftlpu::MemInstruction::Read(
                            activation_read_address(activation_prefetch->column0, activation_prefetch->pass0, row),
                            activation_prefetch->stream0));
                }
            }

            if (activation_prefetch->dual_stream) {
                const auto latency1 = east_stream_cycles_to_sreg11(activation_prefetch->column1);
                if (cycle + latency1 >= kTotalCycles) {
                    const auto row = cycle + latency1 - kTotalCycles;
                    if (row < kActivationRows) {
                        system.icu().enqueue_mem(
                            activation_prefetch->column1,
                            ftlpu::MemInstruction::Read(
                                activation_read_address(activation_prefetch->column1, activation_prefetch->pass1, row),
                                activation_prefetch->stream1));
                    }
                }
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

        if (cycle + activation_read_latency0 >= activation_handoff_base) {
            const auto row = cycle + activation_read_latency0 - activation_handoff_base;
            if (row < kActivationRows) {
                system.icu().enqueue_mem(
                    activation_column0,
                    ftlpu::MemInstruction::Read(
                        activation_read_address(activation_column0, activation_pass0, row),
                        kActivationStream));
            }
        }
        if (activation_column1 != activation_column0 || activation_pass1 != activation_pass0) {
            if (cycle + activation_read_latency1 >= activation_handoff_base) {
                const auto row = cycle + activation_read_latency1 - activation_handoff_base;
                if (row < kActivationRows) {
                    system.icu().enqueue_mem(
                        activation_column1,
                        ftlpu::MemInstruction::Read(
                            activation_read_address(activation_column1, activation_pass1, row),
                            kActivationStream1));
                }
            }
        }

        if (cycle >= activation_handoff_base && cycle - activation_handoff_base < kActivationRows) {
            system.icu().enqueue_mxm(0, ftlpu::MxmControlInstruction::Compute());
            system.icu().enqueue_mxm(1, ftlpu::MxmControlInstruction::Compute());
        }
        if (cycle >= activation_handoff_base) {
            const auto compute_cycle = cycle - activation_handoff_base;
            if (compute_cycle >= kBlocks - 1 && compute_cycle < kActivationRows + 2 * kBlocks - 2) {
                system.icu().enqueue_mxm(0, ftlpu::MxmControlInstruction::Output(kGateWestStreamBase));
                system.icu().enqueue_mxm(1, ftlpu::MxmControlInstruction::Output(kUpWestStreamBase));
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

ftlpu::VxmSlice::StreamMatrix add_quant_streams_for_tile(
    const std::array<MatrixI32, kMxmCount>& partials,
    std::size_t tile,
    std::size_t row)
{
    ftlpu::VxmSlice::StreamMatrix streams {};
    for (std::size_t lane = 0; lane < kLanes; ++lane) {
        const auto column = tile * kLanes + lane;
        const auto lhs = ftlpu::VxmLane::pack_int32(partials[0][matrix_index(row, column)]);
        const auto rhs = ftlpu::VxmLane::pack_int32(partials[1][matrix_index(row, column)]);
        for (std::size_t byte = 0; byte < 4; ++byte) {
            streams[lane][kLhsStreamBase + byte] = lhs[byte];
            streams[lane][kRhsStreamBase + byte] = rhs[byte];
        }
    }
    return streams;
}

MatrixI8 run_vxm_add_quant(
    const std::array<MatrixI32, kMxmCount>& partials,
    const ftlpu::VxmLane::AddQuantParams& params,
    TestLogs& logs)
{
    auto vxm = std::make_unique<ftlpu::VxmSlice>();
    auto icu = ftlpu::InstructionControlUnit {};
    MatrixI8 output(kActivationRows * kColumns, 0);

    for (std::size_t cycle = 0; cycle < kActivationRows + kBlocks + kAddQuantLatency; ++cycle) {
        for (std::size_t stage = 0; stage <= kAddQuantLatency; ++stage) {
            if (cycle < stage || cycle - stage >= kActivationRows) {
                continue;
            }
            enqueue_add_quant_stage(
                [&](std::size_t alu, ftlpu::VxmLaneAluInstruction instruction) {
                    icu.enqueue_vxm(alu, instruction);
                },
                params,
                stage,
                kLhsStreamBase,
                kRhsStreamBase,
                kOutputStream);
        }

        for (std::size_t tile = 0; tile < kBlocks; ++tile) {
            if (cycle >= tile && cycle - tile < kActivationRows) {
                vxm->set_stream_inputs(tile, add_quant_streams_for_tile(partials, tile, cycle - tile));
            }
        }
        icu.dispatch_vxm(*vxm, &logs.icu);
        vxm->tick(&logs.vxm, 0);

        for (std::size_t tile = 0; tile < kBlocks; ++tile) {
            const auto& tile_output = vxm->output_at(tile);
            if (!tile_output.has_value()) {
                continue;
            }
            const auto row = cycle - tile - kAddQuantLatency;
            if (row >= kActivationRows) {
                continue;
            }
            for (std::size_t lane = 0; lane < kLanes; ++lane) {
                output[matrix_index(row, tile * kLanes + lane)] = tile_output->values[lane];
            }
        }
    }

    logs.vxm << "vxm_add_quant lhs_streams=W0..W3 rhs_streams=W4..W7 out=E0\n";
    return output;
}

void store_int8_matrix_to_mem(
    ftlpu::TspSliceSystem& system,
    std::size_t column,
    std::size_t pass,
    const MatrixI8& matrix,
    TestLogs& logs)
{
    const auto write_latency = column / ftlpu::hw::kSlicesPerGroup + 1;
    const auto total_cycles = kActivationRows + kBlocks + write_latency + 1;
    for (std::size_t cycle = 0; cycle < total_cycles; ++cycle) {
        for (std::size_t tile = 0; tile < kBlocks; ++tile) {
            if (cycle < tile || cycle - tile >= kActivationRows) {
                continue;
            }
            const auto row = cycle - tile;
            for (std::size_t lane = 0; lane < kLanes; ++lane) {
                const auto matrix_column = tile * kLanes + lane;
                system.mem().set_east_stream_input(
                    tile,
                    lane,
                    kOutputStream,
                    ftlpu::TileArrayModel::DataWord {
                        static_cast<std::uint8_t>(matrix[matrix_index(row, matrix_column)]),
                        lane + 1 == kLanes,
                    });
            }
        }

        if (cycle >= write_latency) {
            const auto row = cycle - write_latency;
            if (row < kActivationRows) {
                system.icu().enqueue_mem(column, ftlpu::MemInstruction::Write(swiglu_address(pass, row, 0), kOutputStream));
            }
        }

        system.dispatch_icu_only(&logs.icu);
        system.mem().tick(logs.mem, kLogTile);
    }
    logs.mem << "stored int8 matrix column=" << column
             << " hidden_pass=" << pass
             << " rows=160 columns=320 via ICU MEM Write\n";
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

} // namespace

int main()
try
{
    const ftlpu::VxmLane::SwigluParams swiglu_params {
        0.001953125f,
        0.001953125f,
        0.00390625f,
        0,
    };
    const ftlpu::VxmLane::AddQuantParams down_params {
        0.00390625f,
        0.00390625f,
        0.125f,
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
                system->mem().sram_byte(hidden_mem_column, hidden_tile, swiglu_address(hidden_pass, row, hidden_lane)));
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
                system->mem().sram_byte(kFinalMemColumn, column / kLanes, swiglu_address(0, row, column % kLanes)));
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

    return 0;
}
catch (const std::exception& ex) {
    std::cerr << "mem_dual_mxm_swiglu_test failed: " << ex.what() << '\n';
    return 1;
}
