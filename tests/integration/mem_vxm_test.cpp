#include "ftlpu/system/tsp_slice_system.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace {

constexpr std::size_t kGateAddress = 128;
constexpr std::size_t kUpAddress = 256;
constexpr std::size_t kOutputAddress = 512;
constexpr std::size_t kGateColumnBase = 36;
constexpr std::size_t kUpColumnBase = 40;
constexpr std::size_t kOutputColumn = 0;
constexpr std::size_t kGateStreamBase = 32;
constexpr std::size_t kUpStreamBase = 36;
constexpr std::size_t kOutputStream = 0;
constexpr std::size_t kStoreIssueCycle = 22;
constexpr std::size_t kTotalCycles = kStoreIssueCycle + ftlpu::hw::kTileRows;

std::int32_t gate_value(std::size_t tile, std::size_t lane)
{
    return static_cast<std::int32_t>((tile * 5 + lane * 3) % 31) - 15;
}

std::int32_t up_value(std::size_t tile, std::size_t lane)
{
    return static_cast<std::int32_t>((tile * 7 + lane * 11) % 29) - 14;
}

std::int8_t expected_swiglu(
    std::int32_t gate,
    std::int32_t up,
    const ftlpu::VxmLane::SwigluParams& params)
{
    const auto gate_fp32 = static_cast<float>(gate) * params.gate_scale;
    const auto up_fp32 = static_cast<float>(up) * params.up_scale;
    const auto sigmoid = 1.0f / (1.0f + std::exp(-gate_fp32));
    const auto product = gate_fp32 * sigmoid * up_fp32;
    return ftlpu::VxmAlu::quantize_scalar(product, params.output_scale, params.output_zero_point);
}

bool require(bool condition, const std::string& message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

void enqueue_swiglu_program(
    ftlpu::InstructionControlUnit& icu,
    const ftlpu::VxmLane::SwigluParams& params,
    std::size_t gate_stream_base,
    std::size_t up_stream_base,
    std::size_t output_stream)
{
    icu.enqueue_vxm(0, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Cast,
        ftlpu::VxmLaneOperand::StreamInt32(gate_stream_base),
        ftlpu::VxmLaneOperand::Imm(0.0f),
        1.0f,
        0,
        ftlpu::VxmCastTarget::Float32,
    });
    icu.enqueue_vxm(1, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Cast,
        ftlpu::VxmLaneOperand::StreamInt32(up_stream_base),
        ftlpu::VxmLaneOperand::Imm(0.0f),
        1.0f,
        0,
        ftlpu::VxmCastTarget::Float32,
    });
    icu.enqueue_vxm(2, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Multiply,
        ftlpu::VxmLaneOperand::Alu(0),
        ftlpu::VxmLaneOperand::Imm(params.gate_scale),
    });
    icu.enqueue_vxm(3, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Multiply,
        ftlpu::VxmLaneOperand::Alu(1),
        ftlpu::VxmLaneOperand::Imm(params.up_scale),
    });
    icu.enqueue_vxm(4, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Multiply,
        ftlpu::VxmLaneOperand::Alu(2),
        ftlpu::VxmLaneOperand::Alu(3),
    });
    icu.enqueue_vxm(5, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Negate,
        ftlpu::VxmLaneOperand::Alu(2),
    });
    icu.enqueue_vxm(6, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Exp,
        ftlpu::VxmLaneOperand::Alu(5),
    });
    icu.enqueue_vxm(7, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Add,
        ftlpu::VxmLaneOperand::Alu(6),
        ftlpu::VxmLaneOperand::Imm(1.0f),
    });
    icu.enqueue_vxm(8, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Divide,
        ftlpu::VxmLaneOperand::Imm(1.0f),
        ftlpu::VxmLaneOperand::Alu(7),
    });
    icu.enqueue_vxm(9, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Pass,
        ftlpu::VxmLaneOperand::Alu(4),
    });
    icu.enqueue_vxm(10, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Pass,
        ftlpu::VxmLaneOperand::Alu(9),
    });
    icu.enqueue_vxm(11, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Pass,
        ftlpu::VxmLaneOperand::Alu(10),
    });
    icu.enqueue_vxm(12, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Multiply,
        ftlpu::VxmLaneOperand::Alu(11),
        ftlpu::VxmLaneOperand::Alu(8),
    });
    icu.enqueue_vxm(13, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Multiply,
        ftlpu::VxmLaneOperand::Alu(12),
        ftlpu::VxmLaneOperand::Imm(1.0f / params.output_scale),
    });
    icu.enqueue_vxm(14, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Add,
        ftlpu::VxmLaneOperand::Alu(13),
        ftlpu::VxmLaneOperand::Imm(static_cast<float>(params.output_zero_point)),
    });
    icu.enqueue_vxm(15, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Cast,
        ftlpu::VxmLaneOperand::Alu(14),
        ftlpu::VxmLaneOperand::Imm(0.0f),
        1.0f,
        0,
        ftlpu::VxmCastTarget::Int8,
        output_stream,
    });
}

void initialize_mem_inputs(ftlpu::TspSliceSystem& system)
{
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            const auto gate_bytes = ftlpu::VxmLane::pack_int32(gate_value(tile, lane));
            const auto up_bytes = ftlpu::VxmLane::pack_int32(up_value(tile, lane));
            for (std::size_t byte = 0; byte < 4; ++byte) {
                system.mem().set_sram_byte(kGateColumnBase + byte, tile, kGateAddress + lane, gate_bytes[byte]);
                system.mem().set_sram_byte(kUpColumnBase + byte, tile, kUpAddress + lane, up_bytes[byte]);
            }
        }
    }
}

void issue_up_reads(ftlpu::TspSliceSystem& system)
{
    for (std::size_t byte = 0; byte < 4; ++byte) {
        system.mem().enqueue_instruction(
            kUpColumnBase + byte,
            ftlpu::MemInstruction::Read(kUpAddress, kUpStreamBase + byte));
    }
}

void issue_gate_reads(ftlpu::TspSliceSystem& system)
{
    for (std::size_t byte = 0; byte < 4; ++byte) {
        system.mem().enqueue_instruction(
            kGateColumnBase + byte,
            ftlpu::MemInstruction::Read(kGateAddress, kGateStreamBase + byte));
    }
}

} // namespace

int main()
try
{
    const ftlpu::VxmLane::SwigluParams params {
        0.03125f,
        0.0625f,
        0.00390625f,
        0,
    };

    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    initialize_mem_inputs(*system);
    enqueue_swiglu_program(
        system->icu(),
        params,
        kGateStreamBase,
        kUpStreamBase,
        kOutputStream);

    std::cout << "mem_vxm_integration\n";
    std::cout << "  topology: VXM <- MEM -> MXM\n";
    std::cout << "  vxm instructions: 16 ALU queues gate_stream_base="
              << kGateStreamBase << " up_stream_base=" << kUpStreamBase
              << " output_stream=" << kOutputStream << '\n';
    std::cout << "  mem reads: gate columns " << kGateColumnBase << ".." << (kGateColumnBase + 3)
              << " addr=" << kGateAddress << " -> streams " << kGateStreamBase << ".." << (kGateStreamBase + 3)
              << ", up columns " << kUpColumnBase << ".." << (kUpColumnBase + 3)
              << " addr=" << kUpAddress << " -> streams " << kUpStreamBase << ".." << (kUpStreamBase + 3)
              << '\n';
    std::cout << "  mem store: column " << kOutputColumn << " addr=" << kOutputAddress
              << " <- stream " << kOutputStream << " issued at cycle " << kStoreIssueCycle << '\n';

    std::ostringstream log;
    for (std::size_t cycle = 0; cycle < kTotalCycles; ++cycle) {
        if (cycle == 0) {
            issue_up_reads(*system);
        }
        if (cycle == 1) {
            issue_gate_reads(*system);
        }
        if (cycle == kStoreIssueCycle) {
            system->mem().enqueue_instruction(kOutputColumn, ftlpu::MemInstruction::Write(kOutputAddress, kOutputStream));
        }

        system->tick(log);
    }

    std::array<std::size_t, 4> sample_tiles {0, 1, 10, 19};
    std::array<std::size_t, 4> sample_lanes {0, 5, 10, 15};
    std::cout << "  samples:\n";
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            const auto expected = expected_swiglu(gate_value(tile, lane), up_value(tile, lane), params);
            const auto actual = static_cast<std::int8_t>(
                system->mem().sram_byte(kOutputColumn, tile, kOutputAddress + lane));
            if (!require(
                    actual == expected,
                    "output mismatch tile=" + std::to_string(tile)
                        + " lane=" + std::to_string(lane)
                        + " actual=" + std::to_string(static_cast<int>(actual))
                        + " expected=" + std::to_string(static_cast<int>(expected)))) {
                return 1;
            }
        }
    }

    for (std::size_t index = 0; index < sample_tiles.size(); ++index) {
        const auto tile = sample_tiles[index];
        const auto lane = sample_lanes[index];
        const auto actual = static_cast<std::int8_t>(
            system->mem().sram_byte(kOutputColumn, tile, kOutputAddress + lane));
        const auto expected = expected_swiglu(gate_value(tile, lane), up_value(tile, lane), params);
        std::cout << "    tile=" << tile << " lane=" << lane
                  << " gate=" << gate_value(tile, lane)
                  << " up=" << up_value(tile, lane)
                  << " out=" << static_cast<int>(actual)
                  << " expected=" << static_cast<int>(expected) << '\n';
    }

    const auto text = log.str();
    if (!require(text.find("MEM.edge -> VXM tile 0") != std::string::npos, "missing tile0 MEM->VXM log")) {
        return 1;
    }
    if (!require(text.find("MEM.edge -> VXM tile 19") != std::string::npos, "missing tile19 MEM->VXM log")) {
        return 1;
    }
    if (!require(text.find("VXM -> MEM tile 0 stream 0") != std::string::npos, "missing tile0 VXM->MEM log")) {
        return 1;
    }
    if (!require(text.find("VXM -> MEM tile 19 stream 0") != std::string::npos, "missing tile19 VXM->MEM log")) {
        return 1;
    }
    if (!require(text.find("c0.t0=Write(a=512,s=0)") != std::string::npos, "missing tile0 MEM store instruction")) {
        return 1;
    }
    if (!require(text.find("c0.t19=Write(a=512,s=0)") != std::string::npos, "missing tile19 MEM store instruction")) {
        return 1;
    }

    const auto log_path = "mem_vxm_integration.log";
    std::ofstream log_file(log_path);
    if (!log_file) {
        std::cerr << "failed to open " << log_path << '\n';
        return 1;
    }
    log_file << text;
    std::cout << "  wrote detailed log: " << log_path << '\n';
    std::cout << "  result: PASS\n";

    return 0;
}
catch (const std::exception& ex) {
    std::cerr << "mem_vxm_test failed: " << ex.what() << '\n';
    return 1;
}
