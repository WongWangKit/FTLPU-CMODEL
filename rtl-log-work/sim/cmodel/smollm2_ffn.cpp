#include "rtl_schedule.hpp"

#include "ftlpu/core/bf16.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>

namespace {

using ftlpu::vmodel_test::RtlSchedule;
constexpr std::size_t kRows = 8;
constexpr std::size_t kHidden = 32;
constexpr std::size_t kIntermediate = 32;
constexpr std::size_t kBlocks = 4;
constexpr std::size_t kTiles = 4;
constexpr std::size_t kLanes = 8;
constexpr std::size_t kActivationAddress = 8;
constexpr std::size_t kGateAddress = 20;
constexpr std::size_t kUpAddress = 20;
constexpr std::size_t kFinalAddress = 30;

std::uint64_t read_word(
    const ftlpu::TspSliceSystem& system,
    std::size_t slice,
    std::size_t address,
    std::size_t tile)
{
    std::uint64_t value = 0;
    for (std::size_t lane = 0; lane < kLanes; ++lane)
        value |= static_cast<std::uint64_t>(
            system.read_mem_sram_lane_byte(
                ftlpu::Hemisphere::East, slice, tile, address, lane))
            << (lane * 8);
    return value;
}

float activation(std::size_t row, std::size_t column)
{
    return ftlpu::Bf16::from_float(
        static_cast<float>(static_cast<int>((row * 5 + column * 3) % 17) - 8)
        * 0.0625f).to_float();
}

std::int8_t gate_weight(std::size_t k, std::size_t n)
{
    return static_cast<std::int8_t>(
        static_cast<int>((k * 3 + n * 5 + 1) % 15) - 7);
}

std::int8_t up_weight(std::size_t k, std::size_t n)
{
    return static_cast<std::int8_t>(
        static_cast<int>((k * 7 + n * 2 + 4) % 13) - 6);
}

std::int8_t down_weight(std::size_t k, std::size_t n)
{
    return static_cast<std::int8_t>(
        static_cast<int>((k * 5 + n * 11 + 2) % 17) - 8);
}

void initialize(ftlpu::TspSliceSystem& system)
{
    for (std::size_t block = 0; block < kBlocks; ++block) {
        for (std::size_t stream = 0; stream < 8; ++stream) {
            for (std::size_t tile = 0; tile < kTiles; ++tile) {
                for (std::size_t lane = 0; lane < kLanes; ++lane) {
                    const auto k = tile * kLanes + lane;
                    const auto n = block * 8 + stream;
                    system.initialize_mem_sram_lane_byte(
                        ftlpu::Hemisphere::East, stream, tile, block, lane,
                        static_cast<std::uint8_t>(gate_weight(k, n)));
                    system.initialize_mem_sram_lane_byte(
                        ftlpu::Hemisphere::East, 8 + stream, tile, block, lane,
                        static_cast<std::uint8_t>(up_weight(k, n)));
                    system.initialize_mem_sram_lane_byte(
                        ftlpu::Hemisphere::East, stream, tile, 4 + block, lane,
                        static_cast<std::uint8_t>(down_weight(k, n)));
                }
            }
        }
    }
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t tile = 0; tile < kTiles; ++tile) {
            for (std::size_t lane = 0; lane < kLanes; ++lane) {
                const auto bits = ftlpu::Bf16::from_float(
                    activation(row, tile * kLanes + lane)).bits();
                for (std::size_t byte = 0; byte < 2; ++byte)
                    system.initialize_mem_sram_lane_byte(
                        ftlpu::Hemisphere::East, 32 + row * 2 + byte,
                        tile, kActivationAddress, lane,
                        static_cast<std::uint8_t>(bits >> (byte * 8)));
            }
        }
    }
}

void write_init(const char* path, const ftlpu::TspSliceSystem& system)
{
    auto output = std::ofstream(path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open FFN init output");
    output << std::hex << std::setfill('0');
    for (std::size_t block = 0; block < 4; ++block)
        for (std::size_t slice = 0; slice < 16; ++slice)
            for (std::size_t tile = 0; tile < 4; ++tile)
                output << std::setw(16)
                       << read_word(system, slice, block, tile) << '\n';
    for (std::size_t block = 0; block < 4; ++block)
        for (std::size_t slice = 0; slice < 8; ++slice)
            for (std::size_t tile = 0; tile < 4; ++tile)
                output << std::setw(16)
                       << read_word(system, slice, 4 + block, tile) << '\n';
    for (std::size_t slice = 32; slice < 48; ++slice)
        for (std::size_t tile = 0; tile < 4; ++tile)
            output << std::setw(16)
                   << read_word(system, slice, kActivationAddress, tile) << '\n';
}

std::size_t east_latency(std::size_t slice) { return 15 - slice / 4; }
std::size_t west_output_latency(std::size_t slice) { return 14 - slice / 4; }

RtlSchedule gate_up_schedule(ftlpu::TspSliceSystem& system)
{
    auto schedule = RtlSchedule(system);
    constexpr auto scale = 0.015625f;
    for (std::size_t block = 0; block < 4; ++block) {
        const auto load_cycle = 20 + block;
        for (std::size_t mxm = 0; mxm < 2; ++mxm) {
            for (std::size_t stream = 0; stream < 8; ++stream) {
                const auto slice = mxm * 8 + stream;
                schedule.mem_at(slice, load_cycle - east_latency(slice),
                    ftlpu::MemInstruction::Read(
                        block, ftlpu::StreamId::East(mxm * 8 + stream)));
            }
            schedule.mxm_dequant_at(mxm, load_cycle,
                ftlpu::MxmDequantInstruction::Scale(scale));
            schedule.mxm_load_at(mxm, load_cycle,
                ftlpu::MxmControlInstruction::IW(0, block));
        }
    }
    constexpr std::size_t compute_cycle = 28;
    for (std::size_t slice = 32; slice < 48; ++slice)
        schedule.mem_at(slice, compute_cycle - east_latency(slice),
            ftlpu::MemInstruction::Read(
                kActivationAddress, ftlpu::StreamId::East(slice - 32)));
    for (std::size_t mxm = 0; mxm < 2; ++mxm)
        schedule.mxm_compute_at(mxm, compute_cycle,
            ftlpu::MxmControlInstruction::Compute(
                0, 0, 0, 0, 1,
                ftlpu::MxmAccumulatorDestination::Sram,
                ftlpu::MxmDataFormat::BFloat16,
                ftlpu::MxmComputeMode::Block8));

    for (std::size_t row = 0; row < kRows; ++row) {
        const auto gate_read = 60 + row * 12;
        const auto up_read = gate_read + 6;
        schedule.mxm_compute_at(0, gate_read,
            ftlpu::MxmControlInstruction::AccumulatorRead(
                0, 0, false, ftlpu::MxmComputeMode::Block8));
        schedule.mxm_compute_at(1, up_read,
            ftlpu::MxmControlInstruction::AccumulatorRead(
                0, 0, false, ftlpu::MxmComputeMode::Block8));
        for (std::size_t byte = 0; byte < 4; ++byte) {
            schedule.mem_at(byte, gate_read + west_output_latency(byte),
                ftlpu::MemInstruction::Write(
                    kGateAddress + row,
                    ftlpu::StreamId::West(row * 4 + byte)));
            schedule.mem_at(4 + byte, up_read + west_output_latency(4 + byte),
                ftlpu::MemInstruction::Write(
                    kUpAddress + row,
                    ftlpu::StreamId::West(row * 4 + byte)));
        }
    }
    return schedule;
}

ftlpu::VxmLaneAluInstruction vxm_instruction(
    ftlpu::VxmAluOpcode opcode,
    ftlpu::VxmLaneOperand lhs,
    ftlpu::VxmLaneOperand rhs,
    ftlpu::VxmCastTarget cast = ftlpu::VxmCastTarget::Float32,
    std::optional<std::size_t> output = std::nullopt)
{
    return {opcode, lhs, rhs, 1.0f, 0, cast, output,
        ftlpu::Hemisphere::East, ftlpu::Hemisphere::East};
}

RtlSchedule swiglu_schedule(ftlpu::TspSliceSystem& system)
{
    auto schedule = RtlSchedule(system);
    for (std::size_t row = 0; row < kRows; ++row) {
        const auto cycle = 20 + row;
        for (std::size_t byte = 0; byte < 4; ++byte) {
            schedule.mem_at(byte, cycle - 2,
                ftlpu::MemInstruction::Read(
                    kGateAddress + row, ftlpu::StreamId::West(byte)));
            schedule.mem_at(4 + byte, cycle - 3,
                ftlpu::MemInstruction::Read(
                    kUpAddress + row, ftlpu::StreamId::West(4 + byte)));
        }
        schedule.vxm_at(0, cycle, vxm_instruction(ftlpu::VxmAluOpcode::Negate,
            ftlpu::VxmLaneOperand::StreamFloat32(32), ftlpu::VxmLaneOperand::Imm(0.0f)));
        schedule.vxm_at(1, cycle, vxm_instruction(ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::StreamFloat32(32), ftlpu::VxmLaneOperand::StreamFloat32(36)));
        schedule.vxm_at(2, cycle + 1, vxm_instruction(ftlpu::VxmAluOpcode::Exp,
            ftlpu::VxmLaneOperand::Alu(0), ftlpu::VxmLaneOperand::Imm(0.0f)));
        schedule.vxm_at(5, cycle + 1, vxm_instruction(ftlpu::VxmAluOpcode::Pass,
            ftlpu::VxmLaneOperand::Alu(1), ftlpu::VxmLaneOperand::Imm(0.0f)));
        schedule.vxm_at(3, cycle + 2, vxm_instruction(ftlpu::VxmAluOpcode::Add,
            ftlpu::VxmLaneOperand::Alu(2), ftlpu::VxmLaneOperand::Imm(1.0f)));
        schedule.vxm_at(6, cycle + 2, vxm_instruction(ftlpu::VxmAluOpcode::Pass,
            ftlpu::VxmLaneOperand::Alu(5), ftlpu::VxmLaneOperand::Imm(0.0f)));
        schedule.vxm_at(4, cycle + 3, vxm_instruction(ftlpu::VxmAluOpcode::Divide,
            ftlpu::VxmLaneOperand::Imm(1.0f), ftlpu::VxmLaneOperand::Alu(3)));
        schedule.vxm_at(7, cycle + 3, vxm_instruction(ftlpu::VxmAluOpcode::Pass,
            ftlpu::VxmLaneOperand::Alu(6), ftlpu::VxmLaneOperand::Imm(0.0f)));
        schedule.vxm_at(8, cycle + 4, vxm_instruction(ftlpu::VxmAluOpcode::Multiply,
            ftlpu::VxmLaneOperand::Alu(7), ftlpu::VxmLaneOperand::Alu(4)));
        schedule.vxm_at(9, cycle + 5, vxm_instruction(ftlpu::VxmAluOpcode::Cast,
            ftlpu::VxmLaneOperand::Alu(8), ftlpu::VxmLaneOperand::Imm(0.0f),
            ftlpu::VxmCastTarget::BFloat16, 0));
        for (std::size_t byte = 0; byte < 2; ++byte) {
            const auto slice = 32 + row * 2 + byte;
            schedule.mem_at(slice, cycle + 6 + slice / 4,
                ftlpu::MemInstruction::Write(
                    0, ftlpu::StreamId::East(byte)));
        }
    }
    return schedule;
}

RtlSchedule down_schedule(ftlpu::TspSliceSystem& system)
{
    auto schedule = RtlSchedule(system);
    constexpr auto scale = 0.015625f;
    for (std::size_t block = 0; block < 4; ++block) {
        const auto cycle = 20 + block;
        for (std::size_t stream = 0; stream < 8; ++stream)
            schedule.mem_at(stream, cycle - east_latency(stream),
                ftlpu::MemInstruction::Read(
                    4 + block, ftlpu::StreamId::East(stream)));
        schedule.mxm_dequant_at(0, cycle,
            ftlpu::MxmDequantInstruction::Scale(scale));
        schedule.mxm_load_at(0, cycle,
            ftlpu::MxmControlInstruction::IW(0, block));
    }
    constexpr std::size_t compute_cycle = 28;
    for (std::size_t slice = 32; slice < 48; ++slice)
        schedule.mem_at(slice, compute_cycle - east_latency(slice),
            ftlpu::MemInstruction::Read(0, ftlpu::StreamId::East(slice - 32)));
    schedule.mxm_compute_at(0, compute_cycle,
        ftlpu::MxmControlInstruction::Compute(
            0, 0, 0, 0, 1,
            ftlpu::MxmAccumulatorDestination::Sram,
            ftlpu::MxmDataFormat::BFloat16,
            ftlpu::MxmComputeMode::Block8));
    constexpr std::size_t read_cycle = 60;
    schedule.mxm_compute_at(0, read_cycle,
        ftlpu::MxmControlInstruction::AccumulatorRead(
            0, 0, false, ftlpu::MxmComputeMode::Block8));
    for (std::size_t slice = 0; slice < 32; ++slice)
        schedule.mem_at(slice, read_cycle + west_output_latency(slice),
            ftlpu::MemInstruction::Write(
                kFinalAddress, ftlpu::StreamId::West(slice)));
    return schedule;
}

void run(ftlpu::TspSliceSystem& system, std::size_t cycles)
{
    for (std::size_t cycle = 0; cycle < cycles; ++cycle) system.tick({});
}

void write_golden(const char* path, const ftlpu::TspSliceSystem& system)
{
    auto output = std::ofstream(path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open FFN golden output");
    output << std::hex << std::setfill('0');
    for (std::size_t row = 0; row < 8; ++row)
        for (std::size_t slice = 0; slice < 8; ++slice)
            for (std::size_t tile = 0; tile < 4; ++tile)
                output << std::setw(16)
                       << read_word(system, slice, 20 + row, tile) << '\n';
    for (std::size_t slice = 32; slice < 48; ++slice)
        for (std::size_t tile = 0; tile < 4; ++tile)
            output << std::setw(16) << read_word(system, slice, 0, tile) << '\n';
    for (std::size_t slice = 0; slice < 32; ++slice)
        for (std::size_t tile = 0; tile < 4; ++tile)
            output << std::setw(16)
                   << read_word(system, slice, kFinalAddress, tile) << '\n';
}

} // namespace

int main(int argc, char** argv)
try {
    if (argc != 6) {
        std::cerr << "usage: smollm2_ffn <init> <golden> <gate> <swiglu> <down>\n";
        return 2;
    }
    auto system = ftlpu::TspSliceSystem {};
    initialize(system);
    write_init(argv[1], system);
    auto gate = gate_up_schedule(system);
    gate.write(argv[3]);
    run(system, 180);
    system.reset_execution_state();
    auto swiglu = swiglu_schedule(system);
    swiglu.write(argv[4]);
    run(system, 64);
    system.reset_execution_state();
    auto down = down_schedule(system);
    down.write(argv[5]);
    run(system, 96);
    write_golden(argv[2], system);
    std::cout << "C model reduced SmolLM2 FFN generated: gate=" << gate.size()
              << " swiglu=" << swiglu.size() << " down=" << down.size() << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
