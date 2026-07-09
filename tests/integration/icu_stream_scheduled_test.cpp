#include "ftlpu/system/tsp_slice_system.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <streambuf>
#include <string>

namespace {

constexpr std::size_t kInputAddress = 1024;
constexpr std::size_t kOutputAddress = 2048;
constexpr std::size_t kInputStreamBase = 32;
constexpr std::size_t kOutputStream = 0;
constexpr std::size_t kVxmIssueCycle = 2;
constexpr std::size_t kMemWriteIssueCycle = 3;
constexpr std::size_t kMemWriteQueueDelayAfterRead = 2;
constexpr std::size_t kTotalCycles = kMemWriteIssueCycle + ftlpu::hw::kTileRows + 2;

class NullBuffer : public std::streambuf {
public:
    int overflow(int c) override
    {
        return c;
    }
};

bool require(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

std::int32_t input_value(std::size_t tile, std::size_t lane)
{
    return static_cast<std::int32_t>((tile * 17 + lane * 13) % 313) - 156;
}

void initialize_mem(ftlpu::TspSliceSystem& system)
{
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            const auto bytes = ftlpu::VxmLane::pack_int32(input_value(tile, lane));
            for (std::size_t byte = 0; byte < bytes.size(); ++byte) {
                system.mem().set_sram_byte(byte, tile, kInputAddress + lane, bytes[byte]);
            }
        }
    }
}

void issue_mem_reads(ftlpu::TspSliceSystem& system)
{
    for (std::size_t byte = 0; byte < 4; ++byte) {
        system.icu().enqueue_mem(byte, ftlpu::MemInstruction::Read(kInputAddress, kInputStreamBase + byte));
    }
}

void issue_vxm_cast(ftlpu::TspSliceSystem& system)
{
    system.icu().enqueue_vxm(0, ftlpu::VxmLaneAluInstruction {
        ftlpu::VxmAluOpcode::Cast,
        ftlpu::VxmLaneOperand::StreamInt32(kInputStreamBase),
        ftlpu::VxmLaneOperand::Imm(0.0f),
        1.0f,
        0,
        ftlpu::VxmCastTarget::Int8,
        kOutputStream,
    });
}

void issue_mem_writes(ftlpu::TspSliceSystem& system)
{
    system.icu().enqueue_mem(0, ftlpu::MemInstruction::Write(kOutputAddress, kOutputStream));
}

std::size_t cycle_for(const std::string& log, const std::string& pattern)
{
    const auto pos = log.find(pattern);
    if (pos == std::string::npos) {
        throw std::logic_error("missing ICU log pattern: " + pattern);
    }
    const auto cycle_pos = log.rfind("icu cycle ", pos);
    if (cycle_pos == std::string::npos) {
        throw std::logic_error("missing ICU cycle header before pattern: " + pattern);
    }
    const auto value_pos = cycle_pos + std::string("icu cycle ").size();
    const auto end_pos = log.find('\n', value_pos);
    return static_cast<std::size_t>(std::stoul(log.substr(value_pos, end_pos - value_pos)));
}

bool verify_icu_queue_nop_and_repeat()
{
    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    std::ostringstream log;

    system->icu().enqueue_mem_nop(0, 2);
    system->icu().enqueue_mem(0, ftlpu::MemInstruction::Read(100, 0));
    system->icu().enqueue_mem_repeat(0, 2, 1, 16);

    system->icu().enqueue_mem(1, ftlpu::MemInstruction::Read(200, 1));

    system->icu().enqueue_mem(2, ftlpu::MemInstruction::Read(300, 2));
    system->icu().enqueue_mem_repeat(2, 2, 2, 8);

    for (std::size_t cycle = 0; cycle < 5; ++cycle) {
        system->dispatch_icu_only(&log);
    }

    const auto text = log.str();
    if (!require(cycle_for(text, "ICU -> MEM q1 Read address=200 stream=1") == 0, "per-queue NOP blocked another MEM queue")) {
        return false;
    }
    if (!require(cycle_for(text, "ICU -> MEM q0 Read address=100 stream=0") == 2, "MEM queue NOP did not delay by two cycles")) {
        return false;
    }
    if (!require(cycle_for(text, "ICU -> MEM q0 Read address=116 stream=0") == 3, "MEM repeat stride did not issue next cycle")) {
        return false;
    }
    if (!require(cycle_for(text, "ICU -> MEM q0 Read address=132 stream=0") == 4, "MEM repeat stride did not produce second repeat")) {
        return false;
    }
    if (!require(cycle_for(text, "ICU -> MEM q2 Read address=300 stream=2") == 0, "MEM repeat base issued on wrong cycle")) {
        return false;
    }
    if (!require(cycle_for(text, "ICU -> MEM q2 Read address=308 stream=2") == 2, "MEM repeat interval d=2 did not delay first repeat")) {
        return false;
    }
    if (!require(cycle_for(text, "ICU -> MEM q2 Read address=316 stream=2") == 4, "MEM repeat interval d=2 did not delay second repeat")) {
        return false;
    }
    return true;
}

} // namespace

int main()
try
{
    if (!verify_icu_queue_nop_and_repeat()) {
        return 1;
    }

    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    NullBuffer null_buffer;
    std::ostream log(&null_buffer);

    initialize_mem(*system);
    issue_mem_reads(*system);
    system->icu().enqueue_vxm_nop(0, kVxmIssueCycle);
    issue_vxm_cast(*system);
    system->icu().enqueue_mem_nop(0, kMemWriteQueueDelayAfterRead);
    issue_mem_writes(*system);

    for (std::size_t cycle = 0; cycle < kTotalCycles; ++cycle) {
        system->tick(log);
    }

    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            const auto expected = ftlpu::VxmAlu::cast_scalar_to_int8(static_cast<float>(input_value(tile, lane)));
            const auto actual = static_cast<std::int8_t>(
                system->mem().sram_byte(0, tile, kOutputAddress + lane));
            if (actual != expected) {
                std::cerr << "ICU scheduled stream result mismatch"
                          << " tile=" << tile
                          << " lane=" << lane
                          << " actual=" << static_cast<int>(actual)
                          << " expected=" << static_cast<int>(expected)
                          << '\n';
                return 1;
            }
        }
    }

    return 0;
}
catch (const std::exception& ex) {
    std::cerr << "icu_stream_scheduled_test failed: " << ex.what() << '\n';
    return 1;
}
