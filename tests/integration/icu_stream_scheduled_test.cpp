#include "ftlpu/system/tsp_slice_system.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <ostream>
#include <streambuf>

namespace {

constexpr std::size_t kInputAddress = 1024;
constexpr std::size_t kOutputAddress = 2048;
constexpr std::size_t kInputStreamBase = 32;
constexpr std::size_t kOutputStream = 0;
constexpr std::size_t kMemReadIssueCycle = 0;
constexpr std::size_t kVxmIssueCycle = 1;
constexpr std::size_t kMemWriteIssueCycle = 3;
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

} // namespace

int main()
try
{
    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    NullBuffer null_buffer;
    std::ostream log(&null_buffer);

    initialize_mem(*system);

    for (std::size_t cycle = 0; cycle < kTotalCycles; ++cycle) {
        if (cycle == kMemReadIssueCycle) {
            issue_mem_reads(*system);
        }
        if (cycle == kVxmIssueCycle) {
            issue_vxm_cast(*system);
        }
        if (cycle == kMemWriteIssueCycle) {
            issue_mem_writes(*system);
        }
        system->tick(log);
    }

    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            const auto expected = ftlpu::VxmAlu::cast_scalar_to_int8(static_cast<float>(input_value(tile, lane)));
            const auto actual = static_cast<std::int8_t>(
                system->mem().sram_byte(0, tile, kOutputAddress + lane));
            if (!require(actual == expected, "ICU scheduled stream result mismatch")) {
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
