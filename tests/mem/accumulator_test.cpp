#include "ftlpu/mem/mem_array.hpp"

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace {

void initialize_accumulator(
    ftlpu::MemArrayModel& mem,
    ftlpu::MemLocalWordAddress13 address,
    float value)
{
    const auto raw = std::bit_cast<std::uint32_t>(value);
    for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
        for (std::size_t lane = 0;
             lane < ftlpu::hw::kLanesPerTile;
             ++lane) {
            mem.set_sram_lane_byte(
                ftlpu::hw::kEastAccumulatorMemSliceBase + byte,
                0,
                address,
                lane,
                static_cast<std::uint8_t>(raw >> (byte * 8)));
        }
    }
}

void stage_fp32(
    ftlpu::StreamRegisterFabric& fabric,
    std::size_t column,
    std::size_t stream_base,
    float value,
    std::uint64_t tag)
{
    const auto raw = std::bit_cast<std::uint32_t>(value);
    fabric.begin_cycle();
    for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
        ftlpu::StreamPayloadSegment16 segment{};
        segment.fill(
            static_cast<std::uint8_t>(raw >> (byte * 8)));
        fabric.stage_payload_segment(
            column,
            0,
            ftlpu::StreamId::West(stream_base + byte),
            segment,
            tag,
            "accumulator test");
    }
    fabric.commit_cycle();
}

float load_accumulator_lane(
    const ftlpu::MemArrayModel& mem,
    ftlpu::MemLocalWordAddress13 address,
    std::size_t lane)
{
    std::uint32_t raw = 0;
    for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
        raw |= static_cast<std::uint32_t>(
                   mem.sram_lane_byte(
                       ftlpu::hw::kEastAccumulatorMemSliceBase + byte,
                       0,
                       address,
                       lane))
            << (byte * 8);
    }
    return std::bit_cast<float>(raw);
}

} // namespace

int main()
{
    constexpr auto kAddress =
        ftlpu::MemLocalWordAddress13::FromFields(1, 29);
    constexpr std::size_t kStreamBase = 8;

    ftlpu::StreamRegisterFabric fabric(
        ftlpu::hw::kMemBoundaryStreamRegisterColumns);
    ftlpu::MemArrayModel mem;
    const auto input_column = mem.ports().input_column(
        ftlpu::hw::kEastAccumulatorMemSliceBase,
        ftlpu::StreamDirection::West);

    initialize_accumulator(mem, kAddress, 1.25f);
    stage_fp32(fabric, input_column, kStreamBase, 2.5f, 71);
    mem.enqueue_instruction(
        ftlpu::hw::kEastAccumulatorMemSliceBase,
        ftlpu::MemInstruction::Accumulate(
            kAddress,
            ftlpu::StreamId::West(kStreamBase),
            ftlpu::MemAccumulatorDestination::Sram));
    fabric.begin_cycle();
    mem.evaluate(fabric);
    fabric.commit_cycle();
    for (std::size_t lane = 0;
         lane < ftlpu::hw::kLanesPerTile;
         ++lane) {
        assert(load_accumulator_lane(mem, kAddress, lane) == 3.75f);
    }

    mem.reset_execution_state();
    fabric.reset();
    stage_fp32(fabric, input_column, kStreamBase, 0.25f, 72);
    mem.enqueue_instruction(
        ftlpu::hw::kEastAccumulatorMemSliceBase,
        ftlpu::MemInstruction::Accumulate(
            kAddress,
            ftlpu::StreamId::West(kStreamBase),
            ftlpu::MemAccumulatorDestination::Stream));
    fabric.begin_cycle();
    mem.evaluate(fabric);
    fabric.commit_cycle();

    const auto output_column = mem.ports().output_column(
        ftlpu::hw::kEastAccumulatorMemSliceBase,
        ftlpu::StreamDirection::West);
    for (std::size_t lane = 0;
         lane < ftlpu::hw::kLanesPerTile;
         ++lane) {
        std::uint32_t raw = 0;
        for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
            const auto& cell = fabric.cell(
                output_column,
                0,
                lane,
                ftlpu::StreamId::West(kStreamBase + byte));
            assert(cell.valid);
            assert(cell.vector_tag == 72);
            raw |= static_cast<std::uint32_t>(cell.data) << (byte * 8);
        }
        assert(std::bit_cast<float>(raw) == 4.0f);
        assert(load_accumulator_lane(mem, kAddress, lane) == 0.0f);
    }
    return 0;
}
