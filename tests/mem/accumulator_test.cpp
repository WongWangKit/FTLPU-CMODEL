#include "ftlpu/mem/mem_array.hpp"

#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace {

void set_memory_value(
    ftlpu::MemArrayModel& mem,
    std::size_t group_base,
    std::size_t tile,
    std::size_t lane,
    std::size_t address,
    float value)
{
    const auto raw = std::bit_cast<std::uint32_t>(value);
    for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
        mem.set_sram_lane_byte(
            group_base + byte,
            tile,
            address,
            lane,
            static_cast<std::uint8_t>((raw >> (byte * 8)) & 0xffu));
    }
}

float memory_value(
    const ftlpu::MemArrayModel& mem,
    std::size_t group_base,
    std::size_t tile,
    std::size_t lane,
    std::size_t address)
{
    std::uint32_t raw = 0;
    for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
        raw |= static_cast<std::uint32_t>(mem.sram_lane_byte(
            group_base + byte,
            tile,
            address,
            lane)) << (byte * 8);
    }
    return std::bit_cast<float>(raw);
}

} // namespace

int main()
{
    constexpr std::array<std::size_t, 2> kGroupBases {
        ftlpu::hw::kWestAccumulatorMemSliceBase,
        ftlpu::hw::kEastAccumulatorMemSliceBase,
    };
    constexpr std::array<std::size_t, 2> kAddresses {37, 53};
    constexpr std::array<std::size_t, 2> kStreamBases {8, 12};
    ftlpu::MemArrayModel mem;
    ftlpu::StreamRegisterFabric fabric(ftlpu::hw::kSystemStreamRegisterColumns);

    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            for (std::size_t group = 0; group < kGroupBases.size(); ++group) {
                set_memory_value(mem, kGroupBases[group], tile, lane, kAddresses[group], 1.25f + group);
            }
        }
    }
    for (std::size_t group = 0; group < kGroupBases.size(); ++group) {
        mem.enqueue_instruction(
            kGroupBases[group],
            ftlpu::MemInstruction::Accumulate(
                kAddresses[group],
                ftlpu::StreamId::West(kStreamBases[group]),
                group == 0
                    ? ftlpu::MemAccumulatorDestination::Sram
                    : ftlpu::MemAccumulatorDestination::Stream));
    }

    for (std::size_t cycle = 0; cycle < ftlpu::hw::kTileRows; ++cycle) {
        const auto tile = cycle;
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            for (std::size_t group = 0; group < kGroupBases.size(); ++group) {
                const auto input = static_cast<float>(tile * 10 + lane) + 0.5f + group;
                const auto raw = std::bit_cast<std::uint32_t>(input);
                const auto input_column = kGroupBases[group] / ftlpu::hw::kMemSlicesPerGroup + 1;
                for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
                    fabric.initialize_cell(
                        input_column,
                        tile,
                        lane,
                        ftlpu::StreamId::West(kStreamBases[group] + byte),
                        ftlpu::StreamCell::Valid(
                            static_cast<std::uint8_t>((raw >> (byte * 8)) & 0xffu),
                            lane + 1 == ftlpu::hw::kLanesPerTile));
                }
            }
        }

        fabric.begin_cycle();
        mem.evaluate(fabric);
        fabric.stage_linear_links();
        fabric.commit_cycle();

        const auto stream_group = std::size_t {1};
        const auto output_column = kGroupBases[stream_group] / ftlpu::hw::kMemSlicesPerGroup;
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            const auto expected = 1.25f + stream_group
                + static_cast<float>(tile * 10 + lane) + 0.5f + stream_group;
            const auto raw = std::bit_cast<std::uint32_t>(expected);
            for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
                const auto& cell = fabric.cell(
                    output_column,
                    tile,
                    lane,
                    ftlpu::StreamId::West(kStreamBases[stream_group] + byte));
                assert(cell.valid);
                assert(cell.data == static_cast<std::uint8_t>((raw >> (byte * 8)) & 0xffu));
            }
        }
    }

    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            for (std::size_t group = 0; group < kGroupBases.size(); ++group) {
                const auto expected = group == 0
                    ? 1.25f + group + static_cast<float>(tile * 10 + lane) + 0.5f + group
                    : 0.0f;
                assert(memory_value(mem, kGroupBases[group], tile, lane, kAddresses[group]) == expected);
                if (group != 0) continue;
                const auto output_column = kGroupBases[group] / ftlpu::hw::kMemSlicesPerGroup;
                for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
                    assert(!fabric.cell(
                        output_column,
                        tile,
                        lane,
                        ftlpu::StreamId::West(kStreamBases[group] + byte)).valid);
                }
            }
        }
    }

    // The same four slices retain ordinary MEM storage behavior.
    mem.set_sram_lane_byte(ftlpu::hw::kWestAccumulatorMemSliceBase + 2, 1, 91, 3, 0xa5);
    assert(mem.sram_lane_byte(ftlpu::hw::kWestAccumulatorMemSliceBase + 2, 1, 91, 3) == 0xa5);
    return 0;
}
