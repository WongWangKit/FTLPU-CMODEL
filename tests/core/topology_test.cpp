#include "ftlpu/core/topology.hpp"

#include <cassert>
#include <cstddef>
#include <stdexcept>

int main()
{
    static_assert(ftlpu::hw::kTileRows == 20);
    static_assert(ftlpu::hw::kSliceColumns == 44);
    static_assert(ftlpu::hw::kSlicesPerGroup == 4);
    static_assert(ftlpu::hw::kSliceGroups == 11);
    static_assert(ftlpu::hw::kStreamRegisterColumns == 12);
    static_assert(ftlpu::hw::kLanesPerTile == 16);
    static_assert(ftlpu::hw::kStreams == 64);
    static_assert(ftlpu::hw::kEastStreams == 32);
    static_assert(ftlpu::hw::kWestStreams == 32);
    static_assert(ftlpu::hw::kMemReadBytesPerCycle == 16);
    static_assert(ftlpu::hw::kMemWriteBytesPerCycle == 16);
    static_assert(ftlpu::hw::kPublicSramBlocks == 88);
    static_assert(ftlpu::hw::kSramBlocks == 44);
    static_assert(ftlpu::hw::kPhysicalVectorBytes == 320);
    static_assert(ftlpu::hw::kSramDepthWords == 8192);
    static_assert(ftlpu::hw::kSramBlockBytes == 2621440);
    static_assert(ftlpu::hw::kTotalSramBytes == 115343360);
    static_assert(ftlpu::hw::kPublicTotalSramBytes == 230686720);

    auto group = ftlpu::slice_group_for({3, 10});
    assert(group.row == 3);
    assert(group.group == 2);
    assert(group.local_slice == 2);

    assert(ftlpu::stream_register_before_slice(0) == 0);
    assert(ftlpu::stream_register_after_slice(0) == 1);
    assert(ftlpu::stream_register_before_slice(3) == 0);
    assert(ftlpu::stream_register_after_slice(3) == 1);
    assert(ftlpu::stream_register_before_slice(4) == 1);
    assert(ftlpu::stream_register_after_slice(4) == 2);
    assert(ftlpu::stream_register_before_slice(43) == 10);
    assert(ftlpu::stream_register_after_slice(43) == 11);

    assert(ftlpu::sram_block_index({0}) == 0);
    assert(ftlpu::sram_block_index({1}) == 1);
    assert(ftlpu::sram_block_index({43}) == 43);

    assert(ftlpu::sram_byte_address(0, 0, 0) == 0);
    assert(ftlpu::sram_byte_address(0, 1, 0) == 320);
    assert(ftlpu::sram_byte_address(1, 0, 0) == ftlpu::hw::kSramBlockBytes);
    assert(ftlpu::sram_byte_address(43, 8191, 319) == ftlpu::hw::kTotalSramBytes - 1);

    bool caught = false;
    try {
        static_cast<void>(ftlpu::slice_group_for({20, 0}));
    } catch (const std::out_of_range&) {
        caught = true;
    }
    assert(caught);

    caught = false;
    try {
        static_cast<void>(ftlpu::sram_byte_address(44, 0, 0));
    } catch (const std::out_of_range&) {
        caught = true;
    }
    assert(caught);

    return 0;
}
