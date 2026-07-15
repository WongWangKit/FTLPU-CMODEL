#pragma once

#include "ftlpu/core/hardware_params.hpp"

#include <cstddef>
#include <stdexcept>

namespace ftlpu {

// Coordinates inside the MEM region only.  They must not be confused with a
// whole-chip functional-slice or physical SR-column coordinate.
struct MemTileCoord {
    std::size_t row{0};
    std::size_t mem_slice{0};
};

struct MemGroupCoord {
    std::size_t row{0};
    std::size_t group{0};
    union {
        std::size_t local_mem_slice;
        std::size_t local_slice;
    };

    constexpr MemGroupCoord()
        : local_mem_slice(0)
    {
    }

    constexpr MemGroupCoord(
        std::size_t row_value,
        std::size_t group_value,
        std::size_t local_value)
        : row(row_value)
        , group(group_value)
        , local_mem_slice(local_value)
    {
    }
};

struct SramCoord {
    std::size_t mem_slice{0};
};

inline MemGroupCoord mem_group_for(MemTileCoord coord)
{
    if (coord.row >= hw::kTileRows) {
        throw std::out_of_range("tile row is outside the TSP grid");
    }
    if (coord.mem_slice >= hw::kMemSliceColumns) {
        throw std::out_of_range("MEM slice is outside the modeled hemisphere");
    }

    return MemGroupCoord {
        coord.row,
        coord.mem_slice / hw::kMemSlicesPerGroup,
        coord.mem_slice % hw::kMemSlicesPerGroup,
    };
}

inline std::size_t mem_boundary_before_slice(std::size_t mem_slice)
{
    if (mem_slice >= hw::kMemSliceColumns) {
        throw std::out_of_range("MEM slice is outside the modeled hemisphere");
    }
    return mem_slice / hw::kMemSlicesPerGroup;
}

inline std::size_t mem_boundary_after_slice(std::size_t mem_slice)
{
    return mem_boundary_before_slice(mem_slice) + 1;
}

inline std::size_t sram_block_index(SramCoord coord)
{
    if (coord.mem_slice >= hw::kMemSliceColumns) {
        throw std::out_of_range("SRAM slice is outside the modeled hemisphere");
    }
    return coord.mem_slice;
}

inline std::size_t sram_byte_address(
    std::size_t block,
    std::size_t word_index,
    std::size_t byte_offset)
{
    if (block >= hw::kSramBlocks) {
        throw std::out_of_range("SRAM block index is outside the TSP memory");
    }
    if (word_index >= hw::kSramDepthWords) {
        throw std::out_of_range("SRAM word index is outside the block");
    }
    if (byte_offset >= hw::kPhysicalVectorBytes) {
        throw std::out_of_range("SRAM byte offset is outside the vector word");
    }

    return block * hw::kSramBlockBytes
        + word_index * hw::kPhysicalVectorBytes
        + byte_offset;
}

// Compatibility aliases for existing examples.  New code should use the
// MEM-specific names so that 44 MEM slices/12 MEM boundaries are not mistaken
// for the complete functional-slice/SR topology.
using TileCoord = MemTileCoord;
using SliceGroupCoord = MemGroupCoord;

inline SliceGroupCoord slice_group_for(TileCoord coord)
{
    const auto group = mem_group_for(coord);
    return SliceGroupCoord {group.row, group.group, group.local_mem_slice};
}

inline std::size_t stream_register_before_slice(std::size_t slice)
{
    return mem_boundary_before_slice(slice);
}

inline std::size_t stream_register_after_slice(std::size_t slice)
{
    return mem_boundary_after_slice(slice);
}

} // namespace ftlpu
