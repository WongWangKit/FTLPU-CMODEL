#pragma once

#include "ftlpu/hardware_params.hpp"

#include <cstddef>
#include <stdexcept>

namespace ftlpu {

struct TileCoord {
    std::size_t row{0};
    std::size_t slice{0};
};

struct SliceGroupCoord {
    std::size_t row{0};
    std::size_t group{0};
    std::size_t local_slice{0};
};

struct SramCoord {
    std::size_t slice{0};
};

inline SliceGroupCoord slice_group_for(TileCoord coord)
{
    if (coord.row >= hw::kTileRows) {
        throw std::out_of_range("tile row is outside the TSP grid");
    }
    if (coord.slice >= hw::kSliceColumns) {
        throw std::out_of_range("slice column is outside the TSP grid");
    }

    return SliceGroupCoord {
        coord.row,
        coord.slice / hw::kSlicesPerGroup,
        coord.slice % hw::kSlicesPerGroup,
    };
}

inline std::size_t stream_register_after_slice(std::size_t slice)
{
    if (slice >= hw::kSliceColumns) {
        throw std::out_of_range("slice column is outside the TSP grid");
    }

    return slice / hw::kSlicesPerGroup + 1;
}

inline std::size_t stream_register_before_slice(std::size_t slice)
{
    if (slice >= hw::kSliceColumns) {
        throw std::out_of_range("slice column is outside the TSP grid");
    }

    return slice / hw::kSlicesPerGroup;
}

inline std::size_t sram_block_index(SramCoord coord)
{
    if (coord.slice >= hw::kSliceColumns) {
        throw std::out_of_range("SRAM slice is outside the TSP grid");
    }

    return coord.slice;
}

inline std::size_t sram_byte_address(std::size_t block, std::size_t word_index, std::size_t byte_offset)
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

} // namespace ftlpu
