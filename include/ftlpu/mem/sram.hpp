#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/stream.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ftlpu {

// One SRAM storage block owned by one MEM functional slice. It contains 8192
// rows (two 4096-row banks), and every row spans all configured tiles:
//
//   8192 rows * (4 tiles * 16 bytes) = 512 KiB.
//
// MEM instructions address rows.  As an instruction travels through a tile,
// that tile reads or writes its own contiguous 16-byte portion of the row.
class SramBank {
public:
    static constexpr std::size_t kRows = hw::kSramDepthRows;
    static constexpr std::size_t kBytesPerRow = hw::kSramRowBytes;
    static constexpr std::size_t kBytesPerTileSegment = hw::kLanesPerTile;
    static constexpr std::size_t kCapacityBytes = kRows * kBytesPerRow;

    SramBank()
        : bytes_(kCapacityBytes, 0)
    {
    }

    void clear()
    {
        std::fill(bytes_.begin(), bytes_.end(), 0);
    }

    std::uint8_t byte(std::size_t row, std::size_t byte_offset) const
    {
        return bytes_.at(flat_index(row, byte_offset));
    }

    void set_byte(std::size_t row, std::size_t byte_offset, std::uint8_t value)
    {
        bytes_.at(flat_index(row, byte_offset)) = value;
    }

    StreamPayloadSegment16 read_segment(std::size_t tile, std::size_t row) const
    {
        const auto byte_offset = tile_byte_offset(tile);
        check_row(row);
        StreamPayloadSegment16 result{};
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            result[lane] = byte(row, byte_offset + lane);
        }
        return result;
    }

    void write_segment(
        std::size_t tile,
        std::size_t row,
        const StreamPayloadSegment16& values)
    {
        const auto byte_offset = tile_byte_offset(tile);
        check_row(row);
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            set_byte(row, byte_offset + lane, values[lane]);
        }
    }

private:
    static void check_row(std::size_t row)
    {
        if (row >= kRows) {
            throw std::out_of_range("SRAM row is outside the 8192-row bank");
        }
    }

    static std::size_t tile_byte_offset(std::size_t tile)
    {
        if (tile >= hw::kTileRows) {
            throw std::out_of_range("SRAM tile is outside the configured MEM slice");
        }
        return tile * kBytesPerTileSegment;
    }

    static std::size_t flat_index(std::size_t row, std::size_t byte_offset)
    {
        check_row(row);
        if (byte_offset >= kBytesPerRow) {
            throw std::out_of_range("SRAM byte offset is outside the physical vector row");
        }
        return row * kBytesPerRow + byte_offset;
    }

    std::vector<std::uint8_t> bytes_{};
};

class SramArray {
public:
    SramArray()
        : banks_(hw::kMemSliceColumns)
    {
    }

    void clear()
    {
        for (auto& bank : banks_) {
            bank.clear();
        }
    }

    SramBank& bank(std::size_t mem_slice)
    {
        return banks_.at(mem_slice);
    }

    const SramBank& bank(std::size_t mem_slice) const
    {
        return banks_.at(mem_slice);
    }

private:
    std::vector<SramBank> banks_{};
};

} // namespace ftlpu
