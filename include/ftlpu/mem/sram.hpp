#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/stream.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace ftlpu {

// One single-port SRAM bank owned by one MEM functional slice. It contains
// 2048 architecturally visible rows by default, and
// every row spans all configured tiles:
//
//   2048 rows * (4 tiles * 8 bytes) = 64 KiB.
//
// MEM instructions address rows.  As an instruction travels through a tile,
// that tile reads or writes its own contiguous 8-byte portion of the row.
class Sram {
public:
    static constexpr std::size_t kRows = hw::kSramMaxDepthRows;
    static constexpr std::size_t kBytesPerRow = hw::kSramRowBytes;
    static constexpr std::size_t kBytesPerTileSegment = hw::kLanesPerTile;
    static constexpr std::size_t kCapacityBytes = kRows * kBytesPerRow;

    Sram();
    ~Sram();
    Sram(Sram&&) noexcept;
    Sram& operator=(Sram&&) noexcept;
    Sram(const Sram&) = delete;
    Sram& operator=(const Sram&) = delete;

    void set_active_rows(std::size_t rows);
    std::size_t active_rows() const noexcept { return active_rows_; }
    void clear();
    std::uint8_t byte(std::size_t row, std::size_t byte_offset) const;
    void set_byte(std::size_t row, std::size_t byte_offset, std::uint8_t value);
    StreamPayloadTileSegment read_segment(std::size_t tile, std::size_t row) const;
    void write_segment(
        std::size_t tile,
        std::size_t row,
        const StreamPayloadTileSegment& values);

private:
    static constexpr std::size_t kPageBytes = 4096;
    static constexpr std::size_t kPageCount =
        (kCapacityBytes + kPageBytes - 1) / kPageBytes;
    using Page = std::array<std::uint8_t, kPageBytes>;

    void check_row(std::size_t row) const;
    static std::size_t tile_byte_offset(std::size_t tile);
    std::size_t flat_index(std::size_t row, std::size_t byte_offset) const;
    Page& ensure_page(std::size_t page);

    std::array<std::unique_ptr<Page>, kPageCount> pages_{};
    std::size_t active_rows_{hw::kSramDepthRows};
};

class SramArray {
public:
    SramArray();

    void set_active_rows(std::size_t rows);
    void clear();

    Sram& bank(std::size_t mem_slice, std::size_t bank);

    const Sram& bank(std::size_t mem_slice, std::size_t bank) const;

private:
    std::vector<std::array<Sram, hw::kMemBanksPerSlice>> slices_{};
};

} // namespace ftlpu
