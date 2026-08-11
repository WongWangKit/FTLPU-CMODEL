#include "ftlpu/mem/sram.hpp"

#include <algorithm>

namespace ftlpu {

Sram::Sram() = default;
Sram::~Sram() = default;
Sram::Sram(Sram&&) noexcept = default;
Sram& Sram::operator=(Sram&&) noexcept = default;

void Sram::set_active_rows(std::size_t rows)
{
    if (rows == 0 || rows > kRows)
        throw std::invalid_argument(
            "SRAM active depth exceeds its physical capacity");
    active_rows_ = rows;
}

void Sram::clear()
{
    for (auto& page : pages_) {
        page.reset();
    }
}

std::uint8_t Sram::byte(std::size_t row, std::size_t byte_offset) const
{
    const auto index = flat_index(row, byte_offset);
    const auto& page = pages_[index / kPageBytes];
    return page == nullptr ? 0 : (*page)[index % kPageBytes];
}

void Sram::set_byte(std::size_t row, std::size_t byte_offset, std::uint8_t value)
{
    const auto index = flat_index(row, byte_offset);
    const auto page_index = index / kPageBytes;
    if (value == 0 && pages_[page_index] == nullptr) {
        return;
    }
    ensure_page(page_index)[index % kPageBytes] = value;
}

StreamPayloadTileSegment Sram::read_segment(std::size_t tile, std::size_t row) const
{
    const auto byte_offset = tile_byte_offset(tile);
    check_row(row);
    StreamPayloadTileSegment result{};
    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
        result[lane] = byte(row, byte_offset + lane);
    }
    return result;
}

void Sram::write_segment(
    std::size_t tile,
    std::size_t row,
    const StreamPayloadTileSegment& values)
{
    const auto byte_offset = tile_byte_offset(tile);
    check_row(row);
    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
        set_byte(row, byte_offset + lane, values[lane]);
    }
}

void Sram::check_row(std::size_t row) const
{
    if (row >= active_rows_) {
        throw std::out_of_range("SRAM row is outside the configured slice SRAM");
    }
}

std::size_t Sram::tile_byte_offset(std::size_t tile)
{
    if (tile >= hw::kTileRows) {
        throw std::out_of_range("SRAM tile is outside the configured MEM slice");
    }
    return tile * kBytesPerTileSegment;
}

std::size_t Sram::flat_index(
    std::size_t row, std::size_t byte_offset) const
{
    check_row(row);
    if (byte_offset >= kBytesPerRow) {
        throw std::out_of_range("SRAM byte offset is outside the physical vector row");
    }
    return row * kBytesPerRow + byte_offset;
}

Sram::Page& Sram::ensure_page(std::size_t page)
{
    if (pages_[page] == nullptr) {
        pages_[page] = std::make_unique<Page>();
        pages_[page]->fill(0);
    }
    return *pages_[page];
}

SramArray::SramArray()
    : slices_(hw::kMemSliceColumns)
{
}

void SramArray::set_active_rows(std::size_t rows)
{
    for (auto& sram : slices_) sram.set_active_rows(rows);
}

void SramArray::clear()
{
    for (auto& sram : slices_) {
        sram.clear();
    }
}

Sram& SramArray::slice(std::size_t mem_slice)
{
    return slices_.at(mem_slice);
}

const Sram& SramArray::slice(std::size_t mem_slice) const
{
    return slices_.at(mem_slice);
}

} // namespace ftlpu
