#include "ftlpu/mxm/block_accumulator.hpp"

#include <algorithm>
#include <stdexcept>

namespace ftlpu {

MxmBlockAccumulator::MxmBlockAccumulator()
    : rows_(hw::kMxmBlockAccumulatorRows)
{
}

void MxmBlockAccumulator::clear()
{
    std::fill(rows_.begin(), rows_.end(), Row {});
}

void MxmBlockAccumulator::accumulate(
    std::size_t address,
    std::size_t column_block,
    const Segment& values)
{
    auto& row = checked_row(address);
    const auto base = checked_column_base(column_block);
    for (std::size_t output_row = 0;
         output_row < hw::kMxmBlockRows;
         ++output_row) {
        for (std::size_t lane = 0; lane < hw::kMxmSupercellColumns; ++lane) {
            row[output_row][base + lane] += values[output_row][lane];
        }
    }
}

MxmBlockAccumulator::Segment MxmBlockAccumulator::read(
    std::size_t address,
    std::size_t column_block) const
{
    const auto& row = checked_row(address);
    const auto base = checked_column_base(column_block);
    Segment result {};
    for (std::size_t output_row = 0;
         output_row < hw::kMxmBlockRows;
         ++output_row) {
        std::copy_n(
            row[output_row].begin() + static_cast<std::ptrdiff_t>(base),
            hw::kMxmSupercellColumns,
            result[output_row].begin());
    }
    return result;
}

void MxmBlockAccumulator::clear_segment(
    std::size_t address,
    std::size_t column_block)
{
    auto& row = checked_row(address);
    const auto base = checked_column_base(column_block);
    for (auto& output_row : row) {
        std::fill_n(
            output_row.begin() + static_cast<std::ptrdiff_t>(base),
            hw::kMxmSupercellColumns,
            0.0f);
    }
}

float MxmBlockAccumulator::value(
    std::size_t address,
    std::size_t output_row,
    std::size_t column) const
{
    if (output_row >= hw::kMxmBlockRows) {
        throw std::out_of_range(
            "MXM block accumulator output row is outside the SRAM row");
    }
    if (column >= hw::kMxmColumns) {
        throw std::out_of_range(
            "MXM block accumulator column is outside the SRAM row");
    }
    return checked_row(address)[output_row][column];
}

std::size_t MxmBlockAccumulator::checked_column_base(
    std::size_t column_block)
{
    if (column_block >= hw::kMxmSupercellsPerPlane) {
        throw std::out_of_range(
            "MXM block accumulator column block is outside the SRAM row");
    }
    return column_block * hw::kMxmSupercellColumns;
}

MxmBlockAccumulator::Row& MxmBlockAccumulator::checked_row(
    std::size_t address)
{
    return rows_.at(address);
}

const MxmBlockAccumulator::Row& MxmBlockAccumulator::checked_row(
    std::size_t address) const
{
    return rows_.at(address);
}

} // namespace ftlpu
