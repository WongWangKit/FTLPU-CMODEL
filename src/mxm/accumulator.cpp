#include "ftlpu/mxm/accumulator.hpp"

#include <algorithm>

namespace ftlpu {

MxmAccumulator::MxmAccumulator()
    : rows_(hw::kMxmAccumulatorRows)
{
}

void MxmAccumulator::clear()
{
    std::fill(rows_.begin(), rows_.end(), Row {});
}

void MxmAccumulator::accumulate(
    std::size_t address,
    std::size_t column_block,
    const Segment& values)
{
    auto& row = checked_row(address);
    const auto base = checked_column_base(column_block);
    for (std::size_t lane = 0; lane < values.size(); ++lane) {
        row[base + lane] += values[lane];
    }
}

MxmAccumulator::Segment MxmAccumulator::read(
    std::size_t address,
    std::size_t column_block) const
{
    const auto& row = checked_row(address);
    const auto base = checked_column_base(column_block);
    Segment result{};
    for (std::size_t lane = 0; lane < result.size(); ++lane) {
        result[lane] = row[base + lane];
    }
    return result;
}

void MxmAccumulator::clear_segment(
    std::size_t address,
    std::size_t column_block)
{
    auto& row = checked_row(address);
    const auto base = checked_column_base(column_block);
    std::fill_n(
        row.begin() + static_cast<std::ptrdiff_t>(base),
        hw::kMxmSupercellColumns,
        0.0f);
}

float MxmAccumulator::value(std::size_t address, std::size_t column) const
{
    if (column >= hw::kMxmColumns) {
        throw std::out_of_range("MXM accumulator column is outside the row");
    }
    return checked_row(address)[column];
}

std::size_t MxmAccumulator::checked_column_base(std::size_t column_block)
{
    if (column_block >= hw::kMxmSupercellsPerPlane) {
        throw std::out_of_range("MXM accumulator column block is outside the row");
    }
    return column_block * hw::kMxmSupercellColumns;
}

MxmAccumulator::Row& MxmAccumulator::checked_row(std::size_t address)
{
    return rows_.at(address);
}

const MxmAccumulator::Row& MxmAccumulator::checked_row(std::size_t address) const
{
    return rows_.at(address);
}

} // namespace ftlpu
