#include "ftlpu/mxm/accumulator.hpp"

#include <algorithm>
#include <type_traits>

namespace ftlpu {

MxmAccumulator::MxmAccumulator()
    : rows_(hw::kMxmAccumulatorRows)
{
}

void MxmAccumulator::low()
{
    if (cycle_open_) {
        throw std::logic_error("MXM accumulator LOW executed more than once");
    }
    pending_operations_.clear();
    cycle_open_ = true;
}

void MxmAccumulator::validate_posedge() const
{
    if (!cycle_open_) {
        throw std::logic_error("MXM accumulator committed without LOW");
    }
    for (const auto& pending : pending_operations_) {
        std::visit([this](const auto& operation) {
            (void)checked_row(operation.address);
            (void)checked_column_base(operation.column_block);
        }, pending);
    }
}

void MxmAccumulator::posedge()
{
    validate_posedge();
    for (const auto& pending : pending_operations_) {
        std::visit([this](const auto& operation) { apply(operation); }, pending);
    }
    pending_operations_.clear();
    cycle_open_ = false;
}

void MxmAccumulator::clear()
{
    if (cycle_open_) {
        throw std::logic_error("MXM accumulator cannot reset during an open cycle");
    }
    std::fill(rows_.begin(), rows_.end(), Row {});
    pending_operations_.clear();
}

void MxmAccumulator::accumulate(
    std::size_t address,
    std::size_t column_block,
    const Segment& values)
{
    (void)checked_row(address);
    (void)checked_column_base(column_block);
    if (!cycle_open_) {
        throw std::logic_error("MXM accumulator write requires LOW");
    }
    pending_operations_.push_back(Accumulate {address, column_block, values});
}

MxmAccumulator::Segment MxmAccumulator::read(
    std::size_t address,
    std::size_t column_block) const
{
    return projected_segment(address, column_block);
}

void MxmAccumulator::clear_segment(
    std::size_t address,
    std::size_t column_block)
{
    (void)checked_row(address);
    (void)checked_column_base(column_block);
    if (!cycle_open_) {
        throw std::logic_error("MXM accumulator clear requires LOW");
    }
    pending_operations_.push_back(ClearSegment {address, column_block});
}

float MxmAccumulator::value(std::size_t address, std::size_t column) const
{
    if (column >= hw::kMxmColumns) {
        throw std::out_of_range("MXM accumulator column is outside the row");
    }
    const auto column_block = column / hw::kMxmSupercellColumns;
    const auto lane = column % hw::kMxmSupercellColumns;
    return projected_segment(address, column_block)[lane];
}

MxmAccumulator::Segment MxmAccumulator::projected_segment(
    std::size_t address,
    std::size_t column_block) const
{
    const auto& row = checked_row(address);
    const auto base = checked_column_base(column_block);
    Segment result{};
    std::copy_n(row.begin() + static_cast<std::ptrdiff_t>(base), result.size(), result.begin());
    if (!cycle_open_) return result;
    for (const auto& pending : pending_operations_) {
        std::visit([&](const auto& operation) {
            if (operation.address != address || operation.column_block != column_block) return;
            using Operation = std::decay_t<decltype(operation)>;
            if constexpr (std::is_same_v<Operation, Accumulate>) {
                for (std::size_t lane = 0; lane < result.size(); ++lane)
                    result[lane] += operation.values[lane];
            } else {
                result.fill(0.0f);
            }
        }, pending);
    }
    return result;
}

void MxmAccumulator::apply(const Accumulate& operation)
{
    auto& row = checked_row(operation.address);
    const auto base = checked_column_base(operation.column_block);
    for (std::size_t lane = 0; lane < operation.values.size(); ++lane)
        row[base + lane] += operation.values[lane];
}

void MxmAccumulator::apply(const ClearSegment& operation)
{
    auto& row = checked_row(operation.address);
    const auto base = checked_column_base(operation.column_block);
    std::fill_n(row.begin() + static_cast<std::ptrdiff_t>(base),
        hw::kMxmSupercellColumns, 0.0f);
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
