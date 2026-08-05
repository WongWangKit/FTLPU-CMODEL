#pragma once

#include "ftlpu/core/hardware_params.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace ftlpu {

// Persistent 1 MiB FP32 accumulator with one 8x32 block per SRAM row.
class MxmBlockAccumulator {
public:
    using SegmentRow = std::array<float, hw::kMxmSupercellColumns>;
    using Segment = std::array<SegmentRow, hw::kMxmBlockRows>;
    using Row = std::array<
        std::array<float, hw::kMxmColumns>,
        hw::kMxmBlockRows>;

    MxmBlockAccumulator();

    void accumulate(
        std::size_t address,
        std::size_t column_block,
        const Segment& values);

    Segment read(std::size_t address, std::size_t column_block) const;

    void clear();
    void clear_segment(std::size_t address, std::size_t column_block);

    float value(
        std::size_t address,
        std::size_t output_row,
        std::size_t column) const;

private:
    static std::size_t checked_column_base(std::size_t column_block);
    Row& checked_row(std::size_t address);
    const Row& checked_row(std::size_t address) const;

    std::vector<Row> rows_{};
};

} // namespace ftlpu
