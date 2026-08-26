#pragma once

#include "ftlpu/core/hardware_params.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ftlpu {

// Persistent JSON-sized FP32 accumulator local to one MXM.
class MxmAccumulator {
public:
    using Row = std::array<float, hw::kMxmColumns>;
    using Segment = std::array<float, hw::kMxmSupercellColumns>;

    MxmAccumulator();

    void accumulate(
        std::size_t address,
        std::size_t column_block,
        const Segment& values);

    Segment read(std::size_t address, std::size_t column_block) const;

    void clear();
    void clear_segment(std::size_t address, std::size_t column_block);

    float value(std::size_t address, std::size_t column) const;

private:
    static std::size_t checked_column_base(std::size_t column_block);
    Row& checked_row(std::size_t address);
    const Row& checked_row(std::size_t address) const;

    std::vector<Row> rows_{};
};

} // namespace ftlpu
