#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/timing.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <variant>
#include <vector>

namespace ftlpu {

// Persistent JSON-sized FP32 accumulator local to one MXM.
class MxmAccumulator : public timing::ClockedModule {
public:
    using Row = std::array<float, hw::kMxmColumns>;
    using Segment = std::array<float, hw::kMxmSupercellColumns>;

    MxmAccumulator();

    // Opens one physical-cycle request window. Reads during the window observe
    // the projected value after earlier same-cycle requests, while rows_ itself
    // remains Current until posedge(). This models the legacy accumulator's
    // combinational result path without making SRAM/register state visible early.
    void low() override;
    void validate_posedge() const override;
    void posedge() override;

    void accumulate(
        std::size_t address,
        std::size_t column_block,
        const Segment& values);

    Segment read(std::size_t address, std::size_t column_block) const;

    void clear();
    void clear_segment(std::size_t address, std::size_t column_block);

    float value(std::size_t address, std::size_t column) const;

private:
    struct Accumulate {
        std::size_t address{};
        std::size_t column_block{};
        Segment values{};
    };
    struct ClearSegment {
        std::size_t address{};
        std::size_t column_block{};
    };
    using PendingOperation = std::variant<Accumulate, ClearSegment>;

    static std::size_t checked_column_base(std::size_t column_block);
    Row& checked_row(std::size_t address);
    const Row& checked_row(std::size_t address) const;
    Segment projected_segment(std::size_t address, std::size_t column_block) const;
    void apply(const Accumulate& operation);
    void apply(const ClearSegment& operation);

    std::vector<Row> rows_{};
    std::vector<PendingOperation> pending_operations_{};
    bool cycle_open_{};
};

} // namespace ftlpu
