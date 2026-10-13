#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/stream_fabric.hpp"

#include <cstddef>
#include <stdexcept>

namespace ftlpu {

// The VXM contains kVxmAluCount VALU columns interleaved with one more ordinary
// SR column. This wrapper only names the physical attachment points; storage,
// consumption, collision detection, and passive forwarding are exactly the
// same StreamRegisterFabric behavior used by the rest of the chip.
class VxmStreamFabric {
public:
    static constexpr std::size_t kAluColumns = hw::kVxmAluCount;
    static constexpr std::size_t kRegisterColumns = kAluColumns + 1;

    VxmStreamFabric() : fabric_(kRegisterColumns) {}

    StreamRegisterFabric& registers() noexcept { return fabric_; }
    const StreamRegisterFabric& registers() const noexcept { return fabric_; }

    static constexpr std::size_t ingress_column(StreamDirection direction)
    {
        return direction == StreamDirection::East ? 0 : kAluColumns;
    }

    static constexpr std::size_t egress_column(StreamDirection direction)
    {
        return direction == StreamDirection::East ? kAluColumns : 0;
    }

    static std::size_t operand_column(std::size_t alu, StreamId stream)
    {
        check_alu(alu);
        return stream.direction() == StreamDirection::East ? alu : alu + 1;
    }

    static std::size_t result_column(std::size_t alu, StreamId stream)
    {
        check_alu(alu);
        return stream.direction() == StreamDirection::East ? alu + 1 : alu;
    }

    void consume_operand(std::size_t alu, std::size_t tile,
        StreamId stream, const char* consumer = "VXM VALU")
    {
        fabric_.consume_segment(
            operand_column(alu, stream), tile, stream, consumer);
    }

    void stage_result(std::size_t alu, std::size_t tile, StreamId stream,
        const StreamTileSegment& values,
        const char* producer = "VXM VALU")
    {
        fabric_.stage_segment(
            result_column(alu, stream), tile, stream, values, producer);
    }

private:
    static void check_alu(std::size_t alu)
    {
        if (alu >= kAluColumns)
            throw std::out_of_range("VXM VALU column exceeds the configured physical column count");
    }

    StreamRegisterFabric fabric_;
};

} // namespace ftlpu
