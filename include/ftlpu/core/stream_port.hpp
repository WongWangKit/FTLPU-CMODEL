#pragma once

#include "ftlpu/core/stream_fabric.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

namespace ftlpu {

// Functional slices use ports rather than receiving unrestricted access to
// the whole SR fabric.  An input port observes current-cycle state; an output
// port stages next-cycle state.  The owning system performs the single global
// commit after every slice has evaluated.
class StreamInputPort {
public:
    StreamInputPort(
        StreamRegisterFabric& fabric,
        std::size_t column,
        StreamDirection direction,
        std::string consumer)
        : fabric_(fabric)
        , column_(column)
        , direction_(direction)
        , consumer_(std::move(consumer))
    {
        if (column_ >= fabric_.column_count()) {
            throw std::out_of_range("stream input port column is outside fabric");
        }
    }

    std::size_t column() const noexcept
    {
        return column_;
    }

    StreamDirection direction() const noexcept
    {
        return direction_;
    }

    const StreamCell& cell(
        std::size_t tile,
        std::size_t lane,
        std::size_t stream_index) const
    {
        return fabric_.cell(column_, tile, lane, stream_id(stream_index));
    }

    StreamSegment16 peek_segment(
        std::size_t tile,
        std::size_t stream_index) const
    {
        return fabric_.segment(column_, tile, stream_id(stream_index));
    }

    bool segment_valid(
        std::size_t tile,
        std::size_t stream_index) const
    {
        return fabric_.segment_valid(column_, tile, stream_id(stream_index));
    }

    StreamSegment16 consume_segment(
        std::size_t tile,
        std::size_t stream_index)
    {
        const auto id = stream_id(stream_index);
        auto result = fabric_.segment(column_, tile, id);
        fabric_.consume_segment(column_, tile, id, consumer_.c_str());
        return result;
    }

private:
    StreamId stream_id(std::size_t stream_index) const
    {
        return direction_ == StreamDirection::East
            ? StreamId::East(stream_index)
            : StreamId::West(stream_index);
    }

    StreamRegisterFabric& fabric_;
    std::size_t column_{0};
    StreamDirection direction_{StreamDirection::East};
    std::string consumer_{};
};

class StreamOutputPort {
public:
    StreamOutputPort(
        StreamRegisterFabric& fabric,
        std::size_t column,
        StreamDirection direction,
        std::string producer)
        : fabric_(fabric)
        , column_(column)
        , direction_(direction)
        , producer_(std::move(producer))
    {
        if (column_ >= fabric_.column_count()) {
            throw std::out_of_range("stream output port column is outside fabric");
        }
    }

    std::size_t column() const noexcept
    {
        return column_;
    }

    StreamDirection direction() const noexcept
    {
        return direction_;
    }

    void write_segment(
        std::size_t tile,
        std::size_t stream_index,
        const StreamSegment16& values)
    {
        fabric_.stage_segment(
            column_,
            tile,
            stream_id(stream_index),
            values,
            producer_.c_str());
    }

    void write_payload_segment(
        std::size_t tile,
        std::size_t stream_index,
        const StreamPayloadSegment16& values,
        std::uint64_t vector_tag = 0)
    {
        fabric_.stage_payload_segment(
            column_,
            tile,
            stream_id(stream_index),
            values,
            vector_tag,
            producer_.c_str());
    }

private:
    StreamId stream_id(std::size_t stream_index) const
    {
        return direction_ == StreamDirection::East
            ? StreamId::East(stream_index)
            : StreamId::West(stream_index);
    }

    StreamRegisterFabric& fabric_;
    std::size_t column_{0};
    StreamDirection direction_{StreamDirection::East};
    std::string producer_{};
};

} // namespace ftlpu
