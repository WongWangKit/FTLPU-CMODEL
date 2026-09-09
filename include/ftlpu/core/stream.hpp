#pragma once

#include "ftlpu/core/hardware_params.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace ftlpu {

// A logical TSP stream is identified by a lane-local index (0..31) and a
// direction.  The same StreamId names all byte lanes of a functional
// slice; it is not a scalar channel owned by one lane.
enum class StreamDirection : std::uint8_t {
    East,
    West,
};

class StreamId {
public:
    constexpr StreamId() = default;

    static StreamId East(std::size_t index)
    {
        return StreamId(StreamDirection::East, index);
    }

    static StreamId West(std::size_t index)
    {
        return StreamId(StreamDirection::West, index);
    }

    static StreamId from_packed(std::size_t packed)
    {
        if (packed >= hw::kStreams) {
            throw std::out_of_range("packed stream selector is outside the 64 encoded streams");
        }
        return packed < hw::kEastStreams
            ? East(packed)
            : West(packed - hw::kEastStreams);
    }

    constexpr StreamDirection direction() const noexcept
    {
        return direction_;
    }

    constexpr std::size_t index() const noexcept
    {
        return index_;
    }

    constexpr std::size_t packed() const noexcept
    {
        return direction_ == StreamDirection::East ? index_ : hw::kEastStreams + index_;
    }

private:
    StreamId(StreamDirection direction, std::size_t index)
        : direction_(direction)
        , index_(index)
    {
        if (index >= hw::kStreamsPerDirection) {
            throw std::out_of_range("stream id is outside the 32 streams in one direction");
        }
    }

    StreamDirection direction_{StreamDirection::East};
    std::size_t index_{0};
};

inline constexpr bool operator==(StreamId lhs, StreamId rhs) noexcept
{
    return lhs.direction() == rhs.direction() && lhs.index() == rhs.index();
}

inline constexpr bool operator!=(StreamId lhs, StreamId rhs) noexcept
{
    return !(lhs == rhs);
}

// Generic packet helper retained for non-physical, typed unit tests and
// arithmetic block interfaces.  Physical stream-register storage uses
// StreamCell below.
template <typename T>
struct StreamWord {
    T data{};
    bool last{false};
};

template <typename T>
using StreamValue = std::optional<StreamWord<T>>;

// One physical byte register at one (SR column, tile, lane, StreamId).
// valid distinguishes an empty register from a valid byte whose value is 0.
// vector_tag is diagnostic metadata for detecting stale/misaligned vectors;
// hardware need not physically store it.
struct StreamCell {
    std::uint8_t data{0};
    bool valid{false};
    bool last{false};
    std::uint64_t vector_tag{0};

    static constexpr StreamCell Invalid() noexcept
    {
        return {};
    }

    static constexpr StreamCell Valid(
        std::uint8_t value,
        bool is_last = false,
        std::uint64_t tag = 0) noexcept
    {
        return StreamCell {value, true, is_last, tag};
    }

};

// Spatial views of one logical stream. A vector is a logical/debug aggregation,
// not an atomic transfer. Hardware-facing slices act on one tile segment per
// cycle; northbound instruction staggering can skew segments across cycles.
using StreamTileSegment = std::array<StreamCell, hw::kLanesPerTile>;
using StreamSliceVector = std::array<StreamTileSegment, hw::kTileRows>;
using StreamPayloadTileSegment = std::array<std::uint8_t, hw::kLanesPerTile>;
using StreamPayloadSliceVector =
    std::array<StreamPayloadTileSegment, hw::kTileRows>;

} // namespace ftlpu
