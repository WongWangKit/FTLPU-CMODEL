#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/stream.hpp"

#include <array>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ftlpu {

struct StreamLaneRegisterFile {
    std::array<StreamCell, hw::kEastStreams> east{};
    std::array<StreamCell, hw::kWestStreams> west{};
};

struct StreamRegisterColumn {
    std::array<std::array<StreamLaneRegisterFile, hw::kLanesPerTile>, hw::kTileRows> lanes{};
};

class StreamRegisterFabric {
public:
    struct Link {
        std::size_t source_column{0};
        std::size_t destination_column{0};
        StreamDirection direction{StreamDirection::East};
        bool enabled{true};
    };

    explicit StreamRegisterFabric(std::size_t column_count)
        : current_(column_count)
        , next_(column_count)
        , consumed_(column_count)
        , current_active_(column_count)
        , next_active_(column_count)
    {
        if (column_count == 0) {
            throw std::invalid_argument("stream-register fabric must contain at least one column");
        }
    }

    void reset()
    {
        clear_active_columns(current_, current_active_);
        clear_active_columns(next_, next_active_);
        clear_consumed();
        cycle_ = 0;
        cycle_open_ = false;
    }

    std::size_t column_count() const noexcept
    {
        return current_.size();
    }

    std::size_t cycle() const noexcept
    {
        return cycle_;
    }

    bool cycle_open() const noexcept
    {
        return cycle_open_;
    }

    void begin_cycle()
    {
        if (cycle_open_) {
            throw std::logic_error("stream-register cycle is already open");
        }
        clear_active_columns(next_, next_active_);
        clear_consumed();
        cycle_open_ = true;
    }

    const StreamCell& cell(
        std::size_t column,
        std::size_t tile,
        std::size_t lane,
        StreamId stream) const
    {
        check_location(column, tile, lane, stream);
        return select(current_[column].lanes[tile][lane], stream);
    }

    StreamSegment16 segment(std::size_t column, std::size_t tile, StreamId stream) const
    {
        check_column(column);
        check_tile(tile);
        StreamSegment16 result{};
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            result[lane] = cell(column, tile, lane, stream);
        }
        return result;
    }

    StreamVector320 vector(std::size_t column, StreamId stream) const
    {
        check_column(column);
        StreamVector320 result{};
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            result[tile] = segment(column, tile, stream);
        }
        return result;
    }

    bool segment_valid(std::size_t column, std::size_t tile, StreamId stream) const
    {
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            if (!cell(column, tile, lane, stream).valid) {
                return false;
            }
        }
        return true;
    }

    bool vector_valid(std::size_t column, StreamId stream) const
    {
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            if (!segment_valid(column, tile, stream)) {
                return false;
            }
        }
        return true;
    }

    void stage_write(
        std::size_t column,
        std::size_t tile,
        std::size_t lane,
        StreamId stream,
        StreamCell value,
        const char* producer = "functional slice")
    {
        require_open_cycle();
        check_location(column, tile, lane, stream);
        if (!value.valid) {
            return;
        }

        auto& destination = select(next_[column].lanes[tile][lane], stream);
        if (destination.valid) {
            throw std::logic_error(
                std::string("stream-register write collision at column ")
                + std::to_string(column) + ", tile " + std::to_string(tile)
                + ", lane " + std::to_string(lane) + ", stream "
                + direction_name(stream.direction()) + std::to_string(stream.index())
                + " while staging " + producer);
        }
        destination = value;
        next_active_[column][stream.packed()] = true;
    }

    void stage_segment(
        std::size_t column,
        std::size_t tile,
        StreamId stream,
        const StreamSegment16& values,
        const char* producer = "functional slice")
    {
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            stage_write(column, tile, lane, stream, values[lane], producer);
        }
    }

    void stage_payload_segment(
        std::size_t column,
        std::size_t tile,
        StreamId stream,
        const StreamPayloadSegment16& values,
        std::uint64_t vector_tag = 0,
        const char* producer = "functional slice")
    {
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            stage_write(
                column,
                tile,
                lane,
                stream,
                StreamCell::Valid(values[lane], lane + 1 == hw::kLanesPerTile, vector_tag),
                producer);
        }
    }

    void stage_payload_vector(
        std::size_t column,
        StreamId stream,
        const StreamPayloadVector320& values,
        std::uint64_t vector_tag = 0,
        const char* producer = "functional slice")
    {
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            stage_payload_segment(column, tile, stream, values[tile], vector_tag, producer);
        }
    }

    void consume(
        std::size_t column,
        std::size_t tile,
        std::size_t lane,
        StreamId stream,
        const char* consumer = "functional slice")
    {
        require_open_cycle();
        check_location(column, tile, lane, stream);
        const auto& source = cell(column, tile, lane, stream);
        if (!source.valid) {
            throw std::logic_error(
                std::string("read of invalid stream cell by ") + consumer
                + " at column " + std::to_string(column) + ", tile "
                + std::to_string(tile) + ", lane " + std::to_string(lane));
        }

        const auto cell_index = tile * hw::kLanesPerTile + lane;
        auto& mask = consumed_[column][stream.packed()];
        if (mask.test(cell_index)) {
            throw std::logic_error("stream cell was consumed more than once in the same cycle");
        }
        mask.set(cell_index);
    }

    void consume_segment(
        std::size_t column,
        std::size_t tile,
        StreamId stream,
        const char* consumer = "functional slice")
    {
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            consume(column, tile, lane, stream, consumer);
        }
    }

    void consume_vector(
        std::size_t column,
        StreamId stream,
        const char* consumer = "functional slice")
    {
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            consume_segment(column, tile, stream, consumer);
        }
    }

    // Passive SR-to-SR transfer.  Functional slices should first mark any
    // consumed cells and stage their outputs; then the system stages enabled
    // links.  A consumed value does not continue downstream.
    void stage_link(const Link& link)
    {
        require_open_cycle();
        if (!link.enabled) {
            return;
        }
        check_column(link.source_column);
        check_column(link.destination_column);

        for (std::size_t index = 0; index < hw::kStreamsPerDirection; ++index) {
            const auto stream = link.direction == StreamDirection::East
                ? StreamId::East(index)
                : StreamId::West(index);
            if (!current_active_[link.source_column][stream.packed()]) {
                continue;
            }
            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
                for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                    const auto& source = cell(link.source_column, tile, lane, stream);
                    if (!source.valid || is_consumed(link.source_column, tile, lane, stream)) {
                        continue;
                    }
                    stage_write(
                        link.destination_column,
                        tile,
                        lane,
                        stream,
                        source,
                        "passive SR link");
                }
            }
        }
    }

    void stage_links(const std::vector<Link>& links)
    {
        for (const auto& link : links) {
            stage_link(link);
        }
    }

    // Convenience for the legacy MEM-only 12-column linear path.
    void stage_linear_links()
    {
        require_open_cycle();
        for (std::size_t column = 0; column + 1 < column_count(); ++column) {
            stage_link(Link {column, column + 1, StreamDirection::East, true});
        }
        for (std::size_t column = column_count(); column > 1; --column) {
            stage_link(Link {column - 1, column - 2, StreamDirection::West, true});
        }
    }

    void commit_cycle()
    {
        require_open_cycle();
        current_.swap(next_);
        current_active_.swap(next_active_);
        clear_active_columns(next_, next_active_);
        clear_consumed();
        cycle_open_ = false;
        ++cycle_;
    }

    // Initialization hook for tests/host staging before cycle 0.  Runtime
    // producers should use begin_cycle + stage_write + commit_cycle instead.
    void initialize_cell(
        std::size_t column,
        std::size_t tile,
        std::size_t lane,
        StreamId stream,
        StreamCell value)
    {
        if (cycle_open_) {
            throw std::logic_error("cannot initialize stream fabric during an open cycle");
        }
        check_location(column, tile, lane, stream);
        select(current_[column].lanes[tile][lane], stream) = value;
        current_active_[column][stream.packed()] = value.valid;
    }

private:
    using CellConsumeMask = std::bitset<hw::kPhysicalVectorBytes>;
    using ColumnConsumeMask = std::array<CellConsumeMask, hw::kStreams>;
    using ColumnActiveMask = std::array<bool, hw::kStreams>;

    static StreamCell& select(StreamLaneRegisterFile& lane, StreamId stream)
    {
        return stream.direction() == StreamDirection::East
            ? lane.east[stream.index()]
            : lane.west[stream.index()];
    }

    static const StreamCell& select(const StreamLaneRegisterFile& lane, StreamId stream)
    {
        return stream.direction() == StreamDirection::East
            ? lane.east[stream.index()]
            : lane.west[stream.index()];
    }

    static void clear_active_columns(
        std::vector<StreamRegisterColumn>& columns,
        std::vector<ColumnActiveMask>& active)
    {
        for (std::size_t column = 0; column < columns.size(); ++column) {
            for (std::size_t packed = 0; packed < hw::kStreams; ++packed) {
                if (!active[column][packed]) {
                    continue;
                }
                const auto stream = StreamId::from_packed(packed);
                for (auto& tile : columns[column].lanes) {
                    for (auto& lane : tile) {
                        select(lane, stream).reset();
                    }
                }
                active[column][packed] = false;
            }
        }
    }

    void clear_consumed()
    {
        for (auto& column : consumed_) {
            for (auto& stream : column) {
                stream.reset();
            }
        }
    }

    bool is_consumed(
        std::size_t column,
        std::size_t tile,
        std::size_t lane,
        StreamId stream) const
    {
        return consumed_[column][stream.packed()].test(tile * hw::kLanesPerTile + lane);
    }

    void require_open_cycle() const
    {
        if (!cycle_open_) {
            throw std::logic_error("stream-register operation requires begin_cycle()");
        }
    }

    void check_location(
        std::size_t column,
        std::size_t tile,
        std::size_t lane,
        StreamId stream) const
    {
        check_column(column);
        check_tile(tile);
        check_lane(lane);
        if (stream.index() >= hw::kStreamsPerDirection) {
            throw std::out_of_range("stream index is outside one directional register file");
        }
    }

    void check_column(std::size_t column) const
    {
        if (column >= column_count()) {
            throw std::out_of_range("stream-register column is outside the fabric");
        }
    }

    static void check_tile(std::size_t tile)
    {
        if (tile >= hw::kTileRows) {
            throw std::out_of_range("tile is outside the 20-row slice");
        }
    }

    static void check_lane(std::size_t lane)
    {
        if (lane >= hw::kLanesPerTile) {
            throw std::out_of_range("lane is outside the 16-lane tile");
        }
    }

    static const char* direction_name(StreamDirection direction) noexcept
    {
        return direction == StreamDirection::East ? "E" : "W";
    }

    std::vector<StreamRegisterColumn> current_{};
    std::vector<StreamRegisterColumn> next_{};
    std::vector<ColumnConsumeMask> consumed_{};
    std::vector<ColumnActiveMask> current_active_{};
    std::vector<ColumnActiveMask> next_active_{};
    std::size_t cycle_{0};
    bool cycle_open_{false};
};

} // namespace ftlpu
