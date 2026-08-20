#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/stream.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
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
    struct CycleActivity {
        std::size_t staged_writes{0};
        std::size_t east_staged_writes{0};
        std::size_t west_staged_writes{0};
    };

    struct Link {
        std::size_t source_column{0};
        std::size_t destination_column{0};
        StreamDirection direction{StreamDirection::East};
        bool enabled{true};
    };

    explicit StreamRegisterFabric(std::size_t column_count)
        : current_(column_count)
        , next_(column_count)
        , next_producers_(column_count)
        , consumed_(column_count)
    {
        if (column_count == 0) {
            throw std::invalid_argument("stream-register fabric must contain at least one column");
        }
    }

    void reset()
    {
        clear_columns(current_);
        clear_columns(next_);
        clear_producers();
        clear_consumed();
        current_cells_.clear();
        next_cells_.clear();
        consumed_cells_.clear();
        cycle_ = 0;
        cycle_open_ = false;
        current_activity_ = {};
        last_activity_ = {};
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

    const CycleActivity& last_cycle_activity() const noexcept
    {
        return last_activity_;
    }

    void begin_cycle()
    {
        if (cycle_open_) {
            throw std::logic_error("stream-register cycle is already open");
        }
        // commit_cycle() leaves the staging state clean. Do not rescan the
        // complete fabric a second time at the start of every cycle.
        current_activity_ = {};
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

    StreamTileSegment segment(std::size_t column, std::size_t tile, StreamId stream) const
    {
        check_column(column);
        check_tile(tile);
        StreamTileSegment result{};
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            result[lane] = cell(column, tile, lane, stream);
        }
        return result;
    }

    StreamSliceVector vector(std::size_t column, StreamId stream) const
    {
        check_column(column);
        StreamSliceVector result{};
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
            const auto existing = select(
                next_producers_[column].lanes[tile][lane], stream);
            throw std::logic_error(
                std::string("stream-register write collision at column ")
                + std::to_string(column) + ", tile " + std::to_string(tile)
                + ", lane " + std::to_string(lane) + ", stream "
                + direction_name(stream.direction()) + std::to_string(stream.index())
                + " while staging " + producer + "; existing producer="
                + producer_name(existing));
        }
        destination = value;
        select(next_producers_[column].lanes[tile][lane], stream) =
            classify_producer(producer);
        next_cells_.push_back(encode_cell(column, tile, lane, stream));
        ++current_activity_.staged_writes;
        if (stream.direction() == StreamDirection::East)
            ++current_activity_.east_staged_writes;
        else
            ++current_activity_.west_staged_writes;
    }

    void stage_segment(
        std::size_t column,
        std::size_t tile,
        StreamId stream,
        const StreamTileSegment& values,
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
        const StreamPayloadTileSegment& values,
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

        // Consumption is a broadcast read. Multiple functional units may
        // observe the same current-cycle cell; the shared flag only suppresses
        // passive forwarding after at least one consumer has read it.
        auto& consumed = select(consumed_[column].lanes[tile][lane], stream);
        if (!consumed)
            consumed_cells_.push_back(
                encode_cell(column, tile, lane, stream));
        consumed = true;
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

    // Passive SR-to-SR transfer.  Functional slices should first mark any
    // consumed cells and stage their outputs; then the system stages enabled
    // links. Multiple consumers may read a cell, but a value consumed by any
    // functional unit does not continue downstream.
    void stage_link(const Link& link)
    {
        require_open_cycle();
        if (!link.enabled) {
            return;
        }
        check_column(link.source_column);
        check_column(link.destination_column);

        for (const std::size_t encoded : current_cells_) {
            const auto location = decode_cell(encoded);
            if (location.column != link.source_column
                || location.stream.direction() != link.direction
                || is_consumed(location.column, location.tile,
                    location.lane, location.stream))
                continue;
            stage_write(link.destination_column, location.tile,
                location.lane, location.stream,
                cell(location.column, location.tile,
                    location.lane, location.stream),
                "passive SR link");
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
        for (const std::size_t encoded : current_cells_) {
            const auto location = decode_cell(encoded);
            if (is_consumed(location.column, location.tile,
                    location.lane, location.stream))
                continue;
            const bool east = location.stream.direction()
                == StreamDirection::East;
            if ((east && location.column + 1 >= column_count())
                || (!east && location.column == 0))
                continue;
            const std::size_t destination = east
                ? location.column + 1 : location.column - 1;
            stage_write(destination, location.tile, location.lane,
                location.stream,
                cell(location.column, location.tile,
                    location.lane, location.stream),
                "passive SR link");
        }
    }

    void commit_cycle()
    {
        require_open_cycle();
        current_.swap(next_);
        last_activity_ = current_activity_;
        for (const std::size_t encoded : current_cells_) {
            const auto location = decode_cell(encoded);
            select(next_[location.column].lanes[location.tile][location.lane],
                location.stream) = StreamCell::Invalid();
        }
        current_cells_.swap(next_cells_);
        next_cells_.clear();
        for (const std::size_t encoded : current_cells_) {
            const auto location = decode_cell(encoded);
            select(next_producers_[location.column]
                       .lanes[location.tile][location.lane],
                location.stream) = ProducerKind::None;
        }
        for (const std::size_t encoded : consumed_cells_) {
            const auto location = decode_cell(encoded);
            select(consumed_[location.column]
                       .lanes[location.tile][location.lane],
                location.stream) = false;
        }
        consumed_cells_.clear();
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
        auto& destination = select(
            current_[column].lanes[tile][lane], stream);
        if (!destination.valid && value.valid)
            current_cells_.push_back(
                encode_cell(column, tile, lane, stream));
        else if (destination.valid && !value.valid) {
            const std::size_t encoded =
                encode_cell(column, tile, lane, stream);
            std::erase(current_cells_, encoded);
        }
        destination = value;
    }

private:
    struct CellLocation {
        std::size_t column{0};
        std::size_t tile{0};
        std::size_t lane{0};
        StreamId stream{StreamId::East(0)};
    };

    struct LaneConsumeMask {
        std::array<bool, hw::kEastStreams> east{};
        std::array<bool, hw::kWestStreams> west{};
    };

    struct ColumnConsumeMask {
        std::array<std::array<LaneConsumeMask, hw::kLanesPerTile>, hw::kTileRows> lanes{};
    };

    enum class ProducerKind : std::uint8_t {
        None,
        MemRead,
        Sxm,
        SxmBlock,
        External,
        C2c,
        Other,
    };

    struct LaneProducerMap {
        std::array<ProducerKind, hw::kEastStreams> east{};
        std::array<ProducerKind, hw::kWestStreams> west{};
    };

    struct ColumnProducerMap {
        std::array<std::array<LaneProducerMap, hw::kLanesPerTile>, hw::kTileRows>
            lanes{};
    };

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

    static bool& select(LaneConsumeMask& lane, StreamId stream)
    {
        return stream.direction() == StreamDirection::East
            ? lane.east[stream.index()]
            : lane.west[stream.index()];
    }

    static const bool& select(const LaneConsumeMask& lane, StreamId stream)
    {
        return stream.direction() == StreamDirection::East
            ? lane.east[stream.index()]
            : lane.west[stream.index()];
    }

    static ProducerKind& select(LaneProducerMap& lane, StreamId stream)
    {
        return stream.direction() == StreamDirection::East
            ? lane.east[stream.index()]
            : lane.west[stream.index()];
    }

    static ProducerKind classify_producer(const char* producer)
    {
        const std::string_view name = producer != nullptr
            ? std::string_view(producer) : std::string_view {};
        if (name == "MEM Read") return ProducerKind::MemRead;
        if (name == "SXM") return ProducerKind::Sxm;
        if (name == "SXM parallel block output") return ProducerKind::SxmBlock;
        if (name == "external stream input") return ProducerKind::External;
        if (name.find("c2c") != std::string_view::npos
            || name.find("C2C") != std::string_view::npos)
            return ProducerKind::C2c;
        return name.empty() ? ProducerKind::None : ProducerKind::Other;
    }

    static const char* producer_name(ProducerKind producer)
    {
        switch (producer) {
        case ProducerKind::None: return "none";
        case ProducerKind::MemRead: return "MEM Read";
        case ProducerKind::Sxm: return "SXM";
        case ProducerKind::SxmBlock: return "SXM parallel block output";
        case ProducerKind::External: return "external stream input";
        case ProducerKind::C2c: return "C2C";
        case ProducerKind::Other: return "other functional producer";
        }
        return "unknown";
    }

    static std::size_t encode_cell(
        std::size_t column,
        std::size_t tile,
        std::size_t lane,
        StreamId stream)
    {
        return (((column * hw::kTileRows + tile)
                    * hw::kLanesPerTile + lane)
                   * hw::kStreams)
            + stream.packed();
    }

    static CellLocation decode_cell(std::size_t encoded)
    {
        const auto stream = StreamId::from_packed(encoded % hw::kStreams);
        encoded /= hw::kStreams;
        const std::size_t lane = encoded % hw::kLanesPerTile;
        encoded /= hw::kLanesPerTile;
        const std::size_t tile = encoded % hw::kTileRows;
        return CellLocation {
            encoded / hw::kTileRows, tile, lane, stream};
    }

    static void clear_columns(std::vector<StreamRegisterColumn>& columns)
    {
        std::fill(columns.begin(), columns.end(), StreamRegisterColumn {});
    }

    void clear_consumed()
    {
        std::fill(consumed_.begin(), consumed_.end(), ColumnConsumeMask {});
    }

    void clear_producers()
    {
        std::fill(next_producers_.begin(), next_producers_.end(),
            ColumnProducerMap {});
    }

    bool is_consumed(
        std::size_t column,
        std::size_t tile,
        std::size_t lane,
        StreamId stream) const
    {
        return select(consumed_[column].lanes[tile][lane], stream);
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
            throw std::out_of_range("tile is outside the configured slice");
        }
    }

    static void check_lane(std::size_t lane)
    {
        if (lane >= hw::kLanesPerTile) {
            throw std::out_of_range("lane is outside the configured tile");
        }
    }

    static const char* direction_name(StreamDirection direction) noexcept
    {
        return direction == StreamDirection::East ? "E" : "W";
    }

    std::vector<StreamRegisterColumn> current_{};
    std::vector<StreamRegisterColumn> next_{};
    std::vector<ColumnProducerMap> next_producers_{};
    std::vector<ColumnConsumeMask> consumed_{};
    std::vector<std::size_t> current_cells_{};
    std::vector<std::size_t> next_cells_{};
    std::vector<std::size_t> consumed_cells_{};
    std::size_t cycle_{0};
    bool cycle_open_{false};
    CycleActivity current_activity_{};
    CycleActivity last_activity_{};
};

} // namespace ftlpu
