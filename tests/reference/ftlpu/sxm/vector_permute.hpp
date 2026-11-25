#pragma once

#include "ftlpu/core/stream_port.hpp"
#include "ftlpu/sxm/permute.hpp"

#include <optional>

namespace ftlpu {

// Three fixed register stages. Each SL forwards one lane segment in each
// direction per hop. A receiver can select bytes only from its local taps.
class SxmVectorPermute {
    static constexpr std::size_t rows = hw::kTileRows;
    static constexpr std::size_t lanes = hw::kLanesPerTile;
    using Segments = std::array<StreamTileSegment, rows>;
    struct Capture {
        SxmInstruction instruction;
        Segments input{};
        std::size_t sl{};
    };
    struct Routing {
        SxmInstruction instruction;
        Segments north{}, south{}, selected{};
        std::size_t hop{};
    };
    struct Drain {
        StreamId destination;
        Segments output;
        std::size_t sl{};
    };
    std::optional<Capture> capture_;
    std::optional<Routing> routing_;
    std::optional<Drain> drain_;

    void route_one_hop()
    {
        auto& r = *routing_;
        for (std::size_t sl = 0; sl < rows; ++sl) {
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                const auto src = r.instruction.permute_map[sl * lanes + lane];
                const auto owner = src / lanes;
                if (owner == sl + r.hop)
                    r.selected[sl][lane] = r.north[sl][src % lanes];
                else if (owner + r.hop == sl)
                    r.selected[sl][lane] = r.south[sl][src % lanes];
            }
        }
        if (++r.hop == rows) {
            if (drain_) throw std::logic_error("vector Permute drain overlap");
            drain_.emplace(Drain{StreamId::from_packed(r.instruction.dst_streams[0].stream),
                                 std::move(r.selected)});
            routing_.reset();
            return;
        }
        // Iterate against the direction of travel to read only old registers.
        for (std::size_t sl = 0; sl + 1 < rows; ++sl)
            r.north[sl] = r.north[sl + 1];
        for (std::size_t sl = rows - 1; sl > 0; --sl)
            r.south[sl] = r.south[sl - 1];
    }

public:
    static constexpr std::size_t initiation_interval = rows;
    static constexpr std::size_t output_offset = 2 * rows;
    static constexpr std::size_t execution_cycles = 3 * rows;

    bool idle() const { return !capture_ && !routing_ && !drain_; }
    bool can_issue() const { return !capture_; }

    void issue(SxmInstruction instruction)
    {
        if (!instruction.is_vector_permute() ||
            (instruction.reads_transpose_bank() ? instruction.input_row >= 2 * lanes
                                                : instruction.input_row != SxmInstruction::kAllInputRows) ||
            instruction.output_row != SxmInstruction::kAllOutputRows ||
            instruction.output_tile != SxmInstruction::kAllOutputTiles)
            throw std::invalid_argument("vector Permute requires one complete input/output stream");
        if (!instruction.reads_transpose_bank()) StreamId::from_packed(instruction.src_streams[0].stream);
        StreamId::from_packed(instruction.dst_streams[0].stream);
        Permute320::validate_bijection(instruction.permute_map);
        if (!can_issue()) throw std::logic_error("vector Permute input stage busy");
        capture_.emplace(Capture{std::move(instruction)});
    }

    template <typename Ports, typename ReadBank>
    void evaluate(StreamRegisterFabric& fabric, const Ports& ports, ReadBank read_bank)
    {
        if (drain_) {
            auto& d = *drain_;
            StreamOutputPort port(fabric, ports.output_column(d.destination.direction()),
                                  d.destination.direction(), "vector Permute");
            port.write_segment(d.sl, d.destination.index(), d.output[d.sl]);
            if (++d.sl == rows) drain_.reset();
        }
        if (routing_) route_one_hop();
        if (capture_) {
            auto& c = *capture_;
            if (c.instruction.reads_transpose_bank()) {
                c.input[c.sl] = read_bank(c.sl, c.instruction.input_row);
            } else {
                const auto src = StreamId::from_packed(c.instruction.src_streams[0].stream);
                StreamInputPort port(fabric, ports.input_column(src.direction()), src.direction(), "vector Permute");
                if (!port.segment_valid(c.sl, src.index()))
                    throw std::logic_error("vector Permute scheduled input segment is missing");
                c.input[c.sl] = port.consume_segment(c.sl, src.index());
            }
            if (++c.sl == rows) {
                if (routing_) throw std::logic_error("vector Permute routing overlap");
                routing_.emplace(Routing{std::move(c.instruction), c.input, c.input});
                capture_.reset();
            }
        }
    }
};
} // namespace ftlpu
