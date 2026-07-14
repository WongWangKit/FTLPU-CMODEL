#pragma once

#include "ftlpu/core/stream_port.hpp"
#include "ftlpu/sxm/distributor.hpp"
#include "ftlpu/sxm/permute.hpp"
#include "ftlpu/sxm/shift.hpp"
#include "ftlpu/sxm/transpose.hpp"
#include "ftlpu/sxm/unit_group.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ftlpu {

class SxmStreamPortMap {
public:
    static SxmStreamPortMap BetweenColumns(
        std::size_t east_input,
        std::size_t east_output,
        std::size_t west_input,
        std::size_t west_output)
    {
        return SxmStreamPortMap(east_input, east_output, west_input, west_output);
    }

    static SxmStreamPortMap SameDirection(std::size_t input, std::size_t output)
    {
        return SxmStreamPortMap(input, output, input, output);
    }

    std::size_t input_column(StreamDirection direction) const noexcept
    {
        return direction == StreamDirection::East ? east_input_ : west_input_;
    }

    std::size_t output_column(StreamDirection direction) const noexcept
    {
        return direction == StreamDirection::East ? east_output_ : west_output_;
    }

    void validate_for(const StreamRegisterFabric& fabric) const
    {
        if (east_input_ >= fabric.column_count()
            || east_output_ >= fabric.column_count()
            || west_input_ >= fabric.column_count()
            || west_output_ >= fabric.column_count()) {
            throw std::out_of_range("SXM port maps outside stream-register fabric");
        }
    }

private:
    SxmStreamPortMap(
        std::size_t east_input,
        std::size_t east_output,
        std::size_t west_input,
        std::size_t west_output)
        : east_input_(east_input)
        , east_output_(east_output)
        , west_input_(west_input)
        , west_output_(west_output)
    {
    }

    std::size_t east_input_{0};
    std::size_t east_output_{0};
    std::size_t west_input_{0};
    std::size_t west_output_{0};
};

// SR-facing SXM functional slice.  It owns instruction issue state only; the
// StreamRegisterFabric remains the sole owner of all live stream data.
class SxmSlice {
public:
    using UnitGroup = SxmUnitGroup<std::uint8_t>;

    explicit SxmSlice(SxmStreamPortMap ports)
        : ports_(std::move(ports))
    {
    }

    void reset()
    {
        units_.reset();
    }

    std::size_t cycle() const noexcept
    {
        return units_.cycle();
    }

    bool can_issue(const SxmInstruction& instruction) const
    {
        return units_.can_issue(instruction);
    }

    void issue(SxmInstruction instruction)
    {
        units_.issue(std::move(instruction));
    }

    const SxmStreamPortMap& ports() const noexcept
    {
        return ports_;
    }

    void evaluate(StreamRegisterFabric& fabric)
    {
        ports_.validate_for(fabric);
        if (!fabric.cycle_open()) {
            throw std::logic_error("SxmSlice::evaluate requires an open SR cycle");
        }

        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            auto inputs = read_and_consume_inputs(fabric, tile);
            const auto result = units_.evaluate(inputs);
            write_outputs(fabric, tile, result);
        }
        units_.complete_cycle();
    }

    template <typename T>
    using TileVector = std::array<T, hw::kLanesPerTile>;

    template <typename T>
    using StreamVector = std::array<TileVector<T>, hw::kTileRows>;

    template <typename T>
    using Matrix16 = std::array<TileVector<T>, hw::kLanesPerTile>;

    template <typename T>
    static TileVector<T> distribute(const TileVector<T>& input, const Distribute16::Map& map, T zero = T{})
    {
        return Distribute16::apply(input, map, zero);
    }

    template <typename T>
    static Matrix16<T> transpose(const Matrix16<T>& input)
    {
        return Transpose16x16::apply(input);
    }

    template <typename T>
    static StreamVector<T> shift_select(
        const StreamVector<T>& input,
        SxmShiftSource source,
        std::size_t distance = 1,
        T zero = T{})
    {
        return ShiftSelect::apply(input, source, distance, zero);
    }

    template <typename T>
    static StreamVector<T> permute(const StreamVector<T>& input, const Permute320::Map& map)
    {
        return Permute320::apply(input, map);
    }

private:
    using StreamState = UnitGroup::StreamState;
    using Evaluation = UnitGroup::Evaluation;

    static StreamId physical_stream(SxmStreamId stream)
    {
        return StreamId::from_packed(stream.stream);
    }

    StreamState read_and_consume_inputs(StreamRegisterFabric& fabric, std::size_t tile) const
    {
        StreamState inputs{};
        std::array<bool, hw::kStreams> required{};
        for (const auto& instruction : units_.issued_instructions()) {
            for (const auto stream : instruction.src_streams) {
                required[stream.stream] = true;
            }
        }

        // Validate every operand before consuming any of them, so a missing
        // operand cannot leave a partially consumed fabric cycle.
        for (std::size_t packed = 0; packed < required.size(); ++packed) {
            if (!required[packed]) {
                continue;
            }
            const auto id = StreamId::from_packed(packed);
            StreamInputPort input(
                fabric,
                ports_.input_column(id.direction()),
                id.direction(),
                "SXM");
            if (!input.segment_valid(tile, id.index())) {
                throw std::logic_error("SXM source stream segment is not available");
            }
        }

        for (std::size_t packed = 0; packed < required.size(); ++packed) {
            if (!required[packed]) {
                continue;
            }
            const auto id = StreamId::from_packed(packed);
            StreamInputPort input(
                fabric,
                ports_.input_column(id.direction()),
                id.direction(),
                "SXM");
            const auto segment = input.consume_segment(tile, id.index());
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                inputs[packed][lane] = UnitGroup::Word {
                    segment[lane].data,
                    segment[lane].last,
                };
            }
        }
        return inputs;
    }

    void write_outputs(
        StreamRegisterFabric& fabric,
        std::size_t tile,
        const Evaluation& result) const
    {
        for (std::size_t packed = 0; packed < result.produced.size(); ++packed) {
            if (!result.produced[packed]) {
                continue;
            }
            const auto id = StreamId::from_packed(packed);
            StreamSegment16 segment{};
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                const auto& word = result.outputs[packed][lane];
                if (!word.has_value()) {
                    throw std::logic_error("SXM produced an incomplete stream segment");
                }
                segment[lane] = StreamCell::Valid(word->data, word->last);
            }
            StreamOutputPort output(
                fabric,
                ports_.output_column(id.direction()),
                id.direction(),
                "SXM");
            output.write_segment(tile, id.index(), segment);
        }
    }

    SxmStreamPortMap ports_;
    UnitGroup units_{};
};

} // namespace ftlpu
