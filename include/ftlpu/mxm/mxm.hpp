#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/stream_port.hpp"
#include "ftlpu/mxm/array.hpp"
#include "ftlpu/mxm/control_slice.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <vector>

namespace ftlpu {

class MxmStreamPortMap {
public:
    struct InputEndpoint {
        std::size_t column{0};
        StreamDirection direction{StreamDirection::East};
        bool multicast{false};
    };

    struct WeightEndpoint : InputEndpoint {
        std::size_t stream_base{0};
    };

    struct OutputEndpoint {
        std::size_t column{0};
        StreamDirection direction{StreamDirection::West};
    };

    static MxmStreamPortMap AtBoundary(
        std::size_t column,
        std::size_t weight_stream_base = 0)
    {
        return MxmStreamPortMap {
            WeightEndpoint {{column, StreamDirection::East, false}, weight_stream_base},
            InputEndpoint {column, StreamDirection::East, true},
            OutputEndpoint {column, StreamDirection::West},
        };
    }

    MxmStreamPortMap(
        WeightEndpoint weight_input,
        InputEndpoint activation_input,
        OutputEndpoint result_output)
        : weight_input_(weight_input)
        , activation_input_(activation_input)
        , result_output_(result_output)
    {
        if (weight_input_.stream_base + hw::kMxmLoadStreamsPerCycle
            > hw::kStreamsPerDirection) {
            throw std::out_of_range("MXM weight stream range exceeds one stream direction");
        }
    }

    const WeightEndpoint& weight_input() const noexcept
    {
        return weight_input_;
    }

    const InputEndpoint& activation_input() const noexcept
    {
        return activation_input_;
    }

    const OutputEndpoint& result_output() const noexcept
    {
        return result_output_;
    }

    void validate_for(const StreamRegisterFabric& fabric) const
    {
        if (weight_input_.column >= fabric.column_count()
            || activation_input_.column >= fabric.column_count()
            || result_output_.column >= fabric.column_count()) {
            throw std::out_of_range("MXM port maps outside stream-register fabric");
        }
    }

private:
    WeightEndpoint weight_input_{};
    InputEndpoint activation_input_{};
    OutputEndpoint result_output_{};
};

class Mxm {
public:
    static constexpr std::size_t kWeightBuffers = MxmSupercell::kWeightBuffers;
    static constexpr std::size_t kAccumulatorBanks = hw::kMxmAccumulatorBanks;
    using ActivationData = std::array<std::int8_t, hw::kLanesPerTile>;
    using ResultValues = std::array<std::int32_t, hw::kMxmSupercellColumns>;
    using PartialSum = MxmSupercell::PartialSum;

    struct ColumnOutput {
        std::size_t row{0};
        std::size_t column_block{0};
        ResultValues values{};
        std::size_t accumulator_bank{0};
    };

    Mxm()
        : Mxm(MxmStreamPortMap::AtBoundary(
            hw::kMemBoundaryStreamRegisterColumns - 1))
    {
    }

    explicit Mxm(MxmStreamPortMap ports)
        : control_(array_)
        , ports_(std::move(ports))
    {
        reset_datapath();
    }

    Mxm(const Mxm&) = delete;
    Mxm& operator=(const Mxm&) = delete;

    void reset()
    {
        array_.reset();
        control_.reset();
        reset_datapath();
    }

    MxmArray& array()
    {
        return array_;
    }

    const MxmArray& array() const
    {
        return array_;
    }

    MxmControlSlice& control()
    {
        return control_;
    }

    const MxmControlSlice& control() const
    {
        return control_;
    }

    const MxmStreamPortMap& ports() const noexcept
    {
        return ports_;
    }

    void set_stream_ports(MxmStreamPortMap ports)
    {
        ports_ = std::move(ports);
    }

    void reset_datapath()
    {
        for (auto& column : east_pipeline_) {
            for (auto& lane : column) {
                lane.clear();
            }
        }
        for (auto& row : north_pipeline_) {
            for (auto& column : row) {
                column.clear();
            }
        }
        for (auto& buffer : accumulator_banks_) {
            for (auto& row : buffer) {
                row.fill(0);
            }
        }
        for (auto& row : last_computing_) {
            row.fill(false);
        }
        for (auto& cursor : next_row_for_tile_) {
            cursor.fill(0);
        }
        for (auto& bank : compute_active_by_accumulator_tile_) {
            bank.fill(false);
        }
        last_outputs_.clear();
        active_ = false;
    }

    void evaluate(
        StreamRegisterFabric& fabric,
        std::size_t mxm_id,
        std::ostream* os = nullptr,
        std::optional<std::size_t> log_tile = std::nullopt)
    {
        evaluate_control(fabric, mxm_id, os, log_tile);
        evaluate_datapath(fabric, mxm_id, os, log_tile);
    }

    void evaluate_control(
        StreamRegisterFabric& fabric,
        std::size_t mxm_id,
        std::ostream* os = nullptr,
        std::optional<std::size_t> log_tile = std::nullopt)
    {
        require_open_fabric(fabric);
        auto provider = [this, &fabric, mxm_id, os, log_tile](std::size_t tile) {
            if (os != nullptr && (!log_tile.has_value() || tile == *log_tile)) {
                *os << "  SR -> MXM" << mxm_id << " weights tile " << tile << '\n';
            }
            return collect_weight_input(fabric, tile);
        };
        if (os != nullptr) {
            control_.tick(*os, provider, false, log_tile);
        } else {
            static NullStream null_stream;
            control_.tick(null_stream.stream(), provider, false, log_tile);
        }
    }

    void evaluate_datapath(
        StreamRegisterFabric& fabric,
        std::size_t mxm_id,
        std::ostream* os = nullptr,
        std::optional<std::size_t> log_tile = std::nullopt)
    {
        require_open_fabric(fabric);
        last_outputs_.clear();

        if (!active_ && control_.compute_active(0)) {
            active_ = true;
        }

        auto current_compute_active_by_accumulator_tile =
            std::array<std::array<bool, hw::kMxmSupercellsPerPlane>, kAccumulatorBanks> {};
        for (std::size_t tile = 0; tile < hw::kMxmSupercellsPerPlane; ++tile) {
            if (!control_.compute_active(tile)) {
                continue;
            }
            const auto pulse = control_.compute_pulse(tile).value();
            check_weight_buffer(pulse.weight_buffer);
            check_accumulator_bank(pulse.accumulator_bank);
            current_compute_active_by_accumulator_tile[pulse.accumulator_bank][tile] = true;

            const auto begins_at_tile = pulse.start_of_k_block
                || !compute_active_by_accumulator_tile_[pulse.accumulator_bank][tile];
            if (begins_at_tile) {
                // The block boundary travels with the instruction.  Reset only
                // the cursor of the tile currently seeing that boundary; later
                // tiles may still be processing the preceding K block.
                next_row_for_tile_[pulse.accumulator_bank][tile] = 0;
            }
            if (tile == 0 && begins_at_tile) {
                begin_k_block(pulse.accumulator_bank, pulse.accumulate);
            }
            const auto data = collect_activation(fabric, tile, pulse.activation_stream_base);
            const auto row = next_row_for_tile_[pulse.accumulator_bank][tile]++;
            if (row >= hw::kMxmRows) {
                throw std::overflow_error("MXM K block contains more than 320 result rows");
            }
            east_pipeline_[0][tile].push_back(ActivationEvent {
                tile,
                row,
                pulse.weight_buffer,
                pulse.accumulator_bank,
                pulse.stream_base,
                pulse.accumulate,
                pulse.reduce,
                data,
            });
            if (os != nullptr && (!log_tile.has_value() || *log_tile == tile)) {
                *os << "  MXM" << mxm_id << " consume activation tile=" << tile
                    << " row=" << row
                    << " weight_buffer=" << pulse.weight_buffer
                    << " accumulator=" << pulse.accumulator_bank
                    << " stream=" << pulse.activation_stream_base
                    << " out=" << pulse.stream_base
                    << (pulse.reduce ? " reduce" : " retain") << '\n';
            }
        }

        std::array<std::array<std::deque<ActivationEvent>, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane>
            next_pipeline {};
        std::array<std::array<std::deque<PartialSumEvent>, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane>
            next_north_pipeline {};
        std::array<std::array<bool, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane> computing {};
        for (std::size_t column_block = 0; column_block < hw::kMxmSupercellsPerPlane; ++column_block) {
            for (std::size_t tile = 0; tile < hw::kMxmSupercellsPerPlane; ++tile) {
                for (const auto& event : east_pipeline_[column_block][tile]) {
                    computing[tile][column_block] = true;
                    const auto north = compute_supercell(event, column_block);
                    if (tile + 1 == hw::kMxmSupercellsPerPlane) {
                        commit_north_output(fabric, column_block, north);
                    } else {
                        next_north_pipeline[tile + 1][column_block].push_back(north);
                    }
                    if (column_block + 1 < hw::kMxmSupercellsPerPlane) {
                        next_pipeline[column_block + 1][tile].push_back(event);
                    }
                }
            }
        }
        east_pipeline_ = std::move(next_pipeline);
        north_pipeline_ = std::move(next_north_pipeline);
        last_computing_ = computing;

        if (active_ && pipelines_empty()) {
            active_ = false;
        }
        compute_active_by_accumulator_tile_ = current_compute_active_by_accumulator_tile;
    }

    bool computing_cell(std::size_t tile, std::size_t column_block) const
    {
        check_tile(tile);
        if (column_block >= hw::kMxmSupercellsPerPlane) {
            throw std::out_of_range("MXM column block is outside the 20-column array");
        }
        return last_computing_[tile][column_block];
    }

    const std::vector<ColumnOutput>& last_outputs() const
    {
        return last_outputs_;
    }

    std::int32_t accumulator_value(
        std::size_t accumulator_bank,
        std::size_t row,
        std::size_t column) const
    {
        check_accumulator_bank(accumulator_bank);
        if (row >= hw::kMxmRows || column >= hw::kMxmColumns) {
            throw std::out_of_range("MXM accumulator coordinate is outside the 320x320 bank");
        }
        return accumulator_banks_[accumulator_bank][row][column];
    }

    void clear_accumulator(std::size_t accumulator_bank)
    {
        check_accumulator_bank(accumulator_bank);
        if (pipeline_uses_accumulator(accumulator_bank)) {
            throw std::logic_error("cannot clear an MXM accumulator bank with partial sums in flight");
        }
        clear_accumulator_unchecked(accumulator_bank);
    }

private:
    struct ActivationEvent {
        std::size_t tile{0};
        std::size_t row{0};
        std::size_t weight_buffer{0};
        std::size_t accumulator_bank{0};
        std::size_t output_stream_base{0};
        bool accumulate{false};
        bool reduce{true};
        ActivationData data{};
    };

    struct PartialSumEvent {
        std::size_t row{0};
        std::size_t weight_buffer{0};
        std::size_t accumulator_bank{0};
        std::size_t output_stream_base{0};
        bool reduce{true};
        PartialSum values{};
    };

    static void check_tile(std::size_t tile)
    {
        if (tile >= hw::kMxmSupercellsPerPlane) {
            throw std::out_of_range("MXM tile is outside the 20-row array");
        }
    }

    static void check_weight_buffer(std::size_t weight_buffer)
    {
        if (weight_buffer >= kWeightBuffers) {
            throw std::out_of_range("MXM weight buffer is outside the two-buffer set");
        }
    }

    static void check_accumulator_bank(std::size_t accumulator_bank)
    {
        if (accumulator_bank >= kAccumulatorBanks) {
            throw std::out_of_range("MXM accumulator bank is outside the accumulator-bank set");
        }
    }

    void require_open_fabric(StreamRegisterFabric& fabric) const
    {
        ports_.validate_for(fabric);
        if (!fabric.cycle_open()) {
            throw std::logic_error("MXM evaluate requires an open SR cycle");
        }
    }

    MxmControlSlice::WeightInput collect_weight_input(
        StreamRegisterFabric& fabric,
        std::size_t tile) const
    {
        const auto& endpoint = ports_.weight_input();
        StreamInputPort input(fabric, endpoint.column, endpoint.direction, "MXM IW");
        for (std::size_t stream = 0; stream < hw::kMxmLoadStreamsPerCycle; ++stream) {
            if (!input.segment_valid(tile, endpoint.stream_base + stream)) {
                throw std::logic_error("MXM IW reached tile before all weight streams arrived");
            }
        }

        auto result = MxmControlSlice::WeightInput {};
        for (std::size_t stream = 0; stream < hw::kMxmLoadStreamsPerCycle; ++stream) {
            const auto segment = input.consume_segment(tile, endpoint.stream_base + stream);
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                result[lane][stream] = MxmArray::Supercell::InputWord {
                    static_cast<std::int8_t>(segment[lane].data),
                    stream + 1 == hw::kMxmLoadStreamsPerCycle,
                };
            }
        }
        return result;
    }

    void begin_k_block(std::size_t accumulator_bank, bool accumulate)
    {
        check_accumulator_bank(accumulator_bank);
        if (!accumulate) {
            if (pipeline_uses_accumulator(accumulator_bank)) {
                throw std::logic_error(
                    "cannot clear an MXM accumulator bank while an earlier K block is in flight");
            }
            clear_accumulator_unchecked(accumulator_bank);
        }
    }

    void clear_accumulator_unchecked(std::size_t accumulator_bank)
    {
        for (auto& row : accumulator_banks_[accumulator_bank]) {
            row.fill(0);
        }
    }

    ActivationData collect_activation(
        StreamRegisterFabric& fabric,
        std::size_t tile,
        std::size_t stream_base) const
    {
        if (stream_base >= hw::kStreamsPerDirection) {
            throw std::out_of_range("MXM activation stream is outside its configured direction");
        }
        const auto& endpoint = ports_.activation_input();
        StreamInputPort input(fabric, endpoint.column, endpoint.direction, "MXM Compute");
        if (!input.segment_valid(tile, stream_base)) {
            throw std::logic_error("MXM Compute reached tile before activation stream arrived");
        }
        const auto segment = endpoint.multicast
            ? input.consume_shared_segment(tile, stream_base)
            : input.consume_segment(tile, stream_base);
        ActivationData data {};
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            data[lane] = static_cast<std::int8_t>(segment[lane].data);
        }
        return data;
    }

    PartialSumEvent compute_supercell(const ActivationEvent& event, std::size_t column_block)
    {
        PartialSum south_partial {};
        if (event.tile != 0) {
            auto& incoming = north_pipeline_[event.tile][column_block];
            if (incoming.empty()) {
                throw std::logic_error(
                    "MXM activation reached a supercell before its aligned south partial sum");
            }
            const auto south = incoming.front();
            incoming.pop_front();
            if (south.row != event.row
                || south.weight_buffer != event.weight_buffer
                || south.accumulator_bank != event.accumulator_bank
                || south.output_stream_base != event.output_stream_base
                || south.reduce != event.reduce) {
                throw std::logic_error("MXM activation and south partial-sum pipelines are misaligned");
            }
            south_partial = south.values;
        }

        return PartialSumEvent {
            event.row,
            event.weight_buffer,
            event.accumulator_bank,
            event.output_stream_base,
            event.reduce,
            array_.cell(event.tile, column_block).compute_partial(
                event.data,
                event.weight_buffer,
                south_partial),
        };
    }

    void commit_north_output(
        StreamRegisterFabric& fabric,
        std::size_t column_block,
        const PartialSumEvent& partial)
    {
        const auto global_column_base = column_block * hw::kMxmSupercellColumns;
        for (std::size_t local_column = 0; local_column < hw::kMxmSupercellColumns; ++local_column) {
            accumulator_banks_[partial.accumulator_bank][partial.row][global_column_base + local_column]
                += partial.values[local_column];
        }
        if (partial.reduce) {
            emit_column_output(fabric, column_block, partial);
        }
    }

    void emit_column_output(
        StreamRegisterFabric& fabric,
        std::size_t column_block,
        const PartialSumEvent& event)
    {
        ColumnOutput output {event.row, column_block, {}, event.accumulator_bank};
        const auto global_column_base = column_block * hw::kMxmSupercellColumns;
        std::array<StreamSegment16, 4> output_segments{};
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const auto value = accumulator_banks_[event.accumulator_bank][event.row][global_column_base + lane];
            output.values[lane] = value;
            const auto raw = static_cast<std::uint32_t>(value);
            const std::array<std::uint8_t, 4> bytes {
                static_cast<std::uint8_t>(raw & 0xffu),
                static_cast<std::uint8_t>((raw >> 8) & 0xffu),
                static_cast<std::uint8_t>((raw >> 16) & 0xffu),
                static_cast<std::uint8_t>((raw >> 24) & 0xffu),
            };
            for (std::size_t byte = 0; byte < bytes.size(); ++byte) {
                output_segments[byte][lane] = StreamCell::Valid(
                    bytes[byte],
                    lane + 1 == hw::kLanesPerTile);
            }
        }
        if (event.output_stream_base + output_segments.size() > hw::kStreamsPerDirection) {
            throw std::out_of_range("MXM result stream range exceeds its configured direction");
        }
        const auto& endpoint = ports_.result_output();
        StreamOutputPort output_port(
            fabric,
            endpoint.column,
            endpoint.direction,
            "MXM result");
        for (std::size_t byte = 0; byte < output_segments.size(); ++byte) {
            output_port.write_segment(
                column_block,
                event.output_stream_base + byte,
                output_segments[byte]);
        }
        last_outputs_.push_back(output);
    }

    class NullStream {
    public:
        std::ostream& stream()
        {
            return stream_;
        }

    private:
        class Buffer : public std::streambuf {
        public:
            int overflow(int c) override
            {
                return c;
            }
        };

        Buffer buffer_{};
        std::ostream stream_{&buffer_};
    };

    bool pipelines_empty() const
    {
        for (const auto& column : east_pipeline_) {
            for (const auto& lane : column) {
                if (!lane.empty()) {
                    return false;
                }
            }
        }
        for (const auto& row : north_pipeline_) {
            for (const auto& column : row) {
                if (!column.empty()) {
                    return false;
                }
            }
        }
        return true;
    }

    bool pipeline_uses_accumulator(std::size_t accumulator_bank) const
    {
        for (const auto& column : east_pipeline_) {
            for (const auto& row : column) {
                for (const auto& event : row) {
                    if (event.accumulator_bank == accumulator_bank) {
                        return true;
                    }
                }
            }
        }
        for (const auto& row : north_pipeline_) {
            for (const auto& column : row) {
                for (const auto& event : column) {
                    if (event.accumulator_bank == accumulator_bank) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    MxmArray array_{};
    MxmControlSlice control_;
    MxmStreamPortMap ports_;
    bool active_{false};
    std::array<std::array<std::deque<ActivationEvent>, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane>
        east_pipeline_{};
    // [north destination row][column].  A token produced at row r is only
    // visible to row r+1 on the following evaluate cycle.
    std::array<std::array<std::deque<PartialSumEvent>, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane>
        north_pipeline_{};
    std::array<std::array<bool, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane> last_computing_{};
    std::array<std::array<std::size_t, hw::kMxmSupercellsPerPlane>, kAccumulatorBanks> next_row_for_tile_{};
    std::array<std::array<bool, hw::kMxmSupercellsPerPlane>, kAccumulatorBanks>
        compute_active_by_accumulator_tile_{};
    std::array<std::array<std::array<std::int32_t, hw::kMxmColumns>, hw::kMxmRows>, kAccumulatorBanks>
        accumulator_banks_{};
    std::vector<ColumnOutput> last_outputs_{};
};

} // namespace ftlpu
