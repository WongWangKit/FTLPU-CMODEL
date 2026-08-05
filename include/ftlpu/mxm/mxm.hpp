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
        std::size_t stream_base{0};
    };

    static MxmStreamPortMap AtBoundary(
        std::size_t column,
        std::size_t weight_stream_base = 0)
    {
        return MxmStreamPortMap {
            InputEndpoint {
                column,
                StreamDirection::East,
                weight_stream_base},
        };
    }

    explicit MxmStreamPortMap(InputEndpoint input)
        : input_(input)
    {
        if (input_.stream_base % hw::kMxmLoadStreamsPerCycle != 0) {
            throw std::invalid_argument(
                "MXM input window is not aligned to the configured load-stream count");
        }
        if (input_.stream_base + hw::kMxmLoadStreamsPerCycle
            > hw::kStreamsPerDirection) {
            throw std::out_of_range("MXM weight stream range exceeds one stream direction");
        }
        if (input_.stream_base
                + hw::kMxmActivationBytesPerValue
            > hw::kStreamsPerDirection) {
            throw std::out_of_range(
                "MXM fixed activation group exceeds one stream direction");
        }
    }

    const InputEndpoint& input() const noexcept
    {
        return input_;
    }

    void validate_for(const StreamRegisterFabric& fabric) const
    {
        if (input_.column >= fabric.column_count()) {
            throw std::out_of_range("MXM port maps outside stream-register fabric");
        }
    }

private:
    InputEndpoint input_{};
};

class Mxm {
public:
    static constexpr std::size_t kWeightBuffers = MxmSupercell::kWeightBuffers;
    using ActivationData = MxmSupercell::ActivationData;
    using Accumulator = MxmSupercell::Accumulator;
    using ResultValues =
        std::array<Accumulator, hw::kMxmSupercellColumns>;
    using PartialSum = MxmSupercell::PartialSum;

    struct ColumnOutput {
        std::size_t row{0};
        std::size_t column_block{0};
        ResultValues values{};
        std::size_t output_stream_base{0};
        std::size_t partial_stream_base{0};
        MxmAccumulatorMode accumulator_mode{
            MxmAccumulatorMode::DirectFinal};
        MxmPairMode pair_mode{MxmPairMode::Independent};
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
        for (auto& row : mac_pipeline_) {
            for (auto& column : row) {
                for (auto& stage : column) {
                    stage.clear();
                }
            }
        }
        for (auto& row : last_computing_) {
            row.fill(false);
        }
        for (auto& row : last_mac_token_rows_) {
            for (auto& column : row) {
                for (auto& stage : column) {
                    stage.reset();
                }
            }
        }
        next_row_for_tile_.fill(0);
        compute_active_by_tile_.fill(false);
        last_outputs_.clear();
        for (auto& row : last_mac_token_rows_) {
            for (auto& column : row) {
                for (auto& stage : column) {
                    stage.reset();
                }
            }
        }
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
        auto provider =
            [this, &fabric, mxm_id, os, log_tile](
                std::size_t tile,
                const MxmControlInstruction& instruction) {
            if (os != nullptr && (!log_tile.has_value() || tile == *log_tile)) {
                *os << "  SR -> MXM" << mxm_id << " weights tile " << tile << '\n';
            }
            return collect_weight_input(
                fabric, tile, instruction);
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

        auto current_compute_active_by_tile =
            std::array<bool, hw::kMxmSupercellsPerPlane> {};
        for (std::size_t tile = 0; tile < hw::kMxmSupercellsPerPlane; ++tile) {
            if (!control_.compute_active(tile)) {
                continue;
            }
            const auto pulse = control_.compute_pulse(tile).value();
            check_weight_buffer(pulse.weight_buffer);
            current_compute_active_by_tile[tile] = true;

            const auto begins_at_tile = pulse.start_of_k_block
                || !compute_active_by_tile_[tile];
            if (begins_at_tile) {
                // The block boundary travels with the instruction.  Reset only
                // the cursor of the tile currently seeing that boundary; later
                // tiles may still be processing the preceding K block.
                next_row_for_tile_[tile] = 0;
            }
            const auto data = collect_activation(fabric, tile);
            const auto row = next_row_for_tile_[tile]++;
            if (row >= hw::kMxmRows) {
                throw std::overflow_error(
                    "MXM K block exceeds the configured result-row count");
            }
            east_pipeline_[0][tile].push_back(ActivationEvent {
                tile,
                row,
                pulse.weight_buffer,
                pulse.stream_base,
                pulse.partial_stream_base,
                pulse.accumulator_mode,
                pulse.pair_mode,
                data,
            });
            if (os != nullptr && (!log_tile.has_value() || *log_tile == tile)) {
                *os << "  MXM" << mxm_id << " consume activation tile=" << tile
                    << " row=" << row
                    << " weight_buffer=" << pulse.weight_buffer
                    << " stream=" << ports_.input().stream_base
                    << " out=" << pulse.stream_base
                    << " acc_mode=" << static_cast<int>(pulse.accumulator_mode)
                    << " pair=" << static_cast<int>(pulse.pair_mode) << '\n';
            }
        }

        std::array<std::array<std::deque<ActivationEvent>, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane>
            next_pipeline {};
        std::array<std::array<std::deque<PartialSumEvent>, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane>
            next_north_pipeline {};
        decltype(mac_pipeline_) next_mac_pipeline {};
        std::array<std::array<bool, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane> computing {};
        for (std::size_t column_block = 0; column_block < hw::kMxmSupercellsPerPlane; ++column_block) {
            for (std::size_t tile = 0; tile < hw::kMxmSupercellsPerPlane; ++tile) {
                for (std::size_t stage = 1;
                     stage < MxmSupercell::kMacPipelineStages;
                     ++stage) {
                    auto& events = mac_pipeline_[tile][column_block][stage];
                    if (events.size() > 1) {
                        throw std::logic_error(
                            "MXM MAC stage accepted more than one token in a cycle");
                    }
                    for (auto event : events) {
                        computing[tile][column_block] = true;
                        last_mac_token_rows_[tile][column_block][stage] =
                            event.row;
                        array_.cell(tile, column_block).accumulate_mac_row(
                            event.data,
                            event.weight_buffer,
                            stage,
                            event.values);
                        if (stage + 1
                            < MxmSupercell::kMacPipelineStages) {
                            next_mac_pipeline[tile][column_block]
                                             [stage + 1]
                                .push_back(std::move(event));
                        } else {
                            const auto north = finish_mac_event(event);
                            if (tile + 1
                                == hw::kMxmSupercellsPerPlane) {
                                commit_north_output(
                                    column_block, north);
                            } else {
                                next_north_pipeline[tile + 1]
                                                   [column_block]
                                    .push_back(north);
                            }
                        }
                    }
                }

                auto& activation_events =
                    east_pipeline_[column_block][tile];
                if (activation_events.size() > 1) {
                    throw std::logic_error(
                        "MXM supercell input accepted more than one token in a cycle");
                }
                for (const auto& event : activation_events) {
                    computing[tile][column_block] = true;
                    last_mac_token_rows_[tile][column_block][0] =
                        event.row;
                    auto mac_event = begin_mac_event(
                        event, tile, column_block);
                    array_.cell(tile, column_block).accumulate_mac_row(
                        mac_event.data,
                        mac_event.weight_buffer,
                        0,
                        mac_event.values);
                    if (MxmSupercell::kMacPipelineStages == 1) {
                        const auto north = finish_mac_event(mac_event);
                        if (tile + 1
                            == hw::kMxmSupercellsPerPlane) {
                            commit_north_output(
                                column_block, north);
                        } else {
                            next_north_pipeline[tile + 1]
                                               [column_block]
                                .push_back(north);
                        }
                    } else {
                        next_mac_pipeline[tile][column_block][1]
                            .push_back(std::move(mac_event));
                    }
                    if (column_block + 1 < hw::kMxmSupercellsPerPlane) {
                        next_pipeline[column_block + 1][tile].push_back(event);
                    }
                }
            }
        }
        for (const auto& row : north_pipeline_) {
            for (const auto& column : row) {
                if (!column.empty()) {
                    throw std::logic_error(
                        "MXM partial sum arrived before its compiler-skewed activation");
                }
            }
        }
        east_pipeline_ = std::move(next_pipeline);
        north_pipeline_ = std::move(next_north_pipeline);
        mac_pipeline_ = std::move(next_mac_pipeline);
        last_computing_ = computing;

        if (active_ && pipelines_empty()) {
            active_ = false;
        }
        compute_active_by_tile_ = current_compute_active_by_tile;
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

    std::optional<std::size_t> mac_stage_token_row(
        std::size_t tile,
        std::size_t column_block,
        std::size_t mac_row) const
    {
        check_tile(tile);
        if (column_block >= hw::kMxmSupercellsPerPlane
            || mac_row >= MxmSupercell::kMacPipelineStages) {
            throw std::out_of_range(
                "MXM MAC array coordinate is outside the supercell");
        }
        return last_mac_token_rows_[tile][column_block][mac_row];
    }

private:
    struct ActivationEvent {
        std::size_t tile{0};
        std::size_t row{0};
        std::size_t weight_buffer{0};
        std::size_t output_stream_base{0};
        std::size_t partial_stream_base{0};
        MxmAccumulatorMode accumulator_mode{
            MxmAccumulatorMode::DirectFinal};
        MxmPairMode pair_mode{MxmPairMode::Independent};
        ActivationData data{};
    };

    struct PartialSumEvent {
        std::size_t row{0};
        std::size_t weight_buffer{0};
        std::size_t output_stream_base{0};
        std::size_t partial_stream_base{0};
        MxmAccumulatorMode accumulator_mode{
            MxmAccumulatorMode::DirectFinal};
        MxmPairMode pair_mode{MxmPairMode::Independent};
        PartialSum values{};
    };

    struct MacPipelineEvent {
        std::size_t row{0};
        std::size_t weight_buffer{0};
        std::size_t output_stream_base{0};
        std::size_t partial_stream_base{0};
        MxmAccumulatorMode accumulator_mode{
            MxmAccumulatorMode::DirectFinal};
        MxmPairMode pair_mode{MxmPairMode::Independent};
        ActivationData data{};
        PartialSum values{};
    };

    static void check_tile(std::size_t tile)
    {
        if (tile >= hw::kMxmSupercellsPerPlane) {
            throw std::out_of_range(
                "MXM tile exceeds the configured Supercell-plane rows");
        }
    }

    static void check_weight_buffer(std::size_t weight_buffer)
    {
        if (weight_buffer >= kWeightBuffers) {
            throw std::out_of_range("MXM weight buffer is outside the two-buffer set");
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
        std::size_t tile,
        const MxmControlInstruction& instruction) const
    {
        const auto& endpoint = ports_.input();
        StreamInputPort input(fabric, endpoint.column, endpoint.direction, "MXM IW");
        const auto stream_count =
            instruction.opcode
                    == MxmControlOpcode::LoadScales
            ? hw::kMxmWeightScaleStreams
            : instruction.weight_load_mode
                    == MxmWeightLoadMode::Full
                ? hw::kMxmStoredWeightLoadStreams
                : hw::kMxmBackgroundWeightLoadStreams;
        if (stream_count == 0) {
            throw std::logic_error(
                "MXM LoadScales is not supported by this hardware configuration");
        }
        const auto source_offset =
            instruction.opcode == MxmControlOpcode::IW
                && instruction.weight_load_mode
                    != MxmWeightLoadMode::Full
            ? hw::kMxmLoadStreamsPerCycle - stream_count
            : std::size_t {0};
        const auto destination_offset =
            instruction.opcode == MxmControlOpcode::IW
                && instruction.weight_load_mode
                    == MxmWeightLoadMode::BackgroundUpperHalf
            ? hw::kMxmStoredWeightLoadStreams - stream_count
            : std::size_t {0};
        for (std::size_t stream = 0;
             stream < stream_count;
             ++stream) {
            if (!input.segment_valid(
                    tile,
                    endpoint.stream_base + source_offset + stream)) {
                throw std::logic_error(
                    "MXM weight instruction reached tile before all input streams arrived");
            }
        }

        auto result = MxmControlSlice::WeightInput {};
        for (std::size_t stream = 0;
             stream < stream_count;
             ++stream) {
            const auto segment = input.consume_segment(
                tile,
                endpoint.stream_base + source_offset + stream);
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                result[lane][destination_offset + stream] =
                    MxmArray::Supercell::InputWord {
                    static_cast<MxmArray::Supercell::EncodedWeightByte>(
                        segment[lane].data),
                    stream + 1 == stream_count,
                };
            }
        }
        return result;
    }

    ActivationData collect_activation(
        StreamRegisterFabric& fabric,
        std::size_t tile) const
    {
        const auto stream_base = ports_.input().stream_base;
        if (stream_base + hw::kMxmActivationBytesPerValue
            > hw::kStreamsPerDirection) {
            throw std::out_of_range("MXM activation stream is outside its configured direction");
        }
        const auto& endpoint = ports_.input();
        StreamInputPort input(fabric, endpoint.column, endpoint.direction, "MXM Compute");
        for (std::size_t byte = 0;
             byte < hw::kMxmActivationBytesPerValue;
             ++byte) {
            if (!input.segment_valid(tile, stream_base + byte)) {
                throw std::logic_error(
                    "MXM Compute reached tile before all activation bytes arrived");
            }
        }
        ActivationData data {};
        if constexpr (MxmSupercell::kFp16Datapath) {
            const auto low =
                input.consume_segment(tile, stream_base);
            const auto high =
                input.consume_segment(tile, stream_base + 1);
            for (std::size_t element = 0;
                 element < hw::kLanesPerTile;
                 ++element) {
                const auto bits = static_cast<std::uint16_t>(
                    low[element].data)
                    | (static_cast<std::uint16_t>(
                           high[element].data)
                       << 8);
                data[element] = Fp16::from_bits(bits).to_float();
            }
        } else {
            const auto segment =
                input.consume_segment(tile, stream_base);
            for (std::size_t element = 0;
                 element < hw::kLanesPerTile;
                 ++element) {
                data[element] =
                    static_cast<std::int8_t>(segment[element].data);
            }
        }
        return data;
    }

    MacPipelineEvent begin_mac_event(
        const ActivationEvent& event,
        std::size_t tile,
        std::size_t column_block)
    {
        PartialSum south_partial {};
        if (tile != 0) {
            auto& incoming = north_pipeline_[tile][column_block];
            if (incoming.empty()) {
                throw std::logic_error(
                    "MXM activation reached a MAC array before its aligned south partial sum");
            }
            const auto south = incoming.front();
            incoming.pop_front();
            if (south.row != event.row
                || south.weight_buffer != event.weight_buffer
                || south.output_stream_base != event.output_stream_base
                || south.partial_stream_base != event.partial_stream_base
                || south.accumulator_mode != event.accumulator_mode
                || south.pair_mode != event.pair_mode) {
                throw std::logic_error("MXM activation and south partial-sum pipelines are misaligned");
            }
            south_partial = south.values;
        }

        return MacPipelineEvent {
            event.row,
            event.weight_buffer,
            event.output_stream_base,
            event.partial_stream_base,
            event.accumulator_mode,
            event.pair_mode,
            event.data,
            south_partial,
        };
    }

    static PartialSumEvent finish_mac_event(
        const MacPipelineEvent& event)
    {
        return PartialSumEvent {
            event.row,
            event.weight_buffer,
            event.output_stream_base,
            event.partial_stream_base,
            event.accumulator_mode,
            event.pair_mode,
            event.values,
        };
    }

    void commit_north_output(
        std::size_t column_block,
        const PartialSumEvent& partial)
    {
        auto output = ColumnOutput {
            partial.row,
            column_block,
            {},
            partial.output_stream_base,
            partial.partial_stream_base,
            partial.accumulator_mode,
            partial.pair_mode};
        for (std::size_t lane = 0;
             lane < hw::kLanesPerTile;
             ++lane) {
            output.values[lane] = partial.values[lane];
        }
        last_outputs_.push_back(std::move(output));
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
        for (const auto& row : mac_pipeline_) {
            for (const auto& column : row) {
                for (const auto& stage : column) {
                    if (!stage.empty()) {
                        return false;
                    }
                }
            }
        }
        return true;
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
    // [supercell row][supercell column][next MAC row]. Each stage holds at
    // most one token and advances every cycle, giving latency K and II=1.
    std::array<
        std::array<
            std::array<
                std::deque<MacPipelineEvent>,
                MxmSupercell::kMacPipelineStages>,
            hw::kMxmSupercellsPerPlane>,
        hw::kMxmSupercellsPerPlane> mac_pipeline_{};
    std::array<std::array<bool, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane> last_computing_{};
    std::array<
        std::array<
            std::array<
                std::optional<std::size_t>,
                MxmSupercell::kMacPipelineStages>,
            hw::kMxmSupercellsPerPlane>,
        hw::kMxmSupercellsPerPlane> last_mac_token_rows_{};
    std::array<std::size_t, hw::kMxmSupercellsPerPlane> next_row_for_tile_{};
    std::array<bool, hw::kMxmSupercellsPerPlane> compute_active_by_tile_{};
    std::vector<ColumnOutput> last_outputs_{};
};

} // namespace ftlpu
