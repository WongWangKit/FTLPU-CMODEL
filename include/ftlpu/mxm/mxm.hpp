#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/bf16.hpp"
#include "ftlpu/core/fp16.hpp"
#include "ftlpu/mem/tile_array.hpp"
#include "ftlpu/mxm/accumulator.hpp"
#include "ftlpu/mxm/array.hpp"
#include "ftlpu/mxm/block_accumulator.hpp"
#include "ftlpu/mxm/control_slice.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ftlpu {

class Mxm {
public:
    static constexpr std::size_t kWeightBuffers = MxmSupercell::kWeightBuffers;
    using ActivationData = std::array<float, hw::kLanesPerTile>;
    using ActivationBlock =
        std::array<ActivationData, hw::kMxmBlockRows>;
    using ResultValues = std::array<float, hw::kMxmSupercellColumns>;

    struct ColumnOutput {
        std::size_t row{0};
        std::size_t column_block{0};
        ResultValues values{};
    };

    Mxm()
        : control_(array_)
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

    MxmAccumulator& accumulator() noexcept
    {
        return accumulator_;
    }

    const MxmAccumulator& accumulator() const noexcept
    {
        return accumulator_;
    }

    MxmBlockAccumulator& block_accumulator() noexcept
    {
        return block_accumulator_;
    }

    const MxmBlockAccumulator& block_accumulator() const noexcept
    {
        return block_accumulator_;
    }

    void reset_datapath()
    {
        for (auto& column : east_pipeline_) {
            for (auto& lane : column) {
                lane.clear();
            }
        }
        for (auto& buffer : accumulators_) {
            for (auto& row : buffer) {
                row.fill(0);
            }
        }
        for (auto& buffer : contribution_counts_) {
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
        accumulator_.clear();
        block_accumulator_.clear();
        for (auto& buffer : decode_activation_buffers_) {
            for (auto& tile : buffer) {
                for (auto& cell : tile) {
                    cell.fill(0.0f);
                }
            }
        }
        for (auto& buffer : decode_activation_valid_) {
            for (auto& tile : buffer) {
                for (auto& cell : tile) {
                    cell.fill(false);
                }
            }
        }
        decode_launch_.reset();
        for (auto& stage : decode_stages_) {
            stage.reset();
        }
        for (auto& stage : decode_output_pipeline_) {
            stage.reset();
        }
        decode_streaming_active_ = false;
        decode_layout_ = MxmDecodeLayout::Linear1x16;
        compute_active_by_buffer_.fill(false);
        last_outputs_.clear();
        active_ = false;
    }

    void tick_datapath(
        TileArrayModel& mem,
        std::size_t mxm_id,
        std::ostream* os = nullptr,
        std::optional<std::size_t> log_tile = std::nullopt)
    {
        last_outputs_.clear();

        for (std::size_t tile = 0;
             tile < hw::kMxmSupercellsPerPlane;
             ++tile) {
            if (const auto load =
                    control_.decode_activation_load_pulse(tile);
                load.has_value()) {
                load_decode_activation(mem, tile, *load, mxm_id, os);
            }
            if (const auto compute =
                    control_.decode_stream_compute_pulse(tile);
                compute.has_value()) {
                if (tile == 0) {
                    launch_decode_wave(*compute, mxm_id, os);
                }
            }
        }
        advance_decode_pipeline(mem, mxm_id, os);

        if (!active_ && control_.compute_active(0)) {
            active_ = true;
        }

        auto current_compute_active_by_buffer = std::array<bool, kWeightBuffers> {};
        for (std::size_t tile = 0; tile < hw::kMxmSupercellsPerPlane; ++tile) {
            if (!control_.compute_active(tile)) {
                continue;
            }
            const auto weight_buffer = control_.compute_weight_buffer(tile).value_or(0);
            const auto stream_base = control_.compute_activation_stream_base(tile).value_or(0);
            const auto output_stream_base = control_.output_stream_base(tile).value_or(0);
            const auto compute = control_.compute_pulse(tile).value();
            check_weight_buffer(weight_buffer);
            current_compute_active_by_buffer[weight_buffer] = true;
            const auto row_count = compute_row_count(compute.compute_mode);
            if (tile == 0
                && compute.compute_mode != MxmComputeMode::Block8
                && !compute_active_by_buffer_[weight_buffer]) {
                reset_buffer_state(weight_buffer);
            }
            const auto absolute_row = next_row_for_tile_[weight_buffer][tile];
            next_row_for_tile_[weight_buffer][tile] += row_count;
            const auto row = absolute_row % hw::kMxmRows;
            if (tile == 0 && absolute_row >= hw::kMxmRows) {
                for (std::size_t offset = 0; offset < row_count; ++offset) {
                    reset_accumulator_row(
                        weight_buffer, (row + offset) % hw::kMxmRows);
                }
            }
            const auto data = collect_activation(
                mem,
                tile,
                stream_base,
                compute.data_format,
                compute.compute_mode);
            east_pipeline_[0][tile].push_back(ActivationEvent {
                tile,
                row,
                weight_buffer,
                output_stream_base,
                compute.accumulator_address,
                compute.accumulator_row_stride,
                compute.accumulator_destination,
                compute.accumulator_clear,
                compute.data_format,
                compute.compute_mode,
                compute.accumulator_output_format,
                data});
            if (os != nullptr && (!log_tile.has_value() || *log_tile == tile)) {
                *os << "  MXM" << mxm_id << " consume activation tile=" << tile
                    << " row=" << row
                    << " buffer=" << weight_buffer
                    << " stream=" << stream_base
                    << " format="
                    << mxm_data_format_name(compute.data_format)
                    << " mode="
                    << (compute.compute_mode == MxmComputeMode::Block8
                            ? "block8"
                            : "vector")
                    << " out=" << output_stream_base << '\n';
            }
        }

        std::array<std::array<std::deque<ActivationEvent>, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane>
            next_pipeline {};
        std::array<std::array<bool, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane> computing {};
        for (std::size_t column_block = 0; column_block < hw::kMxmSupercellsPerPlane; ++column_block) {
            for (std::size_t tile = 0; tile < hw::kMxmSupercellsPerPlane; ++tile) {
                for (const auto& event : east_pipeline_[column_block][tile]) {
                    computing[tile][column_block] = true;
                    compute_column_block(event, column_block);
                    if (event.tile + 1 == hw::kMxmSupercellsPerPlane) {
                        emit_column_output(mem, column_block, event);
                    }
                    if (column_block + 1 < hw::kMxmSupercellsPerPlane) {
                        next_pipeline[column_block + 1][tile].push_back(event);
                    }
                }
            }
        }
        east_pipeline_ = std::move(next_pipeline);
        last_computing_ = computing;

        for (std::size_t tile = 0; tile < hw::kMxmSupercellsPerPlane; ++tile) {
            const auto read = control_.accumulator_read_pulse(tile);
            if (!read.has_value()) continue;
            if (read->compute_mode == MxmComputeMode::Block8) {
                const auto values =
                    block_accumulator_.read(read->address, tile);
                emit_block_stream_values(
                    mem,
                    tile,
                    read->stream_base,
                    read->address,
                    values);
                if (read->clear) {
                    block_accumulator_.clear_segment(
                        read->address,
                        tile);
                }
                continue;
            }
            const auto values = accumulator_.read(read->address, tile);
            emit_stream_values(
                mem, tile, read->stream_base, read->address, values);
            if (read->clear) {
                accumulator_.clear_segment(read->address, tile);
            }
        }

        if (active_ && pipelines_empty()) {
            active_ = false;
        }
        compute_active_by_buffer_ = current_compute_active_by_buffer;
    }

    bool computing_cell(std::size_t tile, std::size_t column_block) const
    {
        check_tile(tile);
        if (column_block >= hw::kMxmSupercellsPerPlane) {
            throw std::out_of_range("MXM column block is outside the configured array");
        }
        return last_computing_[tile][column_block];
    }

    const std::vector<ColumnOutput>& last_outputs() const
    {
        return last_outputs_;
    }

private:
    static constexpr std::size_t kLinearDecodeStages =
        hw::kTileRows * hw::kMxmSupercellsPerPlane;
    static constexpr std::size_t kNativeDecodeStages =
        hw::kTileRows + hw::kMxmSupercellsPerPlane - 1;
    static constexpr std::size_t kDecodePipelineStages =
        kLinearDecodeStages;

    using DecodeWeightCell = std::array<
        std::array<float, hw::kMxmSupercellColumns>,
        hw::kLanesPerTile>;
    struct DecodeWaveState {
        std::uint64_t wave_id{0};
        std::size_t activation_buffer{0};
        std::size_t output_stream_base{0};
        MxmDequantInstruction dequant{};
        std::size_t accumulator_address{0};
        std::size_t accumulator_column{0};
        MxmAccumulatorDestination accumulator_destination{
            MxmAccumulatorDestination::Stream};
        bool accumulator_clear{true};
        MxmDecodeLayout layout{MxmDecodeLayout::Linear1x16};
        std::array<ResultValues, hw::kMxmSupercellsPerPlane> partial{};
    };

    struct DecodeCompletedOutput {
        std::uint64_t wave_id{0};
        std::size_t output_stream_base{0};
        std::array<ResultValues, hw::kMxmSupercellsPerPlane> values{};
    };

    struct ActivationEvent {
        std::size_t tile{0};
        std::size_t row{0};
        std::size_t weight_buffer{0};
        std::size_t output_stream_base{0};
        std::size_t accumulator_address{0};
        std::size_t accumulator_row_stride{1};
        MxmAccumulatorDestination accumulator_destination{
            MxmAccumulatorDestination::Stream};
        bool accumulator_clear{true};
        MxmDataFormat data_format{MxmDataFormat::Float16};
        MxmComputeMode compute_mode{MxmComputeMode::Vector};
        MxmAccumulatorOutputFormat accumulator_output_format{
            MxmAccumulatorOutputFormat::Float32};
        ActivationBlock data{};
    };

    static void check_tile(std::size_t tile)
    {
        if (tile >= hw::kMxmSupercellsPerPlane) {
            throw std::out_of_range("MXM tile is outside the configured array");
        }
    }

    static void check_weight_buffer(std::size_t weight_buffer)
    {
        if (weight_buffer >= kWeightBuffers) {
            throw std::out_of_range("MXM weight buffer is outside the two-buffer set");
        }
    }

    void reset_buffer_state(std::size_t weight_buffer)
    {
        check_weight_buffer(weight_buffer);
        next_row_for_tile_[weight_buffer].fill(0);
        for (auto& row : accumulators_[weight_buffer]) {
            row.fill(0.0f);
        }
        for (auto& row : contribution_counts_[weight_buffer]) {
            row.fill(0.0f);
        }
    }

    void reset_accumulator_row(std::size_t weight_buffer, std::size_t row)
    {
        check_weight_buffer(weight_buffer);
        if (row >= hw::kMxmRows) {
            throw std::out_of_range("MXM accumulator row is outside the physical row set");
        }
        accumulators_[weight_buffer][row].fill(0.0f);
        contribution_counts_[weight_buffer][row].fill(0);
    }

    static std::size_t compute_row_count(MxmComputeMode mode)
    {
        return mode == MxmComputeMode::Block8
            ? hw::kMxmBlockRows
            : 1;
    }

    static ActivationBlock collect_activation(
        TileArrayModel& mem,
        std::size_t tile,
        std::size_t stream_base,
        MxmDataFormat format,
        MxmComputeMode mode)
    {
        constexpr auto kTargetSreg = hw::kMxmBoundaryStreamRegisterColumn;
        const auto row_count = compute_row_count(mode);
        const auto stream_count = row_count * hw::kMxmWeightBytesPerValue;
        if (stream_base + stream_count > hw::kEastStreams) {
            throw std::out_of_range(
                "MXM activation stream range is outside the east stream set");
        }
        ActivationBlock data {};
        for (std::size_t output_row = 0;
             output_row < row_count;
             ++output_row) {
            const auto row_stream_base =
                stream_base + output_row * hw::kMxmWeightBytesPerValue;
            for (std::size_t lane = 0;
                 lane < hw::kLanesPerTile;
                 ++lane) {
                const auto low = mem.consume_east_register(
                    tile, lane, kTargetSreg, row_stream_base);
                const auto high = mem.consume_east_register(
                    tile, lane, kTargetSreg, row_stream_base + 1);
                if (!low.has_value() || !high.has_value()) {
                    throw std::logic_error(
                        "MXM Compute reached tile "
                        + std::to_string(tile)
                        + " lane " + std::to_string(lane)
                        + " before activation streams "
                        + std::to_string(row_stream_base)
                        + "/" + std::to_string(row_stream_base + 1)
                        + " arrived (low="
                        + (low.has_value() ? "valid" : "missing")
                        + ", high="
                        + (high.has_value() ? "valid" : "missing")
                        + ")");
                }
                const auto bits = static_cast<std::uint16_t>(low->data)
                    | (static_cast<std::uint16_t>(high->data) << 8);
                data[output_row][lane] = decode_mxm_16bit(bits, format);
            }
        }
        return data;
    }

    void load_decode_activation(
        TileArrayModel& mem,
        std::size_t tile,
        const MxmControlSlice::DecodeActivationLoadPulse& pulse,
        std::size_t mxm_id,
        std::ostream* os)
    {
        constexpr auto kTargetSreg =
            hw::kMxmBoundaryStreamRegisterColumn;
        check_weight_buffer(pulse.activation_buffer);
        for (std::size_t column = 0;
             column < hw::kMxmSupercellsPerPlane;
             ++column) {
            const auto stream_base = pulse.stream_base
                + (pulse.layout == MxmDecodeLayout::Linear1x16
                    ? column * sizeof(std::uint16_t)
                    : 0);
            for (std::size_t lane = 0;
                 lane < hw::kLanesPerTile;
                 ++lane) {
                const auto low = mem.consume_east_register(
                    tile, lane, kTargetSreg, stream_base);
                const auto high = mem.consume_east_register(
                    tile, lane, kTargetSreg, stream_base + 1);
                if (!low.has_value() || !high.has_value()) {
                    throw std::logic_error(
                        "MXM decode activation load reached tile "
                        + std::to_string(tile)
                        + " before its activation vector arrived");
                }
                const auto bits = static_cast<std::uint16_t>(
                    static_cast<std::uint16_t>(low->data)
                    | (static_cast<std::uint16_t>(high->data) << 8));
                decode_activation_buffers_[pulse.activation_buffer]
                                          [tile][column][lane] =
                    decode_mxm_16bit(bits, pulse.data_format);
                decode_activation_valid_[pulse.activation_buffer]
                                        [tile][column][lane] = true;
            }
        }
        if (os != nullptr) {
            *os << "  MXM" << mxm_id
                << " decode activation load tile=" << tile
                << " buffer=" << pulse.activation_buffer
                << (pulse.layout == MxmDecodeLayout::Native4x4
                        ? " broadcast_cells=0.."
                        : " cells=0..")
                << hw::kMxmSupercellsPerPlane - 1
                << " elements=" << hw::kLanesPerTile
                << '\n';
        }
    }

    void launch_decode_wave(
        const MxmControlSlice::DecodeStreamComputePulse& pulse,
        std::size_t mxm_id,
        std::ostream* os)
    {
        check_weight_buffer(pulse.activation_buffer);
        if (decode_launch_.has_value()) {
            throw std::logic_error(
                "MXM decode launched more than one wave in a cycle");
        }
        if (decode_streaming_active_ && pulse.layout != decode_layout_) {
            throw std::logic_error(
                "MXM decode layout cannot change while waves are in flight");
        }
        decode_launch_ = DecodeWaveState {
            pulse.wave_id,
            pulse.activation_buffer,
            pulse.output_stream_base,
            pulse.dequant,
            pulse.accumulator_address,
            pulse.accumulator_column,
            pulse.accumulator_destination,
            pulse.accumulator_clear,
            pulse.layout,
            {}};
        decode_streaming_active_ = true;
        decode_layout_ = pulse.layout;
        if (os != nullptr) {
            *os << "  MXM" << mxm_id
                << " decode launch wave=" << pulse.wave_id
                << " layout="
                << (pulse.layout == MxmDecodeLayout::Native4x4
                        ? "4x4"
                        : "1x16")
                << " scale=" << pulse.dequant.scale()
                << '\n';
        }
    }

    std::optional<DecodeWeightCell> consume_decode_weight_cell(
        TileArrayModel& mem,
        std::size_t tile,
        std::size_t column,
        std::optional<MxmDequantInstruction> dequant)
    {
        constexpr auto kTargetSreg =
            hw::kMxmBoundaryStreamRegisterColumn;
        const auto stream_base =
            column * hw::kMxmInt8LoadStreamsPerCycle;
        bool any = false;
        bool all = true;
        for (std::size_t lane = 0;
             lane < hw::kLanesPerTile;
             ++lane) {
            for (std::size_t output_lane = 0;
                 output_lane < hw::kMxmSupercellColumns;
                 ++output_lane) {
                const auto& word = mem.east_register(
                    tile,
                    lane,
                    kTargetSreg,
                    stream_base + output_lane);
                any = any || word.has_value();
                all = all && word.has_value();
            }
        }
        if (!any) {
            return std::nullopt;
        }
        if (!all) {
            auto missing = std::string {};
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                for (std::size_t output_lane = 0;
                     output_lane < hw::kMxmSupercellColumns;
                     ++output_lane) {
                    if (!mem.east_register(
                            tile,
                            lane,
                            kTargetSreg,
                            stream_base + output_lane).has_value()) {
                        missing = " lane=" + std::to_string(lane)
                            + " stream="
                            + std::to_string(stream_base + output_lane);
                        break;
                    }
                }
                if (!missing.empty()) {
                    break;
                }
            }
            throw std::logic_error(
                "MXM decode diamond weight block is only partially valid"
                " tile=" + std::to_string(tile)
                + " column=" + std::to_string(column) + missing);
        }

        auto quantized = MxmWeightInput {};
        for (std::size_t lane = 0;
             lane < hw::kLanesPerTile;
             ++lane) {
            for (std::size_t output_lane = 0;
                 output_lane < hw::kMxmSupercellColumns;
                 ++output_lane) {
                const auto word = mem.consume_east_register(
                    tile,
                    lane,
                    kTargetSreg,
                    stream_base + output_lane);
                quantized[lane][output_lane] = MxmSupercell::InputWord {
                    word->data,
                    output_lane + 1 == hw::kMxmSupercellColumns};
            }
        }
        if (!dequant.has_value()) {
            return DecodeWeightCell {};
        }

        const auto converted = decode_dequantizer_.convert(
            quantized,
            MxmWeightLoadMode::Supercell,
            0,
            MxmWeightInputMode::Int8DequantBf16,
            *dequant);
        auto weights = DecodeWeightCell {};
        for (std::size_t lane = 0;
             lane < hw::kLanesPerTile;
             ++lane) {
            for (std::size_t output_lane = 0;
                 output_lane < hw::kMxmSupercellColumns;
                 ++output_lane) {
                const auto& weight = converted[lane][output_lane];
                if (!weight.has_value()) {
                    throw std::logic_error(
                        "MXM decode dequantizer produced an incomplete diamond weight block");
                }
                weights[lane][output_lane] =
                    Bf16::from_bits(weight->data).to_float();
            }
        }
        return weights;
    }

    void execute_decode_stage(
        TileArrayModel& mem,
        DecodeWaveState& state,
        std::size_t stage,
        std::ostream* os = nullptr)
    {
        const auto execute_cell = [&](std::size_t tile,
                                      std::size_t column,
                                      std::size_t partial_column) {
            const auto weights = consume_decode_weight_cell(
                mem, tile, column, state.dequant);
            if (!weights.has_value()) {
                throw std::logic_error(
                    "MXM decode partial sum reached a row before its streamed weight arrived");
            }
            for (std::size_t lane = 0;
                 lane < hw::kLanesPerTile;
                 ++lane) {
                if (!decode_activation_valid_[state.activation_buffer]
                                             [tile][column][lane]) {
                    throw std::logic_error(
                        "MXM decode stage used an unloaded broadcast activation vector");
                }
            }
            for (std::size_t output_lane = 0;
                 output_lane < hw::kMxmSupercellColumns;
                 ++output_lane) {
                auto partial = 0.0f;
                for (std::size_t lane = 0;
                     lane < hw::kLanesPerTile;
                     ++lane) {
                    partial +=
                        decode_activation_buffers_[state.activation_buffer]
                                                  [tile][column][lane]
                        * (*weights)[lane][output_lane];
                }
                state.partial[partial_column][output_lane] += partial;
            }
            last_computing_[tile][column] = true;
            if (os != nullptr) {
                *os << "  decode wave=" << state.wave_id
                    << " stage=" << stage
                    << " cell=(" << tile << ',' << column << ')'
                    << " partial0=" << state.partial[partial_column][0]
                    << '\n';
            }
        };

        if (state.layout == MxmDecodeLayout::Linear1x16) {
            execute_cell(
                stage % hw::kTileRows,
                stage / hw::kTileRows,
                0);
            return;
        }

        for (std::size_t column = 0;
             column < hw::kMxmSupercellsPerPlane;
             ++column) {
            if (stage >= column && stage - column < hw::kTileRows) {
                execute_cell(stage - column, column, column);
            }
        }
    }

    void advance_decode_pipeline(
        TileArrayModel& mem,
        std::size_t mxm_id,
        std::ostream* os)
    {
        auto next = std::array<
            std::optional<DecodeWaveState>,
            kDecodePipelineStages> {};
        auto completed = std::optional<DecodeCompletedOutput> {};

        if (decode_launch_.has_value()) {
            auto state = *decode_launch_;
            execute_decode_stage(mem, state, 0, os);
            next[0] = std::move(state);
        }
        decode_launch_.reset();

        for (std::size_t stage = 1;
             stage < kDecodePipelineStages;
             ++stage) {
            if (!decode_stages_[stage - 1].has_value()) {
                continue;
            }
            auto state = *decode_stages_[stage - 1];
            execute_decode_stage(mem, state, stage, os);
            const auto stage_count = state.layout
                    == MxmDecodeLayout::Native4x4
                ? kNativeDecodeStages
                : kLinearDecodeStages;
            if (stage + 1 == stage_count) {
                completed = DecodeCompletedOutput {
                    state.wave_id,
                    state.output_stream_base,
                    {}};
                const auto segment_count = state.layout
                        == MxmDecodeLayout::Native4x4
                    ? hw::kMxmSupercellsPerPlane
                    : std::size_t {1};
                for (std::size_t column = 0;
                     column < segment_count;
                     ++column) {
                    const auto linear_segment =
                        state.accumulator_column + column;
                    const auto address = state.accumulator_address
                        + linear_segment / hw::kMxmSupercellsPerPlane;
                    const auto segment =
                        linear_segment % hw::kMxmSupercellsPerPlane;
                    accumulator_.accumulate(
                        address, segment, state.partial[column]);
                }
                if (state.accumulator_destination
                    == MxmAccumulatorDestination::Stream) {
                    for (std::size_t column = 0;
                         column < segment_count;
                         ++column) {
                        const auto linear_segment =
                            state.accumulator_column + column;
                        const auto address = state.accumulator_address
                            + linear_segment / hw::kMxmSupercellsPerPlane;
                        const auto segment =
                            linear_segment % hw::kMxmSupercellsPerPlane;
                        const auto output_tile = state.layout
                                == MxmDecodeLayout::Native4x4
                            ? column
                            : hw::kTileRows - 1;
                        completed->values[output_tile] =
                            accumulator_.read(address, segment);
                        if (state.accumulator_clear) {
                            accumulator_.clear_segment(address, segment);
                        }
                    }
                } else {
                    completed.reset();
                }
                if (os != nullptr) {
                    *os << "  MXM" << mxm_id
                        << " decode complete wave=" << state.wave_id
                        << " after " << stage_count
                        << " stages\n";
                }
            } else {
                next[stage] = std::move(state);
            }
        }

        decode_stages_ = std::move(next);
        decode_streaming_active_ = std::any_of(
            decode_stages_.begin(),
            decode_stages_.end(),
            [](const auto& stage) { return stage.has_value(); });
        if (!decode_streaming_active_) {
            decode_layout_ = MxmDecodeLayout::Linear1x16;
        }
        advance_decode_output_pipeline(mem, std::move(completed));
    }

    void advance_decode_output_pipeline(
        TileArrayModel& mem,
        std::optional<DecodeCompletedOutput> completed)
    {
        auto next = std::array<
            std::optional<DecodeCompletedOutput>,
            hw::kTileRows> {};
        for (std::size_t tile = hw::kTileRows - 1; tile > 0; --tile) {
            next[tile] = std::move(decode_output_pipeline_[tile - 1]);
        }
        next[0] = std::move(completed);
        decode_output_pipeline_ = std::move(next);

        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            if (!decode_output_pipeline_[tile].has_value()) {
                continue;
            }
            const auto& output = *decode_output_pipeline_[tile];
            for (std::size_t lane = 0;
                 lane < hw::kMxmSupercellColumns;
                 ++lane) {
                const auto value = output.values[tile][lane];
                const auto raw = Bf16::from_float(value).bits();
                for (std::size_t byte = 0;
                     byte < sizeof(std::uint16_t);
                     ++byte) {
                    mem.set_west_stream_cell(
                        tile,
                        lane,
                        output.output_stream_base + byte,
                        StreamCell::Valid(
                            static_cast<std::uint8_t>(
                                (raw >> (byte * 8)) & 0xffu),
                            lane + 1 == hw::kMxmSupercellColumns,
                            output.wave_id));
                }
            }
        }
    }

    void compute_column_block(const ActivationEvent& event, std::size_t column_block)
    {
        const auto row_count = compute_row_count(event.compute_mode);
        for (std::size_t output_row = 0;
             output_row < row_count;
             ++output_row) {
            const auto row = (event.row + output_row) % hw::kMxmRows;
            for (std::size_t local_column = 0;
                 local_column < hw::kMxmSupercellColumns;
                 ++local_column) {
                const auto global_column =
                    column_block * hw::kMxmSupercellColumns + local_column;
                for (std::size_t lane = 0;
                     lane < hw::kLanesPerTile;
                     ++lane) {
                    const auto partial = event.data[output_row][lane]
                        * static_cast<float>(
                            array_.weight(
                                event.weight_buffer,
                                event.tile,
                                column_block,
                                lane,
                                local_column,
                                event.data_format));
                    accumulators_[event.weight_buffer][row][global_column]
                        += partial;
                }
                ++contribution_counts_[event.weight_buffer][row][global_column];
            }
        }
    }

    bool column_output_ready(std::size_t weight_buffer, std::size_t row, std::size_t column_block) const
    {
        const auto global_column_base = column_block * hw::kMxmSupercellColumns;
        for (std::size_t local_column = 0; local_column < hw::kMxmSupercellColumns; ++local_column) {
            if (contribution_counts_[weight_buffer][row][global_column_base + local_column]
                < hw::kMxmSupercellsPerPlane) {
                return false;
            }
        }
        return true;
    }

    void emit_column_output(TileArrayModel& mem, std::size_t column_block, const ActivationEvent& event)
    {
        const auto row_count = compute_row_count(event.compute_mode);
        if (event.compute_mode == MxmComputeMode::Block8) {
            if (event.row % hw::kMxmBlockRows != 0) {
                throw std::logic_error(
                    "MXM Block8 output row is not aligned to the wide accumulator");
            }
            MxmBlockAccumulator::Segment block_values {};
            for (std::size_t output_row = 0;
                 output_row < row_count;
                 ++output_row) {
                const auto row = event.row + output_row;
                check_column_output_ready(
                    event.weight_buffer,
                    row,
                    column_block);
                const auto global_column_base =
                    column_block * hw::kMxmSupercellColumns;
                for (std::size_t lane = 0;
                     lane < hw::kMxmSupercellColumns;
                     ++lane) {
                    block_values[output_row][lane] =
                        accumulators_[event.weight_buffer][row]
                                     [global_column_base + lane];
                }
            }
            const auto address = event.accumulator_address
                + (event.row / hw::kMxmBlockRows)
                    * event.accumulator_row_stride;
            block_accumulator_.accumulate(
                address,
                column_block,
                block_values);
            if (event.accumulator_destination
                == MxmAccumulatorDestination::Sram) {
                return;
            }

            const auto accumulated =
                block_accumulator_.read(address, column_block);
            emit_block_bf16_stream_values(
                mem,
                column_block,
                event.output_stream_base,
                address,
                accumulated);
            if (event.accumulator_clear) {
                block_accumulator_.clear_segment(address, column_block);
            }
            return;
        }

        for (std::size_t output_row = 0;
             output_row < row_count;
             ++output_row) {
            const auto row = (event.row + output_row) % hw::kMxmRows;
            check_column_output_ready(
                event.weight_buffer,
                row,
                column_block);

            ColumnOutput output {row, column_block, {}};
            const auto global_column_base =
                column_block * hw::kMxmSupercellColumns;
            for (std::size_t lane = 0;
                 lane < hw::kLanesPerTile;
                 ++lane) {
                output.values[lane] =
                    accumulators_[event.weight_buffer][row]
                                 [global_column_base + lane];
            }
            const auto address = event.accumulator_address
                + row * event.accumulator_row_stride;
            accumulator_.accumulate(address, column_block, output.values);
            if (event.accumulator_destination
                == MxmAccumulatorDestination::Sram) {
                continue;
            }

            const auto accumulated =
                accumulator_.read(address, column_block);
            if (event.accumulator_output_format
                == MxmAccumulatorOutputFormat::BFloat16) {
                emit_bf16_stream_values(
                    mem, column_block, event.output_stream_base,
                    row, accumulated);
            } else {
                emit_stream_values(
                    mem, column_block, event.output_stream_base,
                    row, accumulated);
            }
            if (event.accumulator_clear) {
                accumulator_.clear_segment(address, column_block);
            }
            output.values = accumulated;
            last_outputs_.push_back(output);
        }
    }

    void check_column_output_ready(
        std::size_t weight_buffer,
        std::size_t row,
        std::size_t column_block) const
    {
        if (column_output_ready(weight_buffer, row, column_block)) {
            return;
        }
        const auto global_column_base =
            column_block * hw::kMxmSupercellColumns;
        const auto detail = " b" + std::to_string(weight_buffer)
            + ":row=" + std::to_string(row)
            + ":count0="
            + std::to_string(
                contribution_counts_[weight_buffer][row]
                                    [global_column_base]);
        throw std::logic_error(
            "MXM automatic output reached column block "
            + std::to_string(column_block)
            + " before a complete result row was ready;" + detail);
    }

    static void emit_stream_values(
        TileArrayModel& mem,
        std::size_t column_block,
        std::size_t stream_base,
        std::uint64_t vector_tag,
        const ResultValues& values)
    {
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const auto value = values[lane];
            const auto raw = std::bit_cast<std::uint32_t>(value);
            const std::array<std::uint8_t, 4> bytes {
                static_cast<std::uint8_t>(raw & 0xffu),
                static_cast<std::uint8_t>((raw >> 8) & 0xffu),
                static_cast<std::uint8_t>((raw >> 16) & 0xffu),
                static_cast<std::uint8_t>((raw >> 24) & 0xffu),
            };
            for (std::size_t byte = 0; byte < bytes.size(); ++byte) {
                mem.set_west_stream_cell(
                    column_block,
                    lane,
                    stream_base + byte,
                    StreamCell::Valid(
                        bytes[byte],
                        lane + 1 == hw::kLanesPerTile,
                        vector_tag));
            }
        }
    }

    static void emit_bf16_stream_values(
        TileArrayModel& mem,
        std::size_t column_block,
        std::size_t stream_base,
        std::uint64_t vector_tag,
        const ResultValues& values)
    {
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const auto raw = Bf16::from_float(values[lane]).bits();
            for (std::size_t byte = 0; byte < sizeof(std::uint16_t); ++byte) {
                mem.set_west_stream_cell(
                    column_block,
                    lane,
                    stream_base + byte,
                    StreamCell::Valid(
                        static_cast<std::uint8_t>(
                            (raw >> (byte * 8)) & 0xffu),
                        lane + 1 == hw::kLanesPerTile,
                        vector_tag));
            }
        }
    }

    static void emit_block_stream_values(
        TileArrayModel& mem,
        std::size_t column_block,
        std::size_t stream_base,
        std::uint64_t vector_tag,
        const MxmBlockAccumulator::Segment& values)
    {
        for (std::size_t output_row = 0;
             output_row < hw::kMxmBlockRows;
             ++output_row) {
            for (std::size_t lane = 0;
                 lane < hw::kMxmSupercellColumns;
                 ++lane) {
                const auto raw =
                    std::bit_cast<std::uint32_t>(values[output_row][lane]);
                for (std::size_t byte = 0; byte < sizeof(float); ++byte) {
                    mem.set_west_stream_cell(
                        column_block,
                        lane,
                        stream_base + output_row * sizeof(float) + byte,
                        StreamCell::Valid(
                            static_cast<std::uint8_t>(
                                (raw >> (byte * 8)) & 0xffu),
                            lane + 1 == hw::kMxmSupercellColumns,
                            vector_tag));
                }
            }
        }
    }

    static void emit_block_bf16_stream_values(
        TileArrayModel& mem,
        std::size_t column_block,
        std::size_t stream_base,
        std::uint64_t vector_tag,
        const MxmBlockAccumulator::Segment& values)
    {
        for (std::size_t output_row = 0;
             output_row < hw::kMxmBlockRows;
             ++output_row) {
            for (std::size_t lane = 0;
                 lane < hw::kMxmSupercellColumns;
                 ++lane) {
                const auto raw = Bf16::from_float(
                    values[output_row][lane]).bits();
                for (std::size_t byte = 0;
                     byte < sizeof(std::uint16_t);
                     ++byte) {
                    mem.set_west_stream_cell(
                        column_block,
                        lane,
                        stream_base
                            + output_row * sizeof(std::uint16_t)
                            + byte,
                        StreamCell::Valid(
                            static_cast<std::uint8_t>(
                                (raw >> (byte * 8)) & 0xffu),
                            lane + 1 == hw::kMxmSupercellColumns,
                            vector_tag));
                }
            }
        }
    }

    bool pipelines_empty() const
    {
        for (const auto& column : east_pipeline_) {
            for (const auto& lane : column) {
                if (!lane.empty()) {
                    return false;
                }
            }
        }
        return true;
    }

    MxmArray array_{};
    MxmAccumulator accumulator_{};
    MxmBlockAccumulator block_accumulator_{};
    MxmWeightDequantizer decode_dequantizer_{};
    std::array<
        std::array<
            std::array<
                std::array<float, hw::kLanesPerTile>,
                hw::kMxmSupercellsPerPlane>,
            hw::kMxmSupercellsPerPlane>,
        kWeightBuffers>
        decode_activation_buffers_{};
    std::array<
        std::array<
            std::array<
                std::array<bool, hw::kLanesPerTile>,
                hw::kMxmSupercellsPerPlane>,
            hw::kMxmSupercellsPerPlane>,
        kWeightBuffers>
        decode_activation_valid_{};
    std::optional<DecodeWaveState> decode_launch_{};
    std::array<std::optional<DecodeWaveState>, kDecodePipelineStages>
        decode_stages_{};
    std::array<
        std::optional<DecodeCompletedOutput>,
        hw::kTileRows>
        decode_output_pipeline_{};
    bool decode_streaming_active_{false};
    MxmDecodeLayout decode_layout_{MxmDecodeLayout::Linear1x16};
    MxmControlSlice control_;
    bool active_{false};
    std::array<std::array<std::deque<ActivationEvent>, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane>
        east_pipeline_{};
    std::array<std::array<bool, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane> last_computing_{};
    std::array<std::array<std::size_t, hw::kMxmSupercellsPerPlane>, kWeightBuffers> next_row_for_tile_{};
    std::array<bool, kWeightBuffers> compute_active_by_buffer_{};
    std::array<std::array<std::array<float, hw::kMxmColumns>, hw::kMxmRows>, kWeightBuffers> accumulators_{};
    std::array<std::array<std::array<std::size_t, hw::kMxmColumns>, hw::kMxmRows>, kWeightBuffers>
        contribution_counts_{};
    std::vector<ColumnOutput> last_outputs_{};
};

} // namespace ftlpu
