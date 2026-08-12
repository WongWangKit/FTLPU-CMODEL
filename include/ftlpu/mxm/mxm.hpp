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
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ftlpu {

class Mxm {
public:
    static constexpr std::size_t kWeightBuffers = MxmSupercell::kWeightBuffers;
    static constexpr std::size_t kLocalMacStages =
        MxmSupercell::kWavefrontStages;
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
        if (!local_mac_pipelines_) {
            local_mac_pipelines_ = std::make_unique<LocalMacGrid>();
        }
        if (!vertical_partial_links_) {
            vertical_partial_links_ =
                std::make_unique<VerticalPartialGrid>();
        }
        for (auto& tile : *local_mac_pipelines_) {
            for (auto& column : tile) {
                for (auto& stage : column) stage.reset();
            }
        }
        for (auto& tile : *vertical_partial_links_) {
            for (auto& column : tile) column.reset();
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
        last_deskew_writes_ = 0;
        last_deskew_vectors_ = 0;
        active_ = false;
    }

    void tick_datapath(
        TileArrayModel& mem,
        std::size_t mxm_id,
        std::ostream* os = nullptr,
        std::optional<std::size_t> log_tile = std::nullopt)
    {
        last_outputs_.clear();
        last_deskew_writes_ = 0;
        last_deskew_vectors_ = 0;
        for (auto& row : last_computing_) row.fill(false);

        for (std::size_t tile = 0;
             tile < hw::kMxmSupercellsPerPlane;
             ++tile) {
            if (const auto load =
                    control_.decode_activation_load_pulse(tile);
                load.has_value()) {
                if (decode_buffer_in_flight(load->activation_buffer)) {
                    throw std::logic_error(
                        "MXM decode cannot overwrite an active ping-pong activation buffer");
                }
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
                && !compute_active_by_buffer_[weight_buffer]) {
                reset_buffer_state(weight_buffer);
            }
            const auto absolute_row = next_row_for_tile_[weight_buffer][tile];
            next_row_for_tile_[weight_buffer][tile] += row_count;
            const auto row = absolute_row % hw::kMxmRows;
            const auto data = collect_activation(
                mem,
                tile,
                stream_base,
                compute.data_format,
                compute.compute_mode);
            east_pipeline_[0][tile].push_back(ActivationEvent {
                absolute_row / row_count,
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

        // Activation/control metadata moves one supercell column to the right
        // per cycle.  A cell accepts the wave into MAC stage 0 immediately;
        // it does not wait for the cell's previous seven waves to retire.
        decltype(east_pipeline_) next_east_pipeline {};
        for (std::size_t column = 0;
             column < hw::kMxmSupercellsPerPlane; ++column) {
            for (std::size_t tile = 0;
                 tile < hw::kMxmSupercellsPerPlane; ++tile) {
                auto& arrivals = east_pipeline_[column][tile];
                if (arrivals.size() > 1) {
                    throw std::logic_error(
                        "MXM supercell received more than one activation wave in a cycle");
                }
                if (arrivals.empty()) continue;
                auto event = std::move(arrivals.front());
                auto& stage0 = (*local_mac_pipelines_)[tile][column][0];
                if (stage0.has_value()) {
                    throw std::logic_error(
                        "MXM supercell MAC pipeline stage 0 is occupied");
                }
                stage0 = PartialEvent {event, {}};
                if (column + 1 < hw::kMxmSupercellsPerPlane) {
                    next_east_pipeline[column + 1][tile].push_back(
                        std::move(event));
                }
            }
        }
        east_pipeline_ = std::move(next_east_pipeline);

        using PartialGrid = std::array<
            std::array<std::optional<PartialEvent>,
                       hw::kMxmSupercellsPerPlane>,
            hw::kMxmSupercellsPerPlane>;
        auto local_completed = PartialGrid {};
        auto computing = std::array<
            std::array<bool, hw::kMxmSupercellsPerPlane>,
            hw::kMxmSupercellsPerPlane> {};
        for (std::size_t tile = 0;
             tile < hw::kMxmSupercellsPerPlane; ++tile) {
            for (std::size_t column = 0;
                 column < hw::kMxmSupercellsPerPlane; ++column) {
                auto next = LocalMacPipeline {};
                auto& pipeline = (*local_mac_pipelines_)[tile][column];
                for (std::size_t stage = 0;
                     stage < kLocalMacStages; ++stage) {
                    if (!pipeline[stage].has_value()) continue;
                    computing[tile][column] = true;
                    auto entry = std::move(*pipeline[stage]);
                    accumulate_local_mac(entry, tile, column, stage);
                    if (stage + 1 == kLocalMacStages) {
                        local_completed[tile][column] = std::move(entry);
                    }
                    else {
                        next[stage + 1] = std::move(entry);
                    }
                }
                pipeline = std::move(next);
            }
        }
        reduce_vertical_partials(mem, std::move(local_completed));
        for (std::size_t tile = 0;
             tile < hw::kMxmSupercellsPerPlane; ++tile) {
            for (std::size_t column = 0;
                 column < hw::kMxmSupercellsPerPlane; ++column) {
                last_computing_[tile][column] =
                    last_computing_[tile][column]
                    || computing[tile][column];
            }
        }

        for (std::size_t tile = 0; tile < hw::kMxmSupercellsPerPlane; ++tile) {
            const auto read = control_.accumulator_read_pulse(tile);
            if (!read.has_value()) continue;
            if (read->compute_mode == MxmComputeMode::Block8) {
                const auto values =
                    block_accumulator_.read(read->address, tile);
                if (read->output_format
                    == MxmAccumulatorOutputFormat::BFloat16) {
                    emit_block_bf16_stream_values(
                        mem, tile, read->stream_base,
                        read->address, values);
                } else {
                    emit_block_stream_values(
                        mem, tile, read->stream_base,
                        read->address, values);
                }
                if (read->clear) {
                    block_accumulator_.clear_segment(
                        read->address,
                        tile);
                }
                continue;
            }
            const auto values = accumulator_.read(read->address, tile);
            if (read->output_format
                == MxmAccumulatorOutputFormat::BFloat16) {
                emit_bf16_stream_values(
                    mem, tile, read->stream_base,
                    read->address, values);
            } else {
                emit_stream_values(
                    mem, tile, read->stream_base,
                    read->address, values);
            }
            if (read->clear) {
                accumulator_.clear_segment(read->address, tile);
            }
        }

        if (active_ && pipelines_empty()) {
            active_ = false;
        }
        for (std::size_t buffer = 0;
             buffer < kWeightBuffers; ++buffer) {
            current_compute_active_by_buffer[buffer] =
                current_compute_active_by_buffer[buffer]
                || buffer_in_flight(buffer);
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

    std::size_t last_deskew_writes() const noexcept
    {
        return last_deskew_writes_;
    }

    std::size_t last_deskew_vectors() const noexcept
    {
        return last_deskew_vectors_;
    }

    void validate_weight_buffer_load(
        std::size_t buffer,
        std::size_t tile) const
    {
        check_weight_buffer(buffer);
        check_tile(tile);
        if (buffer_tile_in_flight(buffer, tile)) {
            throw std::logic_error(
                "MXM cannot overwrite an active ping-pong weight buffer");
        }
    }

private:
    static constexpr std::size_t kLinearDecodeStages =
        hw::kTileRows * hw::kMxmSupercellsPerPlane
        + kLocalMacStages - 1;
    static constexpr std::size_t kNativeDecodeStages =
        hw::kTileRows + hw::kMxmSupercellsPerPlane
        + kLocalMacStages - 2;
    static constexpr std::size_t kDecodePipelineStages =
        kLinearDecodeStages;

    using DecodeWeightCell = std::array<
        std::array<float, hw::kMxmSupercellColumns>,
        hw::kLanesPerTile>;
    struct DecodeCellWave {
        std::optional<DecodeWeightCell> weights{};
        ResultValues partial{};
    };
    struct DecodeWavePayload {
        std::array<
            std::array<DecodeCellWave, hw::kMxmSupercellsPerPlane>,
            hw::kMxmSupercellsPerPlane> cells{};
    };
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
        std::shared_ptr<DecodeWavePayload> payload{};
    };

    struct DecodeCompletedOutput {
        std::uint64_t wave_id{0};
        std::size_t output_stream_base{0};
        std::array<ResultValues, hw::kMxmSupercellsPerPlane> values{};
    };

    struct ActivationEvent {
        std::uint64_t wave_id{0};
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

    // Each 8x8 MAC wave completes lane c at local stage 7+c.  These 28
    // triangular delay registers explicitly hold the early lanes for
    // 7/6/.../0 cycles so that the SXM-facing boundary receives one flat
    // eight-lane vector at stage 14.
    class LaneDeskewRegisters {
    public:
        static constexpr std::size_t kMaximumDelay =
            hw::kMxmSupercellColumns - 1;
        static constexpr std::size_t kRegisterCount =
            kMaximumDelay * (kMaximumDelay + 1) / 2;

        void advance()
        {
            for (const auto& value : outputs_) {
                if (value.has_value()) {
                    throw std::logic_error(
                        "MXM lane deskew output was not consumed");
                }
            }
            for (std::size_t lane = 0;
                 lane < hw::kMxmSupercellColumns;
                 ++lane) {
                const auto depth = delay(lane);
                if (depth == 0) continue;
                const auto base = offset(lane);
                outputs_[lane] = std::move(registers_[base]);
                for (std::size_t stage = 0; stage + 1 < depth; ++stage) {
                    registers_[base + stage] =
                        std::move(registers_[base + stage + 1]);
                }
                registers_[base + depth - 1].reset();
            }
        }

        void latch(std::size_t lane, float value)
        {
            if (lane >= hw::kMxmSupercellColumns) {
                throw std::out_of_range(
                    "MXM lane deskew input is outside the tile");
            }
            const auto depth = delay(lane);
            if (depth == 0) {
                if (outputs_[lane].has_value()) {
                    throw std::logic_error(
                        "MXM lane deskew direct output is occupied");
                }
                outputs_[lane] = value;
                return;
            }
            auto& destination = registers_[offset(lane) + depth - 1];
            if (destination.has_value()) {
                throw std::logic_error(
                    "MXM lane deskew register collision");
            }
            destination = value;
        }

        ResultValues take_aligned()
        {
            auto result = ResultValues {};
            for (std::size_t lane = 0;
                 lane < hw::kMxmSupercellColumns;
                 ++lane) {
                if (!outputs_[lane].has_value()) {
                    throw std::logic_error(
                        "MXM lane deskew vector is not flat at stage 14");
                }
                result[lane] = *outputs_[lane];
                outputs_[lane].reset();
            }
            return result;
        }

    private:
        static constexpr std::size_t delay(std::size_t lane)
        {
            return kMaximumDelay - lane;
        }

        static constexpr std::size_t offset(std::size_t lane)
        {
            auto result = std::size_t {0};
            for (std::size_t previous = 0; previous < lane; ++previous) {
                result += delay(previous);
            }
            return result;
        }

        std::array<std::optional<float>, kRegisterCount> registers_{};
        std::array<
            std::optional<float>, hw::kMxmSupercellColumns> outputs_{};
    };

    // One entry advances through the 15 diagonals of a physical 8x8
    // supercell.  Stage s executes every MAC at row+column=s (with the row
    // direction reversed for upward partial sums).  The pipeline still
    // accepts one new wave per cycle.
    struct PartialEvent {
        ActivationEvent event{};
        ActivationBlock values{};
        std::array<LaneDeskewRegisters, hw::kMxmBlockRows> deskew{};
    };

    using LocalMacPipeline =
        std::array<std::optional<PartialEvent>, kLocalMacStages>;
    using LocalMacGrid = std::array<
        std::array<LocalMacPipeline, hw::kMxmSupercellsPerPlane>,
        hw::kMxmSupercellsPerPlane>;
    using VerticalPartialGrid = std::array<
        std::array<std::optional<PartialEvent>,
                   hw::kMxmSupercellsPerPlane>,
        hw::kMxmSupercellsPerPlane>;

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
            {},
            std::make_shared<DecodeWavePayload>()};
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
                                      std::size_t partial_column,
                                      std::size_t local_stage) {
            if (!state.payload) {
                throw std::logic_error("MXM decode wave lost its cell payload");
            }
            auto& cell = state.payload->cells[tile][column];
            if (local_stage == 0) {
                cell.weights = consume_decode_weight_cell(
                    mem, tile, column, state.dequant);
                if (!cell.weights.has_value()) {
                    throw std::logic_error(
                        "MXM decode wave reached a supercell before its parallel weight block arrived");
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
            }
            if (!cell.weights.has_value()) {
                throw std::logic_error(
                    "MXM decode supercell MAC wavefront has no latched weights");
            }
            for (std::size_t output_lane = 0;
                 output_lane < hw::kMxmSupercellColumns;
                 ++output_lane) {
                if (local_stage < output_lane) continue;
                const auto vertical_step = local_stage - output_lane;
                if (vertical_step >= hw::kMxmSupercellRows) continue;
                const auto input_lane =
                    hw::kMxmSupercellRows - 1 - vertical_step;
                cell.partial[output_lane] +=
                    decode_activation_buffers_[state.activation_buffer]
                                              [tile][column][input_lane]
                    * (*cell.weights)[input_lane][output_lane];
            }
            if (local_stage + 1 == kLocalMacStages) {
                for (std::size_t output_lane = 0;
                     output_lane < hw::kMxmSupercellColumns;
                     ++output_lane) {
                    state.partial[partial_column][output_lane] +=
                        cell.partial[output_lane];
                }
            }
            last_computing_[tile][column] = true;
            if (os != nullptr) {
                *os << "  decode wave=" << state.wave_id
                    << " stage=" << stage
                    << " local_stage=" << local_stage
                    << " cell=(" << tile << ',' << column << ')'
                    << " local_partial0=" << cell.partial[0]
                    << '\n';
            }
        };

        if (state.layout == MxmDecodeLayout::Linear1x16) {
            for (std::size_t column = 0;
                 column < hw::kMxmSupercellsPerPlane; ++column) {
                for (std::size_t tile = 0;
                     tile < hw::kTileRows; ++tile) {
                    const auto launch_stage =
                        column * hw::kTileRows + tile;
                    if (stage < launch_stage) continue;
                    const auto local_stage = stage - launch_stage;
                    if (local_stage >= kLocalMacStages) continue;
                    execute_cell(tile, column, 0, local_stage);
                }
            }
            return;
        }

        for (std::size_t tile = 0;
             tile < hw::kTileRows; ++tile) {
            for (std::size_t column = 0;
                 column < hw::kMxmSupercellsPerPlane;
                 ++column) {
                const auto launch_stage = tile + column;
                if (stage < launch_stage) continue;
                const auto local_stage = stage - launch_stage;
                if (local_stage < kLocalMacStages) {
                    execute_cell(
                        tile, column, column, local_stage);
                }
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

    void accumulate_local_mac(
        PartialEvent& partial,
        std::size_t tile,
        std::size_t column,
        std::size_t stage)
    {
        const auto& event = partial.event;
        if (event.tile != tile) {
            throw std::logic_error(
                "MXM activation wave entered the wrong supercell row");
        }
        const auto row_count = compute_row_count(event.compute_mode);
        for (std::size_t output_row = 0;
             output_row < row_count;
             ++output_row) {
            partial.deskew[output_row].advance();
        }
        // The activation vector is skewed across the 8x8 cell.  At local
        // stage s, column c consumes row 7-(s-c), which is exactly one
        // row+column diagonal.  Each output column therefore completes at
        // stages 7..14 while a new token may enter stage zero every cycle.
        for (std::size_t output_row = 0;
             output_row < row_count; ++output_row) {
            for (std::size_t local_column = 0;
                 local_column < hw::kMxmSupercellColumns;
                 ++local_column) {
                if (stage < local_column) continue;
                const auto vertical_step = stage - local_column;
                if (vertical_step >= hw::kMxmSupercellRows) continue;
                const auto input_row =
                    hw::kMxmSupercellRows - 1 - vertical_step;
                const auto activation =
                    event.data[output_row][input_row];
                partial.values[output_row][local_column] += activation
                    * array_.weight(
                        event.weight_buffer,
                        tile,
                        column,
                        input_row,
                        local_column,
                        event.data_format);
            }
        }
        for (std::size_t local_column = 0;
             local_column < hw::kMxmSupercellColumns;
             ++local_column) {
            if (stage
                != hw::kMxmSupercellRows - 1 + local_column) {
                continue;
            }
            for (std::size_t output_row = 0;
                 output_row < row_count;
                 ++output_row) {
                partial.deskew[output_row].latch(
                    local_column,
                    partial.values[output_row][local_column]);
                ++last_deskew_writes_;
            }
        }
        if (stage + 1 == kLocalMacStages) {
            for (std::size_t output_row = 0;
                 output_row < row_count;
                 ++output_row) {
                partial.values[output_row] =
                    partial.deskew[output_row].take_aligned();
                ++last_deskew_vectors_;
            }
        }
    }

    static bool same_partial_wave(
        const ActivationEvent& lhs,
        const ActivationEvent& rhs)
    {
        return lhs.wave_id == rhs.wave_id
            && lhs.row == rhs.row
            && lhs.weight_buffer == rhs.weight_buffer
            && lhs.output_stream_base == rhs.output_stream_base
            && lhs.accumulator_address == rhs.accumulator_address
            && lhs.accumulator_row_stride == rhs.accumulator_row_stride
            && lhs.accumulator_destination == rhs.accumulator_destination
            && lhs.accumulator_clear == rhs.accumulator_clear
            && lhs.compute_mode == rhs.compute_mode
            && lhs.accumulator_output_format
                == rhs.accumulator_output_format;
    }

    static void add_vertical_partial(
        PartialEvent& local,
        const PartialEvent& from_lower)
    {
        if (!same_partial_wave(local.event, from_lower.event)) {
            throw std::logic_error(
                "MXM vertical partial-sum wave/tag mismatch");
        }
        const auto row_count = compute_row_count(local.event.compute_mode);
        for (std::size_t output_row = 0;
             output_row < row_count; ++output_row) {
            for (std::size_t column = 0;
                 column < hw::kMxmSupercellColumns; ++column) {
                local.values[output_row][column] +=
                    from_lower.values[output_row][column];
            }
        }
    }

    template <typename PartialGrid>
    void reduce_vertical_partials(
        TileArrayModel& mem,
        PartialGrid local_completed)
    {
        auto next_links = VerticalPartialGrid {};
        for (std::size_t column = 0;
             column < hw::kMxmSupercellsPerPlane; ++column) {
            for (std::size_t tile = 0;
                 tile < hw::kMxmSupercellsPerPlane; ++tile) {
                auto& local = local_completed[tile][column];
                if (tile != 0) {
                    auto& incoming = (*vertical_partial_links_)[tile][column];
                    if (local.has_value() != incoming.has_value()) {
                        throw std::logic_error(
                            "MXM local and vertical partial sums lost cycle alignment");
                    }
                    if (local.has_value()) {
                        add_vertical_partial(*local, *incoming);
                    }
                }
                if (!local.has_value()) continue;

                if (tile + 1 < hw::kMxmSupercellsPerPlane) {
                    next_links[tile + 1][column] = std::move(local);
                }
                else {
                    emit_column_output(mem, column, *local);
                }
            }
        }
        *vertical_partial_links_ = std::move(next_links);
    }

    void emit_column_output(
        TileArrayModel& mem,
        std::size_t column_block,
        const PartialEvent& partial)
    {
        const auto& event = partial.event;
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
                for (std::size_t lane = 0;
                     lane < hw::kMxmSupercellColumns;
                     ++lane) {
                    block_values[output_row][lane] =
                        partial.values[output_row][lane];
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

            ColumnOutput output {row, column_block, {}};
            for (std::size_t lane = 0;
                 lane < hw::kLanesPerTile;
                 ++lane) {
                output.values[lane] = partial.values[output_row][lane];
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
        for (const auto& tile : *local_mac_pipelines_) {
            for (const auto& column : tile) {
                for (const auto& stage : column) {
                    if (stage.has_value()) return false;
                }
            }
        }
        for (const auto& tile : *vertical_partial_links_) {
            for (const auto& column : tile) {
                if (column.has_value()) return false;
            }
        }
        return true;
    }

    bool buffer_in_flight(std::size_t buffer) const
    {
        for (const auto& column : east_pipeline_) {
            for (const auto& tile : column) {
                for (const auto& event : tile) {
                    if (event.weight_buffer == buffer) return true;
                }
            }
        }
        for (const auto& tile : *local_mac_pipelines_) {
            for (const auto& column : tile) {
                for (const auto& stage : column) {
                    if (stage.has_value()
                        && stage->event.weight_buffer == buffer) {
                        return true;
                    }
                }
            }
        }
        for (const auto& tile : *vertical_partial_links_) {
            for (const auto& column : tile) {
                if (column.has_value()
                    && column->event.weight_buffer == buffer) {
                    return true;
                }
            }
        }
        return false;
    }

    bool buffer_tile_in_flight(
        std::size_t buffer,
        std::size_t tile) const
    {
        for (const auto& column : east_pipeline_) {
            for (const auto& event : column[tile]) {
                if (event.weight_buffer == buffer) return true;
            }
        }
        for (const auto& column : (*local_mac_pipelines_)[tile]) {
            for (const auto& stage : column) {
                if (stage.has_value()
                    && stage->event.weight_buffer == buffer) {
                    return true;
                }
            }
        }
        return false;
    }

    bool decode_buffer_in_flight(std::size_t buffer) const
    {
        if (decode_launch_.has_value()
            && decode_launch_->activation_buffer == buffer) {
            return true;
        }
        for (const auto& stage : decode_stages_) {
            if (stage.has_value()
                && stage->activation_buffer == buffer) {
                return true;
            }
        }
        return false;
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
    std::unique_ptr<LocalMacGrid> local_mac_pipelines_{};
    // Entry [tile][column] is the registered partial arriving from tile-1.
    // Row zero has no lower neighbor and therefore remains empty.
    std::unique_ptr<VerticalPartialGrid> vertical_partial_links_{};
    std::array<std::array<bool, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane> last_computing_{};
    std::array<std::array<std::size_t, hw::kMxmSupercellsPerPlane>, kWeightBuffers> next_row_for_tile_{};
    std::array<bool, kWeightBuffers> compute_active_by_buffer_{};
    std::vector<ColumnOutput> last_outputs_{};
    std::size_t last_deskew_writes_{0};
    std::size_t last_deskew_vectors_{0};
};

} // namespace ftlpu
