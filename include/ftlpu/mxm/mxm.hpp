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
    using ActivationData = std::array<std::int8_t, hw::kLanesPerTile>;
    using ResultValues = std::array<std::int32_t, hw::kMxmSupercellColumns>;

    struct ColumnOutput {
        std::size_t row{0};
        std::size_t column_block{0};
        ResultValues values{};
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
        compute_active_by_buffer_.fill(false);
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

        auto current_compute_active_by_buffer = std::array<bool, kWeightBuffers> {};
        for (std::size_t tile = 0; tile < hw::kMxmSupercellsPerPlane; ++tile) {
            if (!control_.compute_active(tile)) {
                continue;
            }
            const auto weight_buffer = control_.compute_weight_buffer(tile).value_or(0);
            const auto stream_base = control_.compute_activation_stream_base(tile).value_or(0);
            const auto output_stream_base = control_.output_stream_base(tile).value_or(0);
            check_weight_buffer(weight_buffer);
            current_compute_active_by_buffer[weight_buffer] = true;
            if (tile == 0 && !compute_active_by_buffer_[weight_buffer]) {
                reset_buffer_state(weight_buffer);
            }
            const auto data = collect_activation(fabric, tile, stream_base);
            const auto row = next_row_for_tile_[weight_buffer][tile]++;
            east_pipeline_[0][tile].push_back(ActivationEvent {tile, row, weight_buffer, output_stream_base, data});
            if (os != nullptr && (!log_tile.has_value() || *log_tile == tile)) {
                *os << "  MXM" << mxm_id << " consume activation tile=" << tile
                    << " row=" << row
                    << " buffer=" << weight_buffer
                    << " stream=" << stream_base
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
                        emit_column_output(fabric, column_block, event);
                    }
                    if (column_block + 1 < hw::kMxmSupercellsPerPlane) {
                        next_pipeline[column_block + 1][tile].push_back(event);
                    }
                }
            }
        }
        east_pipeline_ = std::move(next_pipeline);
        last_computing_ = computing;

        if (active_ && pipelines_empty()) {
            active_ = false;
        }
        compute_active_by_buffer_ = current_compute_active_by_buffer;
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

private:
    struct ActivationEvent {
        std::size_t tile{0};
        std::size_t row{0};
        std::size_t weight_buffer{0};
        std::size_t output_stream_base{0};
        ActivationData data{};
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

    void reset_buffer_state(std::size_t weight_buffer)
    {
        check_weight_buffer(weight_buffer);
        next_row_for_tile_[weight_buffer].fill(0);
        for (auto& row : accumulators_[weight_buffer]) {
            row.fill(0);
        }
        for (auto& row : contribution_counts_[weight_buffer]) {
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

    void compute_column_block(const ActivationEvent& event, std::size_t column_block)
    {
        for (std::size_t local_column = 0; local_column < hw::kMxmSupercellColumns; ++local_column) {
            const auto global_column = column_block * hw::kMxmSupercellColumns + local_column;
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                const auto partial = static_cast<std::int32_t>(event.data[lane])
                    * static_cast<std::int32_t>(
                        array_.weight(event.weight_buffer, event.tile, column_block, lane, local_column));
                accumulators_[event.weight_buffer][event.row][global_column] += partial;
            }
            ++contribution_counts_[event.weight_buffer][event.row][global_column];
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

    void emit_column_output(
        StreamRegisterFabric& fabric,
        std::size_t column_block,
        const ActivationEvent& event)
    {
        if (!column_output_ready(event.weight_buffer, event.row, column_block)) {
            std::string detail;
            const auto global_column_base = column_block * hw::kMxmSupercellColumns;
            detail += " b" + std::to_string(event.weight_buffer)
                + ":row=" + std::to_string(event.row)
                + ":count0="
                + std::to_string(contribution_counts_[event.weight_buffer][event.row][global_column_base]);
            throw std::logic_error(
                "MXM automatic output reached column block " + std::to_string(column_block)
                + " before a complete result row was ready;" + detail);
        }

        ColumnOutput output {event.row, column_block, {}};
        const auto global_column_base = column_block * hw::kMxmSupercellColumns;
        std::array<StreamSegment16, 4> output_segments{};
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const auto value = accumulators_[event.weight_buffer][event.row][global_column_base + lane];
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
        return true;
    }

    MxmArray array_{};
    MxmControlSlice control_;
    MxmStreamPortMap ports_;
    bool active_{false};
    std::array<std::array<std::deque<ActivationEvent>, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane>
        east_pipeline_{};
    std::array<std::array<bool, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane> last_computing_{};
    std::array<std::array<std::size_t, hw::kMxmSupercellsPerPlane>, kWeightBuffers> next_row_for_tile_{};
    std::array<bool, kWeightBuffers> compute_active_by_buffer_{};
    std::array<std::array<std::array<std::int32_t, hw::kMxmColumns>, hw::kMxmRows>, kWeightBuffers> accumulators_{};
    std::array<std::array<std::array<std::size_t, hw::kMxmColumns>, hw::kMxmRows>, kWeightBuffers>
        contribution_counts_{};
    std::vector<ColumnOutput> last_outputs_{};
};

} // namespace ftlpu
