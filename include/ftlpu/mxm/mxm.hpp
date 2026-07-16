#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/mem/tile_array.hpp"
#include "ftlpu/mxm/array.hpp"
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
    using ActivationData = std::array<std::int8_t, hw::kLanesPerTile>;
    using ResultValues = std::array<std::int32_t, hw::kMxmSupercellColumns>;

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

    void tick_datapath(
        TileArrayModel& mem,
        std::size_t mxm_id,
        std::ostream* os = nullptr,
        std::optional<std::size_t> log_tile = std::nullopt)
    {
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
            const auto data = collect_activation(mem, tile, stream_base);
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

    static ActivationData collect_activation(const TileArrayModel& mem, std::size_t tile, std::size_t stream_base)
    {
        constexpr auto kTargetSreg = hw::kMxmBoundaryStreamRegisterColumn;
        if (stream_base >= hw::kStreams) {
            throw std::out_of_range("MXM activation stream is outside the stream set");
        }
        ActivationData data {};
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const auto& slot = mem.east_register(tile, lane, kTargetSreg, stream_base);
            if (!slot.has_value()) {
                throw std::logic_error("MXM Compute reached tile before SXM east stream arrived at sreg12");
            }
            data[lane] = static_cast<std::int8_t>(slot->data);
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

    void emit_column_output(TileArrayModel& mem, std::size_t column_block, const ActivationEvent& event)
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
                mem.set_west_stream_input(
                    column_block,
                    lane,
                    event.output_stream_base + byte,
                    TileArrayModel::DataWord {
                        bytes[byte],
                        lane + 1 == hw::kLanesPerTile,
                    });
            }
        }
        last_outputs_.push_back(output);
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
    MxmControlSlice control_;
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
