#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/mem/tile_array.hpp"
#include "ftlpu/mxm/array.hpp"
#include "ftlpu/mxm/control_slice.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace ftlpu {

class Mxm {
public:
    static constexpr std::size_t kWeightBuffers = MxmSupercell::kWeightBuffers;
    using ActivationVector = std::array<std::int8_t, hw::kMxmRows>;
    using ResultValues = std::array<std::int32_t, hw::kMxmSupercellColumns>;
    using ResultVector = std::array<std::int32_t, hw::kMxmColumns>;

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

    MxmArray& array() { return array_; }
    const MxmArray& array() const { return array_; }
    MxmControlSlice& control() { return control_; }
    const MxmControlSlice& control() const { return control_; }

    void reset_datapath()
    {
        for (auto& column : east_pipeline_) column.clear();
        for (auto& row : last_computing_) row.fill(false);
        next_row_.fill(0);
        compute_active_by_buffer_.fill(false);
        last_outputs_.clear();
    }

    void tick_datapath(
        TileArrayModel& mem,
        std::size_t mxm_id,
        std::ostream* os = nullptr,
        std::optional<std::size_t> log_tile = std::nullopt)
    {
        last_outputs_.clear();
        auto current_compute_active = std::array<bool, kWeightBuffers> {};

        if (control_.compute_active(0)) {
            const auto weight_buffer = control_.compute_weight_buffer(0).value_or(0);
            const auto stream_base = control_.compute_activation_stream_base(0).value_or(0);
            const auto output_stream_base = control_.output_stream_base(0).value_or(0);
            check_weight_buffer(weight_buffer);
            current_compute_active[weight_buffer] = true;
            if (!compute_active_by_buffer_[weight_buffer]) next_row_[weight_buffer] = 0;

            const auto row = next_row_[weight_buffer]++;
            east_pipeline_[0].push_back(ActivationEvent {
                row,
                weight_buffer,
                output_stream_base,
                collect_activation(mem, stream_base),
            });
            if (os != nullptr && (!log_tile.has_value() || *log_tile == 0)) {
                *os << "  MXM" << mxm_id << " consume 320B activation row=" << row
                    << " buffer=" << weight_buffer << " stream=" << stream_base
                    << " out=" << output_stream_base << '\n';
            }
        }

        auto next_pipeline = std::array<std::deque<ActivationEvent>, hw::kMxmSupercellsPerPlane> {};
        for (auto& row : last_computing_) row.fill(false);
        for (std::size_t column = 0; column < hw::kMxmSupercellsPerPlane; ++column) {
            for (const auto& event : east_pipeline_[column]) {
                auto advanced = event;
                for (std::size_t tile = 0; tile < hw::kMxmSupercellsPerPlane; ++tile) {
                    last_computing_[tile][column] = true;
                }
                const auto block = compute_column(event, column);
                std::copy(
                    block.begin(),
                    block.end(),
                    advanced.results.begin() + column * hw::kMxmSupercellColumns);
                if (column + 1 < hw::kMxmSupercellsPerPlane) {
                    next_pipeline[column + 1].push_back(std::move(advanced));
                } else {
                    emit_vector_output(mem, advanced);
                }
            }
        }
        east_pipeline_ = std::move(next_pipeline);
        compute_active_by_buffer_ = current_compute_active;
    }

    bool computing_cell(std::size_t tile, std::size_t column_block) const
    {
        check_tile(tile);
        if (column_block >= hw::kMxmSupercellsPerPlane) {
            throw std::out_of_range("MXM column block is outside the 20-column array");
        }
        return last_computing_[tile][column_block];
    }

    const std::vector<ColumnOutput>& last_outputs() const { return last_outputs_; }

private:
    struct ActivationEvent {
        std::size_t row{0};
        std::size_t weight_buffer{0};
        std::size_t output_stream_base{0};
        ActivationVector data{};
        ResultVector results{};
    };

    static void check_tile(std::size_t tile)
    {
        if (tile >= hw::kMxmSupercellsPerPlane) throw std::out_of_range("MXM tile is outside the 20-row array");
    }

    static void check_weight_buffer(std::size_t weight_buffer)
    {
        if (weight_buffer >= kWeightBuffers) throw std::out_of_range("MXM weight buffer is outside the two-buffer set");
    }

    static ActivationVector collect_activation(const TileArrayModel& mem, std::size_t stream_base)
    {
        constexpr auto kTargetSreg = hw::kStreamRegisterColumns - 1;
        if (stream_base >= hw::kStreams) throw std::out_of_range("MXM activation stream is outside the stream set");
        auto data = ActivationVector {};
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                const auto& slot = mem.east_register(tile, lane, kTargetSreg, stream_base);
                if (!slot.has_value()) throw std::logic_error("MXM Compute reached tile before 320B activation arrived at sreg11");
                data[tile * hw::kLanesPerTile + lane] = static_cast<std::int8_t>(slot->data);
            }
        }
        return data;
    }

    ResultValues compute_column(const ActivationEvent& event, std::size_t column_block) const
    {
        auto values = ResultValues {};
        for (std::size_t tile = 0; tile < hw::kMxmSupercellsPerPlane; ++tile) {
            const auto& weights = array_.cell(tile, column_block).weight_buffer(event.weight_buffer);
            const auto activation_base = tile * hw::kLanesPerTile;
            for (std::size_t local_column = 0; local_column < hw::kMxmSupercellColumns; ++local_column) {
                for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                    values[local_column] += static_cast<std::int32_t>(event.data[activation_base + lane])
                        * static_cast<std::int32_t>(weights[lane][local_column]);
                }
            }
        }
        return values;
    }

    void emit_vector_output(TileArrayModel& mem, const ActivationEvent& event)
    {
        for (std::size_t column_block = 0; column_block < hw::kMxmSupercellsPerPlane; ++column_block) {
            auto output = ColumnOutput {event.row, column_block, {}};
            const auto result_base = column_block * hw::kMxmSupercellColumns;
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                const auto value = event.results[result_base + lane];
                output.values[lane] = value;
                const auto raw = static_cast<std::uint32_t>(value);
                const auto bytes = std::array<std::uint8_t, 4> {
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
                        TileArrayModel::DataWord {bytes[byte], lane + 1 == hw::kLanesPerTile});
                }
            }
            last_outputs_.push_back(output);
        }
    }

    MxmArray array_{};
    MxmControlSlice control_;
    std::array<std::deque<ActivationEvent>, hw::kMxmSupercellsPerPlane> east_pipeline_{};
    std::array<std::array<bool, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane> last_computing_{};
    std::array<std::size_t, kWeightBuffers> next_row_{};
    std::array<bool, kWeightBuffers> compute_active_by_buffer_{};
    std::vector<ColumnOutput> last_outputs_{};
};

} // namespace ftlpu
