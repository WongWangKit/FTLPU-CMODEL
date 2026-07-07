#pragma once

#include "ftlpu/hardware_params.hpp"
#include "ftlpu/mxm_array.hpp"
#include "ftlpu/stream.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace ftlpu {

class MxmGemmEngine {
public:
    using ActivationWord = StreamWord<std::int8_t>;
    using ActivationSlot = std::optional<ActivationWord>;
    using ActivationVector = std::array<ActivationSlot, hw::kLanesPerTile>;
    using ActivationData = std::array<std::int8_t, hw::kLanesPerTile>;
    using ResultRow = std::array<std::int32_t, hw::kMxmColumns>;

    struct CompletedRowBlock {
        std::size_t row_block{0};
        std::array<ResultRow, hw::kLanesPerTile> rows{};
    };

    struct ColumnOutput {
        std::size_t target_tile{0};
        std::size_t column_block{0};
        std::array<std::int32_t, hw::kMxmSupercellColumns> values{};
    };

    explicit MxmGemmEngine(const MxmArray& array)
        : array_(array)
    {
        reset();
    }

    void reset()
    {
        cycle_ = 0;
        compute_cycles_ = 0;
        active_ = false;
        for (auto& slot : pending_inputs_) {
            slot.reset();
        }
        for (auto& column : east_pipeline_) {
            for (auto& lane : column) {
                lane.clear();
            }
        }
        for (auto& row : accumulators_) {
            row.fill(0);
        }
        for (auto& row_block : contribution_counts_) {
            row_block.fill(0);
        }
        row_block_completed_columns_.fill(0);
        completed_row_blocks_.clear();
        column_outputs_.clear();
    }

    void start_compute(std::size_t cycles)
    {
        if (cycles == 0) {
            throw std::invalid_argument("GEMM compute requires at least one cycle");
        }
        if (active_) {
            throw std::logic_error("GEMM compute is already active");
        }
        compute_cycles_ = cycles;
        active_ = true;
        cycle_ = 0;
    }

    void set_activation_input(std::size_t tile, ActivationVector input)
    {
        check_tile(tile);
        if (pending_inputs_[tile].has_value()) {
            throw std::logic_error("GEMM tile activation input is already occupied");
        }
        pending_inputs_[tile] = decode_activation(input);
    }

    const std::vector<CompletedRowBlock>& completed_row_blocks() const
    {
        return completed_row_blocks_;
    }

    const std::vector<ColumnOutput>& column_outputs() const
    {
        return column_outputs_;
    }

    const ResultRow& accumulator_row(std::size_t row) const
    {
        if (row >= hw::kMxmRows) {
            throw std::out_of_range("GEMM row is outside the 320-row result");
        }
        return accumulators_[row];
    }

    std::size_t cycle() const
    {
        return cycle_;
    }

    void tick(std::ostream& os)
    {
        os << "mxm_gemm cycle " << cycle_ << '\n';
        completed_row_blocks_.clear();
        column_outputs_.clear();
        std::array<std::array<bool, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane> computing{};

        for (std::size_t tile = 0; tile < hw::kMxmSupercellsPerPlane; ++tile) {
            if (!pending_inputs_[tile].has_value()) {
                continue;
            }
            if (!active_ || cycle_ < tile || cycle_ - tile >= compute_cycles_) {
                throw std::logic_error("GEMM activation valid arrived outside the compute window");
            }

            const auto k = cycle_ - tile;
            east_pipeline_[0][tile].push_back(ActivationEvent {tile, k, *pending_inputs_[tile]});
            os << "  consume activation tile=" << tile << " k=" << k << '\n';
            pending_inputs_[tile].reset();
        }

        std::array<std::array<std::deque<ActivationEvent>, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane>
            next_pipeline{};
        for (std::size_t column_block = 0; column_block < hw::kMxmSupercellsPerPlane; ++column_block) {
            for (std::size_t tile = 0; tile < hw::kMxmSupercellsPerPlane; ++tile) {
                for (const auto& event : east_pipeline_[column_block][tile]) {
                    computing[tile][column_block] = true;
                    compute_column_block(event, column_block, os);
                    if (column_block + 1 < hw::kMxmSupercellsPerPlane) {
                        next_pipeline[column_block + 1][tile].push_back(event);
                    }
                }
            }
        }
        east_pipeline_ = std::move(next_pipeline);
        log_array_state(computing, os);
        ++cycle_;
    }

private:
    struct ActivationEvent {
        std::size_t tile{0};
        std::size_t k{0};
        ActivationData data{};
    };

    static void check_tile(std::size_t tile)
    {
        if (tile >= hw::kMxmSupercellsPerPlane) {
            throw std::out_of_range("GEMM tile is outside the 20-row MXM array");
        }
    }

    static ActivationData decode_activation(const ActivationVector& input)
    {
        ActivationData data{};
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            if (!input[lane].has_value()) {
                throw std::logic_error("GEMM activation stream requires all 16 lanes to be valid");
            }
            data[lane] = input[lane]->data;
        }
        return data;
    }

    void compute_column_block(const ActivationEvent& event, std::size_t column_block, std::ostream& os)
    {
        const auto k_block = event.k / hw::kLanesPerTile;
        const auto k_lane = event.k % hw::kLanesPerTile;
        auto& column_output = output_for_column(column_block);

        os << "  valid tile=" << event.tile
           << " k=" << event.k
           << " col_block=" << column_block
           << " 16MAC";

        for (std::size_t local_column = 0; local_column < hw::kMxmSupercellColumns; ++local_column) {
            const auto global_column = column_block * hw::kMxmSupercellColumns + local_column;
            std::int32_t column_dot = 0;
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                const auto row = event.tile * hw::kLanesPerTile + lane;
                const auto partial = static_cast<std::int32_t>(event.data[lane])
                    * static_cast<std::int32_t>(array_.weight(k_block, column_block, k_lane, local_column));
                column_dot += partial;
                accumulators_[row][global_column] += partial;
                if (lane == 0 && local_column == 0) {
                    os << " psum0+=" << partial;
                }
            }
            column_output.values[local_column] += column_dot;

            auto& count = contribution_counts_[event.tile][global_column];
            ++count;
            if (count == compute_cycles_) {
                ++row_block_completed_columns_[event.tile];
                if (row_block_completed_columns_[event.tile] == hw::kMxmColumns) {
                    CompletedRowBlock block {event.tile, {}};
                    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                        block.rows[lane] = accumulators_[event.tile * hw::kLanesPerTile + lane];
                    }
                    completed_row_blocks_.push_back(block);
                    os << "  north_output row_block=" << event.tile << '\n';
                }
            }
        }
        os << " -> east/north\n";
    }

    ColumnOutput& output_for_column(std::size_t column_block)
    {
        for (auto& output : column_outputs_) {
            if (output.column_block == column_block) {
                return output;
            }
        }
        column_outputs_.push_back(ColumnOutput {column_block, column_block, {}});
        return column_outputs_.back();
    }

    void log_array_state(
        const std::array<std::array<bool, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane>& computing,
        std::ostream& os) const
    {
        os << "  array_state:\n";
        for (std::size_t tile = 0; tile < hw::kMxmSupercellsPerPlane; ++tile) {
            os << "    row " << tile << ": ";
            for (std::size_t column_block = 0; column_block < hw::kMxmSupercellsPerPlane; ++column_block) {
                if (computing[tile][column_block]) {
                    os << 'C';
                } else if (compute_cycles_ != 0 && contribution_counts_[tile][column_block] >= compute_cycles_) {
                    os << '.';
                } else {
                    os << 'L';
                }
            }
            os << '\n';
        }
    }

    const MxmArray& array_;
    std::size_t cycle_{0};
    std::size_t compute_cycles_{0};
    bool active_{false};
    std::array<std::optional<ActivationData>, hw::kMxmSupercellsPerPlane> pending_inputs_{};
    std::array<std::array<std::deque<ActivationEvent>, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane>
        east_pipeline_{};
    std::array<ResultRow, hw::kMxmRows> accumulators_{};
    std::array<std::array<std::size_t, hw::kMxmColumns>, hw::kMxmSupercellsPerPlane> contribution_counts_{};
    std::array<std::size_t, hw::kMxmSupercellsPerPlane> row_block_completed_columns_{};
    std::vector<CompletedRowBlock> completed_row_blocks_{};
    std::vector<ColumnOutput> column_outputs_{};
};

} // namespace ftlpu
