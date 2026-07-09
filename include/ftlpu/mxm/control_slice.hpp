#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/mxm/array.hpp"
#include "ftlpu/mxm/supercell.hpp"

#include <array>
#include <cstddef>
#include <deque>
#include <functional>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <streambuf>

namespace ftlpu {

enum class MxmControlOpcode {
    IW,
    Compute,
    Output,
};

struct MxmControlInstruction {
    MxmControlOpcode opcode{MxmControlOpcode::IW};
    std::size_t supercell_column{0};
    std::size_t stream_base{0};

    static MxmControlInstruction IW(std::size_t supercell_column)
    {
        return MxmControlInstruction {MxmControlOpcode::IW, supercell_column, 0};
    }

    static MxmControlInstruction Compute()
    {
        return MxmControlInstruction {MxmControlOpcode::Compute, 0, 0};
    }

    static MxmControlInstruction Output(std::size_t stream_base)
    {
        return MxmControlInstruction {MxmControlOpcode::Output, 0, stream_base};
    }
};

class MxmControlSlice {
public:
    using WeightInput = MxmArray::InputVector;
    using InstructionSlot = std::optional<MxmControlInstruction>;
    using WeightInputSlot = std::optional<WeightInput>;
    using WeightInputProvider = std::function<WeightInput(std::size_t)>;

    explicit MxmControlSlice(MxmArray& array)
        : array_(array)
    {
    }

    void reset()
    {
        instruction_queue_.clear();
        output_instruction_queue_.clear();
        for (auto& slot : instruction_rows_) {
            slot.reset();
        }
        for (auto& slot : output_instruction_rows_) {
            slot.reset();
        }
        for (auto& slot : weight_inputs_) {
            slot.reset();
        }
        compute_pulses_.fill(false);
        output_stream_bases_.fill(std::nullopt);
        for (auto& row : loaded_cells_) {
            row.fill(false);
        }
        cycle_ = 0;
    }

    std::size_t cycle() const
    {
        return cycle_;
    }

    void issue_south(MxmControlInstruction instruction)
    {
        check_column(instruction.supercell_column);
        if (instruction.opcode == MxmControlOpcode::Output) {
            output_instruction_queue_.push_back(instruction);
        } else {
            instruction_queue_.push_back(instruction);
        }
    }

    void set_weight_input(std::size_t tile, WeightInput input)
    {
        check_tile(tile);
        if (weight_inputs_[tile].has_value()) {
            throw std::logic_error("MXM tile weight input is occupied");
        }
        weight_inputs_[tile] = input;
    }

    const InstructionSlot& instruction_at(std::size_t tile) const
    {
        check_tile(tile);
        return instruction_rows_[tile];
    }

    const WeightInputSlot& weight_input_at(std::size_t tile) const
    {
        check_tile(tile);
        return weight_inputs_[tile];
    }

    bool compute_active(std::size_t tile) const
    {
        check_tile(tile);
        return compute_pulses_[tile];
    }

    std::optional<std::size_t> output_stream_base(std::size_t tile) const
    {
        check_tile(tile);
        return output_stream_bases_[tile];
    }

    bool loaded_cell(std::size_t tile, std::size_t column) const
    {
        check_tile(tile);
        check_column(column);
        return loaded_cells_[tile][column];
    }

    void tick(std::ostream& os, bool print_matrix = true, std::optional<std::size_t> log_tile = std::nullopt)
    {
        tick(os, nullptr, print_matrix, log_tile);
    }

    void tick(
        std::ostream& os,
        const WeightInputProvider& weight_provider,
        bool print_matrix = true,
        std::optional<std::size_t> log_tile = std::nullopt)
    {
        if (log_tile.has_value()) {
            check_tile(*log_tile);
        }
        dispatch_instruction();
        dispatch_output_instruction();
        fill_missing_weight_inputs(weight_provider);
        os << "mxm_control cycle " << cycle_ << '\n';
        execute(os, print_matrix, log_tile);
        advance();
        ++cycle_;
    }

private:
    static void check_tile(std::size_t tile)
    {
        if (tile >= hw::kMxmSupercellsPerPlane) {
            throw std::out_of_range("MXM control tile is outside the 20-row array");
        }
    }

    static void check_column(std::size_t column)
    {
        if (column >= hw::kMxmSupercellsPerPlane) {
            throw std::out_of_range("MXM control supercell column is outside the 20-column array");
        }
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

    void dispatch_instruction()
    {
        if (instruction_rows_[0].has_value() || instruction_queue_.empty()) {
            return;
        }
        instruction_rows_[0] = instruction_queue_.front();
        instruction_queue_.pop_front();
    }

    void dispatch_output_instruction()
    {
        if (output_instruction_rows_[0].has_value() || output_instruction_queue_.empty()) {
            return;
        }
        output_instruction_rows_[0] = output_instruction_queue_.front();
        output_instruction_queue_.pop_front();
    }

    void fill_missing_weight_inputs(const WeightInputProvider& weight_provider)
    {
        if (!weight_provider) {
            return;
        }

        for (std::size_t tile = 0; tile < hw::kMxmSupercellsPerPlane; ++tile) {
            const auto& instruction = instruction_rows_[tile];
            if (!instruction.has_value()
                || instruction->opcode != MxmControlOpcode::IW
                || weight_inputs_[tile].has_value()) {
                continue;
            }
            weight_inputs_[tile] = weight_provider(tile);
        }
    }

    void execute(std::ostream& os, bool print_matrix, std::optional<std::size_t> log_tile)
    {
        compute_pulses_.fill(false);
        output_stream_bases_.fill(std::nullopt);
        bool any = false;
        bool any_logged = false;
        for (std::size_t tile = 0; tile < hw::kMxmSupercellsPerPlane; ++tile) {
            const auto& instruction = instruction_rows_[tile];
            if (instruction.has_value()) {
                any = true;
                const auto should_log = !log_tile.has_value() || tile == *log_tile;
                if (should_log) {
                    any_logged = true;
                    os << "  tile " << tile << " ";
                }
                if (instruction->opcode == MxmControlOpcode::IW) {
                    if (!weight_inputs_[tile].has_value()) {
                        throw std::logic_error("MXM IW reached tile without local weight input");
                    }

                    if (should_log) {
                        os << "IW col=" << instruction->supercell_column << " ";
                        array_.tick_cell_iw_load(tile, instruction->supercell_column, *weight_inputs_[tile], os);
                    } else {
                        static NullStream null_stream;
                        array_.tick_cell_iw_load(
                            tile,
                            instruction->supercell_column,
                            *weight_inputs_[tile],
                            null_stream.stream());
                    }
                    loaded_cells_[tile][instruction->supercell_column] = true;
                    weight_inputs_[tile].reset();
                } else if (instruction->opcode == MxmControlOpcode::Compute) {
                    if (should_log) {
                        os << "Compute\n";
                    }
                    compute_pulses_[tile] = true;
                }
            }

            const auto& output_instruction = output_instruction_rows_[tile];
            if (output_instruction.has_value()) {
                any = true;
                if (!log_tile.has_value() || tile == *log_tile) {
                    any_logged = true;
                    os << "  tile " << tile << " Output stream_base=" << output_instruction->stream_base << '\n';
                }
                output_stream_bases_[tile] = output_instruction->stream_base;
            }
        }

        if (!any || (log_tile.has_value() && !any_logged)) {
            os << "  idle\n";
        }
        if (print_matrix) {
            print_load_matrix(os, loaded_cells_, log_tile);
        }
    }

    static void print_load_matrix(
        std::ostream& os,
        const std::array<std::array<bool, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane>& loaded,
        std::optional<std::size_t> log_tile)
    {
        os << "  load_matrix:\n";
        const auto first_tile = log_tile.value_or(0);
        const auto end_tile = log_tile.has_value() ? first_tile + 1 : hw::kMxmSupercellsPerPlane;
        for (std::size_t tile = first_tile; tile < end_tile; ++tile) {
            os << "    row " << tile << ": ";
            for (std::size_t column = 0; column < hw::kMxmSupercellsPerPlane; ++column) {
                os << (loaded[tile][column] ? 'L' : '.');
            }
            os << '\n';
        }
    }

    void advance()
    {
        for (std::size_t tile = hw::kMxmSupercellsPerPlane - 1; tile > 0; --tile) {
            instruction_rows_[tile] = instruction_rows_[tile - 1];
            output_instruction_rows_[tile] = output_instruction_rows_[tile - 1];
        }
        instruction_rows_[0].reset();
        output_instruction_rows_[0].reset();
    }

    MxmArray& array_;
    std::deque<MxmControlInstruction> instruction_queue_{};
    std::deque<MxmControlInstruction> output_instruction_queue_{};
    std::array<InstructionSlot, hw::kMxmSupercellsPerPlane> instruction_rows_{};
    std::array<InstructionSlot, hw::kMxmSupercellsPerPlane> output_instruction_rows_{};
    std::array<WeightInputSlot, hw::kMxmSupercellsPerPlane> weight_inputs_{};
    std::array<bool, hw::kMxmSupercellsPerPlane> compute_pulses_{};
    std::array<std::optional<std::size_t>, hw::kMxmSupercellsPerPlane> output_stream_bases_{};
    std::array<std::array<bool, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane> loaded_cells_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
