#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/mxm/array.hpp"
#include "ftlpu/mxm/supercell.hpp"

#include <array>
#include <cstddef>
#include <deque>
#include <optional>
#include <ostream>
#include <stdexcept>

namespace ftlpu {

enum class MxmControlOpcode {
    IW,
    Compute,
};

struct MxmControlInstruction {
    MxmControlOpcode opcode{MxmControlOpcode::IW};
    std::size_t supercell_column{0};
    std::size_t cycles{0};

    static MxmControlInstruction IW(std::size_t supercell_column)
    {
        return MxmControlInstruction {MxmControlOpcode::IW, supercell_column, 0};
    }

    static MxmControlInstruction Compute(std::size_t cycles)
    {
        if (cycles == 0) {
            throw std::invalid_argument("MXM compute instruction requires at least one cycle");
        }
        return MxmControlInstruction {MxmControlOpcode::Compute, 0, cycles};
    }
};

class MxmControlSlice {
public:
    using WeightInput = MxmArray::InputVector;
    using InstructionSlot = std::optional<MxmControlInstruction>;
    using WeightInputSlot = std::optional<WeightInput>;

    explicit MxmControlSlice(MxmArray& array)
        : array_(array)
    {
    }

    void reset()
    {
        instruction_queue_.clear();
        for (auto& slot : instruction_rows_) {
            slot.reset();
        }
        for (auto& slot : weight_inputs_) {
            slot.reset();
        }
        compute_windows_.fill(0);
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
        instruction_queue_.push_back(instruction);
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
        return compute_windows_[tile] != 0;
    }

    std::size_t compute_cycles_remaining(std::size_t tile) const
    {
        check_tile(tile);
        return compute_windows_[tile];
    }

    bool loaded_cell(std::size_t tile, std::size_t column) const
    {
        check_tile(tile);
        check_column(column);
        return loaded_cells_[tile][column];
    }

    void tick(std::ostream& os, bool print_matrix = true)
    {
        dispatch_instruction();
        os << "mxm_control cycle " << cycle_ << '\n';
        execute(os, print_matrix);
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

    void dispatch_instruction()
    {
        if (instruction_rows_[0].has_value() || instruction_queue_.empty()) {
            return;
        }
        instruction_rows_[0] = instruction_queue_.front();
        instruction_queue_.pop_front();
    }

    void execute(std::ostream& os, bool print_matrix)
    {
        bool any = false;
        for (std::size_t tile = 0; tile < hw::kMxmSupercellsPerPlane; ++tile) {
            const auto& instruction = instruction_rows_[tile];
            if (!instruction.has_value()) {
                continue;
            }

            any = true;
            os << "  tile " << tile << " ";
            if (instruction->opcode == MxmControlOpcode::IW) {
                if (!weight_inputs_[tile].has_value()) {
                    throw std::logic_error("MXM IW reached tile without local weight input");
                }

                os << "IW col=" << instruction->supercell_column << " ";
                array_.tick_cell_iw_load(tile, instruction->supercell_column, *weight_inputs_[tile], os);
                loaded_cells_[tile][instruction->supercell_column] = true;
                weight_inputs_[tile].reset();
            } else {
                os << "Compute cycles=" << instruction->cycles << '\n';
                compute_windows_[tile] = instruction->cycles;
            }
        }

        if (!any) {
            os << "  idle\n";
        }
        if (print_matrix) {
            print_load_matrix(os, loaded_cells_);
        }
    }

    static void print_load_matrix(
        std::ostream& os,
        const std::array<std::array<bool, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane>& loaded)
    {
        os << "  load_matrix:\n";
        for (std::size_t tile = 0; tile < hw::kMxmSupercellsPerPlane; ++tile) {
            os << "    row " << tile << ": ";
            for (std::size_t column = 0; column < hw::kMxmSupercellsPerPlane; ++column) {
                os << (loaded[tile][column] ? 'L' : '.');
            }
            os << '\n';
        }
    }

    void advance()
    {
        for (auto& cycles : compute_windows_) {
            if (cycles != 0) {
                --cycles;
            }
        }
        for (std::size_t tile = hw::kMxmSupercellsPerPlane - 1; tile > 0; --tile) {
            instruction_rows_[tile] = instruction_rows_[tile - 1];
        }
        instruction_rows_[0].reset();
    }

    MxmArray& array_;
    std::deque<MxmControlInstruction> instruction_queue_{};
    std::array<InstructionSlot, hw::kMxmSupercellsPerPlane> instruction_rows_{};
    std::array<WeightInputSlot, hw::kMxmSupercellsPerPlane> weight_inputs_{};
    std::array<std::size_t, hw::kMxmSupercellsPerPlane> compute_windows_{};
    std::array<std::array<bool, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane> loaded_cells_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
