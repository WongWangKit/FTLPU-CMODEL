#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/mxm/array.hpp"
#include "ftlpu/mxm/supercell.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <iomanip>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <streambuf>

namespace ftlpu {

enum class MxmControlOpcode {
    IW = 0,
    Compute = 1,
    Output = 2,
};

struct MxmControlInstruction {
    MxmControlOpcode opcode{MxmControlOpcode::IW};
    std::size_t supercell_column{0};
    std::size_t weight_buffer{0};
    std::size_t stream_base{0};
    std::size_t activation_stream_base{0};

    static MxmControlInstruction IW(std::size_t supercell_column, std::size_t weight_buffer = 0)
    {
        check_weight_buffer(weight_buffer);
        return MxmControlInstruction {MxmControlOpcode::IW, supercell_column, weight_buffer, 0, 0};
    }

    static MxmControlInstruction Compute(std::size_t weight_buffer = 0, std::size_t activation_stream_base = 0)
    {
        check_weight_buffer(weight_buffer);
        check_activation_stream_base(activation_stream_base);
        return MxmControlInstruction {MxmControlOpcode::Compute, 0, weight_buffer, 0, activation_stream_base};
    }

    static MxmControlInstruction Output(std::size_t stream_base)
    {
        return MxmControlInstruction {MxmControlOpcode::Output, 0, 0, stream_base, 0};
    }

    static void check_weight_buffer(std::size_t weight_buffer)
    {
        if (weight_buffer >= MxmSupercell::kWeightBuffers) {
            throw std::out_of_range("MXM weight buffer is outside the two-buffer set");
        }
    }

    static void check_activation_stream_base(std::size_t activation_stream_base)
    {
        if (activation_stream_base >= hw::kStreams) {
            throw std::out_of_range("MXM activation stream base is outside the stream set");
        }
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
        load_instruction_queue_.clear();
        compute_instruction_queue_.clear();
        output_instruction_queue_.clear();
        for (auto& slot : load_instruction_rows_) {
            slot.reset();
        }
        for (auto& slot : compute_instruction_rows_) {
            slot.reset();
        }
        for (auto& slot : output_instruction_rows_) {
            slot.reset();
        }
        for (auto& slot : weight_inputs_) {
            slot.reset();
        }
        compute_pulses_.fill(false);
        compute_weight_buffers_.fill(std::nullopt);
        compute_activation_stream_bases_.fill(std::nullopt);
        output_stream_bases_.fill(std::nullopt);
        for (auto& buffer : loaded_cells_) {
            for (auto& row : buffer) {
                row.fill(false);
            }
        }
        cycle_ = 0;
    }

    std::size_t cycle() const
    {
        return cycle_;
    }

    void issue_south(MxmControlInstruction instruction)
    {
        check_instruction(instruction);
        if (instruction.opcode == MxmControlOpcode::Output) {
            output_instruction_queue_.push_back(instruction);
        } else if (instruction.opcode == MxmControlOpcode::Compute) {
            compute_instruction_queue_.push_back(instruction);
        } else {
            load_instruction_queue_.push_back(instruction);
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
        return load_instruction_rows_[tile];
    }

    const InstructionSlot& compute_instruction_at(std::size_t tile) const
    {
        check_tile(tile);
        return compute_instruction_rows_[tile];
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
        return loaded_cell(0, tile, column);
    }

    bool loaded_cell(std::size_t weight_buffer, std::size_t tile, std::size_t column) const
    {
        check_tile(tile);
        check_column(column);
        MxmControlInstruction::check_weight_buffer(weight_buffer);
        return loaded_cells_[weight_buffer][tile][column];
    }

    std::optional<std::size_t> compute_weight_buffer(std::size_t tile) const
    {
        check_tile(tile);
        return compute_weight_buffers_[tile];
    }

    std::optional<std::size_t> compute_activation_stream_base(std::size_t tile) const
    {
        check_tile(tile);
        return compute_activation_stream_bases_[tile];
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
        dispatch_load_instruction();
        dispatch_compute_instruction();
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

    static void check_instruction(const MxmControlInstruction& instruction)
    {
        if (instruction.opcode == MxmControlOpcode::IW) {
            check_column(instruction.supercell_column);
        }
        MxmControlInstruction::check_weight_buffer(instruction.weight_buffer);
        if (instruction.opcode == MxmControlOpcode::Compute) {
            MxmControlInstruction::check_activation_stream_base(instruction.activation_stream_base);
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

    void dispatch_load_instruction()
    {
        if (load_instruction_rows_[0].has_value() || load_instruction_queue_.empty()) {
            return;
        }
        load_instruction_rows_[0] = load_instruction_queue_.front();
        load_instruction_queue_.pop_front();
    }

    void dispatch_compute_instruction()
    {
        if (compute_instruction_rows_[0].has_value() || compute_instruction_queue_.empty()) {
            return;
        }
        compute_instruction_rows_[0] = compute_instruction_queue_.front();
        compute_instruction_queue_.pop_front();
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
            const auto& instruction = load_instruction_rows_[tile];
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
        compute_weight_buffers_.fill(std::nullopt);
        compute_activation_stream_bases_.fill(std::nullopt);
        output_stream_bases_.fill(std::nullopt);
        bool any = false;
        bool any_logged = false;
        for (std::size_t tile = 0; tile < hw::kMxmSupercellsPerPlane; ++tile) {
            const auto& instruction = load_instruction_rows_[tile];
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
                        os << "IW b" << instruction->weight_buffer << " col=" << instruction->supercell_column << " ";
                        array_.tick_cell_iw_load(
                            tile,
                            instruction->supercell_column,
                            instruction->weight_buffer,
                            *weight_inputs_[tile],
                            os);
                    } else {
                        static NullStream null_stream;
                        array_.tick_cell_iw_load(
                            tile,
                            instruction->supercell_column,
                            instruction->weight_buffer,
                            *weight_inputs_[tile],
                            null_stream.stream());
                    }
                    loaded_cells_[instruction->weight_buffer][tile][instruction->supercell_column] = true;
                    weight_inputs_[tile].reset();
                }
            }

            const auto& compute_instruction = compute_instruction_rows_[tile];
            if (compute_instruction.has_value()) {
                any = true;
                if (!log_tile.has_value() || tile == *log_tile) {
                    any_logged = true;
                    os << "  tile " << tile << " Compute b" << compute_instruction->weight_buffer
                       << " stream=" << compute_instruction->activation_stream_base << '\n';
                }
                compute_pulses_[tile] = true;
                compute_weight_buffers_[tile] = compute_instruction->weight_buffer;
                compute_activation_stream_bases_[tile] = compute_instruction->activation_stream_base;
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
        const std::array<
            std::array<std::array<bool, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane>,
            MxmSupercell::kWeightBuffers>& loaded,
        std::optional<std::size_t> log_tile)
    {
        os << "  load_matrix:\n";
        const auto first_tile = log_tile.value_or(0);
        const auto end_tile = log_tile.has_value() ? first_tile + 1 : hw::kMxmSupercellsPerPlane;
        for (std::size_t buffer = 0; buffer < MxmSupercell::kWeightBuffers; ++buffer) {
            os << "    buffer " << buffer << ":\n";
            for (std::size_t tile = first_tile; tile < end_tile; ++tile) {
                os << "      row " << tile << ": ";
                for (std::size_t column = 0; column < hw::kMxmSupercellsPerPlane; ++column) {
                    os << (loaded[buffer][tile][column] ? 'L' : '.');
                }
                os << '\n';
            }
        }
    }

    void advance()
    {
        for (std::size_t tile = hw::kMxmSupercellsPerPlane - 1; tile > 0; --tile) {
            load_instruction_rows_[tile] = load_instruction_rows_[tile - 1];
            compute_instruction_rows_[tile] = compute_instruction_rows_[tile - 1];
            output_instruction_rows_[tile] = output_instruction_rows_[tile - 1];
        }
        load_instruction_rows_[0].reset();
        compute_instruction_rows_[0].reset();
        output_instruction_rows_[0].reset();
    }

    MxmArray& array_;
    std::deque<MxmControlInstruction> load_instruction_queue_{};
    std::deque<MxmControlInstruction> compute_instruction_queue_{};
    std::deque<MxmControlInstruction> output_instruction_queue_{};
    std::array<InstructionSlot, hw::kMxmSupercellsPerPlane> load_instruction_rows_{};
    std::array<InstructionSlot, hw::kMxmSupercellsPerPlane> compute_instruction_rows_{};
    std::array<InstructionSlot, hw::kMxmSupercellsPerPlane> output_instruction_rows_{};
    std::array<WeightInputSlot, hw::kMxmSupercellsPerPlane> weight_inputs_{};
    std::array<bool, hw::kMxmSupercellsPerPlane> compute_pulses_{};
    std::array<std::optional<std::size_t>, hw::kMxmSupercellsPerPlane> compute_weight_buffers_{};
    std::array<std::optional<std::size_t>, hw::kMxmSupercellsPerPlane> compute_activation_stream_bases_{};
    std::array<std::optional<std::size_t>, hw::kMxmSupercellsPerPlane> output_stream_bases_{};
    std::array<
        std::array<std::array<bool, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane>,
        MxmSupercell::kWeightBuffers> loaded_cells_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
