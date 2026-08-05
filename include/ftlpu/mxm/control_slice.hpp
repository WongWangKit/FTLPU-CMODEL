#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/mxm/array.hpp"
#include "ftlpu/mxm/output_cast.hpp"
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
#include <vector>

namespace ftlpu {

enum class MxmControlOpcode {
    IW = 0,
    Compute = 1,
    LoadScales = 2,
    ActivationDequantize = 3,
};

enum class MxmAccumulatorMode : std::uint8_t {
    DirectFinal = 0,
    LocalStart = 1,
    LocalAccumulate = 2,
    LocalFinalize = 3,
    MemoryStart = 4,
    MemoryAccumulate = 5,
    MemoryFinalize = 6,
};

enum class MxmPairMode : std::uint8_t {
    Independent = 0,
    Merge = 1,
};

struct MxmControlInstruction {
    MxmControlOpcode opcode{MxmControlOpcode::IW};
    std::size_t weight_buffer{0};
    MxmWeightLoadMode weight_load_mode{MxmWeightLoadMode::Full};
    std::size_t stream_base{0};
    std::size_t partial_stream_base{0};
    MxmAccumulatorMode accumulator_mode{
        MxmAccumulatorMode::DirectFinal};
    MxmPairMode pair_mode{MxmPairMode::Independent};
    bool start_of_k_block{false};

    static MxmControlInstruction IW(
        std::size_t weight_buffer = 0,
        MxmWeightLoadMode load_mode = MxmWeightLoadMode::Full)
    {
        check_weight_buffer(weight_buffer);
        check_weight_load_mode(load_mode);
        auto result = MxmControlInstruction {};
        result.opcode = MxmControlOpcode::IW;
        result.weight_buffer = weight_buffer;
        result.weight_load_mode = load_mode;
        return result;
    }

    static MxmControlInstruction Compute(
        std::size_t weight_buffer = 0,
        std::size_t stream_base = 0)
    {
        check_weight_buffer(weight_buffer);
        check_result_stream_base(
            stream_base, MxmAccumulatorMode::DirectFinal);
        auto result = MxmControlInstruction {};
        result.opcode = MxmControlOpcode::Compute;
        result.weight_buffer = weight_buffer;
        result.stream_base = stream_base;
        return result;
    }

    static MxmControlInstruction LoadScales(
        std::size_t weight_buffer = 0)
    {
        check_weight_buffer(weight_buffer);
        auto result = MxmControlInstruction {};
        result.opcode = MxmControlOpcode::LoadScales;
        result.weight_buffer = weight_buffer;
        return result;
    }

    static MxmControlInstruction ActivationDequantize()
    {
        auto result = MxmControlInstruction {};
        result.opcode = MxmControlOpcode::ActivationDequantize;
        return result;
    }

    static MxmControlInstruction ComputeAccumulating(
        std::size_t weight_buffer,
        std::size_t stream_base,
        MxmAccumulatorMode accumulator_mode,
        std::size_t partial_stream_base = 0,
        MxmPairMode pair_mode = MxmPairMode::Independent,
        bool start_of_k_block = false)
    {
        check_weight_buffer(weight_buffer);
        check_result_stream_base(stream_base, accumulator_mode);
        check_partial_stream_base(
            partial_stream_base, accumulator_mode);
        auto result = MxmControlInstruction {};
        result.opcode = MxmControlOpcode::Compute;
        result.weight_buffer = weight_buffer;
        result.stream_base = stream_base;
        result.partial_stream_base = partial_stream_base;
        result.accumulator_mode = accumulator_mode;
        result.pair_mode = pair_mode;
        result.start_of_k_block = start_of_k_block;
        return result;
    }

    static void check_weight_buffer(std::size_t weight_buffer)
    {
        if (weight_buffer >= MxmSupercell::kWeightBuffers) {
            throw std::out_of_range("MXM weight buffer is outside the two-buffer set");
        }
    }

    static void check_weight_load_mode(MxmWeightLoadMode mode)
    {
        if (static_cast<std::uint8_t>(mode)
            > static_cast<std::uint8_t>(
                MxmWeightLoadMode::BackgroundUpperHalf)) {
            throw std::out_of_range(
                "MXM weight load mode is invalid");
        }
    }

    static bool uses_memory_partial(MxmAccumulatorMode mode)
    {
        return mode == MxmAccumulatorMode::MemoryAccumulate
            || mode == MxmAccumulatorMode::MemoryFinalize;
    }

    static bool produces_final_output(MxmAccumulatorMode mode)
    {
        return mode == MxmAccumulatorMode::DirectFinal
            || mode == MxmAccumulatorMode::LocalFinalize
            || mode == MxmAccumulatorMode::MemoryFinalize;
    }

    static void check_result_stream_base(
        std::size_t stream_base,
        MxmAccumulatorMode mode)
    {
        const auto bytes = produces_final_output(mode)
            ? MxmOutputCast::kOutputBytes
            : sizeof(std::int32_t);
        if (stream_base % bytes != 0) {
            throw std::invalid_argument(
                "MXM accumulator output stream base is not naturally aligned");
        }
        if (stream_base + bytes
            > hw::kStreamsPerDirection) {
            throw std::out_of_range(
                "MXM accumulator output exceeds one stream direction");
        }
    }

    static void check_partial_stream_base(
        std::size_t stream_base,
        MxmAccumulatorMode mode)
    {
        if (!uses_memory_partial(mode)) {
            return;
        }
        if (stream_base % sizeof(std::int32_t) != 0
            || stream_base + sizeof(std::int32_t)
                > hw::kStreamsPerDirection) {
            throw std::out_of_range(
                "MXM memory partial must use an aligned four-byte int32 stream group");
        }
    }
};

class MxmControlSlice {
public:
    using WeightInput = MxmArray::InputVector;
    using InstructionSlot = std::optional<MxmControlInstruction>;
    using WeightInputSlot = std::optional<WeightInput>;
    using WeightInputProvider = std::function<
        WeightInput(
            std::size_t,
            const MxmControlInstruction&)>;
    static constexpr std::size_t kComputeTileLatency =
        MxmSupercell::kMacPipelineStages;
    static constexpr std::size_t kComputePipelineSlots =
        1 + (hw::kMxmSupercellsPerPlane - 1)
                * kComputeTileLatency;

    struct ComputePulse {
        std::size_t weight_buffer{0};
        std::size_t stream_base{0};
        std::size_t partial_stream_base{0};
        MxmAccumulatorMode accumulator_mode{
            MxmAccumulatorMode::DirectFinal};
        MxmPairMode pair_mode{MxmPairMode::Independent};
        bool start_of_k_block{false};
    };

    explicit MxmControlSlice(MxmArray& array)
        : array_(array)
    {
    }

    void reset()
    {
        load_instruction_queue_.clear();
        compute_instruction_queue_.clear();
        for (auto& slot : load_instruction_rows_) {
            slot.reset();
        }
        for (auto& slot : compute_instruction_pipeline_) {
            slot.reset();
        }
        for (auto& slot : weight_inputs_) {
            slot.reset();
        }
        for (auto& slot : weight_pipeline_) {
            slot.reset();
        }
        compute_pulses_.fill(false);
        for (auto& pulse : compute_pulse_details_) {
            pulse.reset();
        }
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
        if (instruction.opcode == MxmControlOpcode::Compute) {
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
        return compute_instruction_pipeline_[
            tile * kComputeTileLatency];
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
        return compute_pulse_details_[tile].has_value()
            ? std::optional<std::size_t> {compute_pulse_details_[tile]->stream_base}
            : std::nullopt;
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
        return compute_pulse_details_[tile].has_value()
            ? std::optional<std::size_t> {compute_pulse_details_[tile]->weight_buffer}
            : std::nullopt;
    }

    std::optional<ComputePulse> compute_pulse(std::size_t tile) const
    {
        check_tile(tile);
        return compute_pulse_details_[tile];
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
        fill_missing_weight_inputs(weight_provider);
        os << "mxm_control cycle " << cycle_ << '\n';
        execute(os, print_matrix, log_tile);
        advance();
        ++cycle_;
    }

private:
    struct WeightToken {
        MxmOpcode opcode{MxmOpcode::IW};
        std::size_t weight_buffer{0};
        MxmWeightLoadMode weight_load_mode{
            MxmWeightLoadMode::Full};
        WeightInput input{};
    };

    using WeightTokenSlot = std::optional<WeightToken>;

    static void check_tile(std::size_t tile)
    {
        if (tile >= hw::kMxmSupercellsPerPlane) {
            throw std::out_of_range(
                "MXM control tile exceeds the configured Supercell-plane rows");
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
        MxmControlInstruction::check_weight_buffer(instruction.weight_buffer);
        MxmControlInstruction::check_weight_load_mode(
            instruction.weight_load_mode);
        if (instruction.opcode == MxmControlOpcode::Compute) {
            MxmControlInstruction::check_result_stream_base(
                instruction.stream_base,
                instruction.accumulator_mode);
            MxmControlInstruction::check_partial_stream_base(
                instruction.partial_stream_base,
                instruction.accumulator_mode);
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
        if (compute_instruction_pipeline_[0].has_value()
            || compute_instruction_queue_.empty()) {
            return;
        }
        compute_instruction_pipeline_[0] =
            compute_instruction_queue_.front();
        compute_instruction_queue_.pop_front();
    }

    void fill_missing_weight_inputs(const WeightInputProvider& weight_provider)
    {
        if (!weight_provider) {
            return;
        }

        for (std::size_t tile = 0; tile < hw::kMxmSupercellsPerPlane; ++tile) {
            const auto& instruction = load_instruction_rows_[tile];
            if (!instruction.has_value()
                || instruction->opcode
                    == MxmControlOpcode::Compute
                || weight_inputs_[tile].has_value()) {
                continue;
            }
            weight_inputs_[tile] =
                weight_provider(tile, *instruction);
        }
    }

    void execute(std::ostream& os, bool print_matrix, std::optional<std::size_t> log_tile)
    {
        compute_pulses_.fill(false);
        for (auto& pulse : compute_pulse_details_) {
            pulse.reset();
        }
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
                if (instruction->opcode
                    != MxmControlOpcode::Compute) {
                    if (!weight_inputs_[tile].has_value()) {
                        throw std::logic_error(
                            "MXM weight-control instruction reached tile without local input");
                    }

                    const auto opcode =
                        instruction->opcode
                                == MxmControlOpcode::IW
                        ? MxmOpcode::IW
                        : MxmOpcode::LoadScales;
                    if (should_log) {
                        os << (opcode == MxmOpcode::IW
                                  ? "IW b"
                                  : "LoadScales b")
                           << instruction->weight_buffer
                           << " inject ";
                    }
                    const auto load_mode = opcode == MxmOpcode::IW
                        ? instruction->weight_load_mode
                        : MxmWeightLoadMode::Full;
                    for (std::size_t column = hw::kMxmSupercellsPerPlane - 1; column > 0; --column) {
                        load_token(
                            opcode,
                            instruction->weight_buffer,
                            load_mode,
                            tile,
                            column)
                            = load_token(
                                opcode,
                                instruction->weight_buffer,
                                load_mode,
                                tile,
                                column - 1);
                    }
                    load_token(
                        opcode,
                        instruction->weight_buffer,
                        load_mode,
                        tile,
                        0)
                        = WeightToken {
                            opcode,
                            instruction->weight_buffer,
                            instruction->weight_load_mode,
                            *weight_inputs_[tile]};
                    weight_inputs_[tile].reset();

                    for (std::size_t column = 0; column < hw::kMxmSupercellsPerPlane; ++column) {
                        const auto& token = load_token(
                            opcode,
                            instruction->weight_buffer,
                            load_mode,
                            tile,
                            column);
                        if (!token.has_value()) {
                            continue;
                        }
                        any = true;
                        if (should_log) {
                            any_logged = true;
                            os << "  tile " << tile << " weight b" << token->weight_buffer << " col=" << column << " ";
                            array_.tick_cell_load(
                                tile,
                                column,
                                MxmInstruction {
                                    token->opcode,
                                    token->weight_buffer,
                                    token->weight_load_mode},
                                token->input,
                                os);
                        } else {
                            static NullStream null_stream;
                            array_.tick_cell_load(
                                tile,
                                column,
                                MxmInstruction {
                                    token->opcode,
                                    token->weight_buffer,
                                    token->weight_load_mode},
                                token->input,
                                null_stream.stream());
                        }
                        if (token->opcode == MxmOpcode::IW) {
                            loaded_cells_[
                                token->weight_buffer]
                                [tile][column] =
                                array_.cell(tile, column)
                                    .weight_buffer_valid(
                                        token->weight_buffer);
                        }
                    }
                }
            }

            const auto& compute_instruction =
                compute_instruction_pipeline_[
                    tile * kComputeTileLatency];
            if (compute_instruction.has_value()) {
                any = true;
                if (!log_tile.has_value() || tile == *log_tile) {
                    any_logged = true;
                    os << "  tile " << tile << " Compute b" << compute_instruction->weight_buffer
                       << " out=" << compute_instruction->stream_base
                       << " acc_mode="
                       << static_cast<int>(compute_instruction->accumulator_mode)
                       << " pair="
                       << static_cast<int>(compute_instruction->pair_mode)
                       << '\n';
                }
                compute_pulses_[tile] = true;
                compute_pulse_details_[tile] = ComputePulse {
                    compute_instruction->weight_buffer,
                    compute_instruction->stream_base,
                    compute_instruction->partial_stream_base,
                    compute_instruction->accumulator_mode,
                    compute_instruction->pair_mode,
                    compute_instruction->start_of_k_block,
                };
            }
        }

        if (!any || (log_tile.has_value() && !any_logged)) {
            os << "  idle\n";
        }
        if (print_matrix) {
            print_load_matrix(os, loaded_cells_, log_tile);
        }
    }

    static constexpr std::size_t load_pipeline_kind(
        MxmOpcode opcode,
        MxmWeightLoadMode load_mode)
    {
        if (opcode == MxmOpcode::LoadScales) {
            return 3;
        }
        return static_cast<std::size_t>(load_mode);
    }

    static constexpr std::size_t load_pipeline_index(
        MxmOpcode opcode,
        std::size_t weight_buffer,
        MxmWeightLoadMode load_mode,
        std::size_t tile,
        std::size_t column)
    {
        return ((load_pipeline_kind(opcode, load_mode)
                     * MxmSupercell::kWeightBuffers
                 + weight_buffer)
                    * hw::kMxmSupercellsPerPlane
                + tile)
                * hw::kMxmSupercellsPerPlane
            + column;
    }

    WeightTokenSlot& load_token(
        MxmOpcode opcode,
        std::size_t weight_buffer,
        MxmWeightLoadMode load_mode,
        std::size_t tile,
        std::size_t column)
    {
        return weight_pipeline_[
            load_pipeline_index(
                opcode,
                weight_buffer,
                load_mode,
                tile,
                column)];
    }

    const WeightTokenSlot& load_token(
        MxmOpcode opcode,
        std::size_t weight_buffer,
        MxmWeightLoadMode load_mode,
        std::size_t tile,
        std::size_t column) const
    {
        return weight_pipeline_[
            load_pipeline_index(
                opcode,
                weight_buffer,
                load_mode,
                tile,
                column)];
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
        }
        for (std::size_t slot = kComputePipelineSlots - 1;
             slot > 0;
             --slot) {
            compute_instruction_pipeline_[slot] =
                compute_instruction_pipeline_[slot - 1];
        }
        load_instruction_rows_[0].reset();
        compute_instruction_pipeline_[0].reset();
    }

    MxmArray& array_;
    std::deque<MxmControlInstruction> load_instruction_queue_{};
    std::deque<MxmControlInstruction> compute_instruction_queue_{};
    std::array<InstructionSlot, hw::kMxmSupercellsPerPlane> load_instruction_rows_{};
    std::array<InstructionSlot, kComputePipelineSlots>
        compute_instruction_pipeline_{};
    std::array<WeightInputSlot, hw::kMxmSupercellsPerPlane> weight_inputs_{};
    std::vector<WeightTokenSlot> weight_pipeline_ = std::vector<WeightTokenSlot>(
        4 * MxmSupercell::kWeightBuffers
        * hw::kMxmSupercellsPerPlane
        * hw::kMxmSupercellsPerPlane);
    std::array<bool, hw::kMxmSupercellsPerPlane> compute_pulses_{};
    std::array<std::optional<ComputePulse>, hw::kMxmSupercellsPerPlane> compute_pulse_details_{};
    std::array<
        std::array<std::array<bool, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane>,
        MxmSupercell::kWeightBuffers> loaded_cells_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
