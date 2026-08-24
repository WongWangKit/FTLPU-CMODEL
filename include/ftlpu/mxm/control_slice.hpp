#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/mxm/array.hpp"
#include "ftlpu/mxm/supercell.hpp"
#include "ftlpu/mxm/weight_dequantizer.hpp"

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
#include <string>

namespace ftlpu {

enum class MxmControlOpcode {
    IW = 0,
    Compute = 1,
    AccumulatorRead = 2,
    Decode = 3,
};

enum class MxmDecodeOperation {
    LoadActivation = 0,
    StreamCompute = 1,
};

enum class MxmDecodeLayout {
    Linear1x16 = 0,
    Native4x4 = 1,
};

enum class MxmAccumulatorDestination {
    Sram = 0,
    Stream = 1,
};

enum class MxmAccumulatorOutputFormat {
    Float32 = 0,
    BFloat16 = 1,
};

enum class MxmComputeMode {
    Vector = 0,
    Block8 = 1,
};

struct MxmControlInstruction {
    MxmControlOpcode opcode{MxmControlOpcode::IW};
    std::size_t weight_buffer{0};
    std::size_t weight_stream_base{0};
    std::size_t stream_base{0};
    std::size_t activation_stream_base{0};
    std::size_t weight_column{0};
    std::size_t accumulator_address{0};
    std::size_t accumulator_row_stride{1};
    MxmAccumulatorDestination accumulator_destination{
        MxmAccumulatorDestination::Stream};
    bool accumulator_clear{true};
    MxmWeightLoadMode weight_load_mode{MxmWeightLoadMode::Supercell};
    MxmWeightInputMode weight_input_mode{
        MxmWeightInputMode::Int8DequantBf16};
    std::size_t weight_inner_column{0};
    MxmDataFormat data_format{MxmDataFormat::Float16};
    MxmComputeMode compute_mode{MxmComputeMode::Vector};
    MxmAccumulatorOutputFormat accumulator_output_format{
        MxmAccumulatorOutputFormat::Float32};
    MxmDecodeOperation decode_operation{MxmDecodeOperation::LoadActivation};
    MxmDecodeLayout decode_layout{MxmDecodeLayout::Linear1x16};

    static MxmControlInstruction IW(
        std::size_t weight_buffer = 0,
        std::size_t weight_column = 0,
        MxmWeightInputMode weight_input_mode =
            MxmWeightInputMode::Int8DequantBf16,
        std::size_t weight_stream_base = 0)
    {
        check_weight_buffer(weight_buffer);
        check_column(weight_column);
        check_weight_stream_base(weight_stream_base, weight_input_mode,
            MxmWeightLoadMode::Supercell);
        auto instruction = MxmControlInstruction {};
        instruction.opcode = MxmControlOpcode::IW;
        instruction.weight_buffer = weight_buffer;
        instruction.weight_column = weight_column;
        instruction.weight_input_mode = weight_input_mode;
        instruction.weight_stream_base = weight_stream_base;
        return instruction;
    }

    static MxmControlInstruction IWDirect16(
        std::size_t weight_buffer = 0,
        std::size_t weight_column = 0,
        std::size_t weight_stream_base = 0)
    {
        return IW(
            weight_buffer,
            weight_column,
            MxmWeightInputMode::Direct16,
            weight_stream_base);
    }

    static MxmControlInstruction IWColumn(
        std::size_t weight_buffer,
        std::size_t weight_column,
        std::size_t weight_inner_column,
        MxmWeightInputMode weight_input_mode =
            MxmWeightInputMode::Int8DequantBf16,
        std::size_t weight_stream_base = 0)
    {
        check_weight_buffer(weight_buffer);
        check_column(weight_column);
        check_inner_column(weight_inner_column);
        auto instruction = IW(
            weight_buffer,
            weight_column,
            weight_input_mode);
        instruction.weight_load_mode = MxmWeightLoadMode::Column;
        instruction.weight_inner_column = weight_inner_column;
        instruction.weight_stream_base = weight_stream_base;
        check_weight_stream_base(weight_stream_base, weight_input_mode,
            MxmWeightLoadMode::Column);
        return instruction;
    }

    static MxmControlInstruction IWColumnDirect16(
        std::size_t weight_buffer,
        std::size_t weight_column,
        std::size_t weight_inner_column,
        std::size_t weight_stream_base = 0)
    {
        return IWColumn(
            weight_buffer,
            weight_column,
            weight_inner_column,
            MxmWeightInputMode::Direct16,
            weight_stream_base);
    }

    static MxmControlInstruction Compute(
        std::size_t weight_buffer = 0,
        std::size_t activation_stream_base = 0,
        std::size_t stream_base = 0,
        std::size_t accumulator_address = 0,
        std::size_t accumulator_row_stride = 1,
        MxmAccumulatorDestination accumulator_destination =
            MxmAccumulatorDestination::Stream,
        MxmDataFormat data_format = MxmDataFormat::Float16,
        MxmComputeMode compute_mode = MxmComputeMode::Vector,
        bool accumulator_clear = true,
        MxmAccumulatorOutputFormat accumulator_output_format =
            MxmAccumulatorOutputFormat::Float32)
    {
        check_weight_buffer(weight_buffer);
        check_compute_mode(compute_mode);
        check_activation_stream_base(activation_stream_base, compute_mode);
        check_stream_base(
            stream_base, compute_mode, accumulator_output_format);
        check_compute_destination(compute_mode, accumulator_destination);
        auto instruction = MxmControlInstruction {};
        instruction.opcode = MxmControlOpcode::Compute;
        instruction.weight_buffer = weight_buffer;
        instruction.stream_base = stream_base;
        instruction.activation_stream_base = activation_stream_base;
        instruction.accumulator_address = accumulator_address;
        instruction.accumulator_row_stride = accumulator_row_stride;
        instruction.accumulator_destination = accumulator_destination;
        instruction.accumulator_clear = accumulator_clear;
        instruction.data_format = data_format;
        instruction.compute_mode = compute_mode;
        instruction.accumulator_output_format = accumulator_output_format;
        return instruction;
    }

    static MxmControlInstruction AccumulatorRead(
        std::size_t accumulator_address,
        std::size_t stream_base = 0,
        bool clear = true,
        MxmComputeMode compute_mode = MxmComputeMode::Vector,
        MxmAccumulatorOutputFormat accumulator_output_format =
            MxmAccumulatorOutputFormat::Float32,
        MxmAccumulatorDestination accumulator_destination =
            MxmAccumulatorDestination::Stream)
    {
        check_compute_mode(compute_mode);
        check_accumulator_address(accumulator_address, compute_mode);
        check_accumulator_read_stream_base(
            stream_base, compute_mode, accumulator_output_format);
        auto instruction = MxmControlInstruction {};
        instruction.opcode = MxmControlOpcode::AccumulatorRead;
        instruction.stream_base = stream_base;
        instruction.accumulator_address = accumulator_address;
        instruction.accumulator_clear = clear;
        instruction.compute_mode = compute_mode;
        instruction.accumulator_output_format = accumulator_output_format;
        instruction.accumulator_destination = accumulator_destination;
        return instruction;
    }

    static MxmControlInstruction DecodeLoadActivation(
        std::size_t activation_buffer = 0,
        std::size_t activation_stream_base = 0,
        MxmDataFormat data_format = MxmDataFormat::BFloat16,
        MxmDecodeLayout decode_layout = MxmDecodeLayout::Linear1x16)
    {
        check_weight_buffer(activation_buffer);
        check_data_format(data_format);
        check_decode_layout(decode_layout);
        check_decode_activation_stream_base(
            activation_stream_base, decode_layout);
        auto instruction = MxmControlInstruction {};
        instruction.opcode = MxmControlOpcode::Decode;
        instruction.decode_operation = MxmDecodeOperation::LoadActivation;
        instruction.weight_buffer = activation_buffer;
        instruction.activation_stream_base = activation_stream_base;
        instruction.data_format = data_format;
        instruction.decode_layout = decode_layout;
        return instruction;
    }

    static MxmControlInstruction DecodeStreamCompute(
        std::size_t activation_buffer,
        std::size_t output_stream_base = 0,
        MxmDataFormat data_format = MxmDataFormat::BFloat16,
        std::size_t accumulator_address = 0,
        std::size_t accumulator_column = 0,
        MxmAccumulatorDestination accumulator_destination =
            MxmAccumulatorDestination::Stream,
        bool accumulator_clear = true,
        MxmDecodeLayout decode_layout = MxmDecodeLayout::Linear1x16)
    {
        check_weight_buffer(activation_buffer);
        check_data_format(data_format);
        check_decode_layout(decode_layout);
        check_accumulator_address(
            accumulator_address, MxmComputeMode::Vector);
        check_column(accumulator_column);
        if (accumulator_destination
            == MxmAccumulatorDestination::Stream) {
            check_decode_output_stream_base(output_stream_base);
        }
        auto instruction = MxmControlInstruction {};
        instruction.opcode = MxmControlOpcode::Decode;
        instruction.decode_operation = MxmDecodeOperation::StreamCompute;
        instruction.weight_buffer = activation_buffer;
        instruction.stream_base = output_stream_base;
        instruction.data_format = data_format;
        instruction.accumulator_address = accumulator_address;
        instruction.weight_column = accumulator_column;
        instruction.accumulator_destination = accumulator_destination;
        instruction.accumulator_clear = accumulator_clear;
        instruction.decode_layout = decode_layout;
        return instruction;
    }

    static void check_weight_buffer(std::size_t weight_buffer)
    {
        if (weight_buffer >= MxmSupercell::kWeightBuffers) {
            throw std::out_of_range("MXM weight buffer is outside the two-buffer set");
        }
    }

    static void check_column(std::size_t column)
    {
        if (column >= hw::kMxmSupercellsPerPlane) {
            throw std::out_of_range("MXM weight column is outside the array");
        }
    }

    static void check_inner_column(std::size_t column)
    {
        if (column >= hw::kMxmSupercellColumns) {
            throw std::out_of_range(
                "MXM weight inner column is outside the supercell");
        }
    }

    static void check_weight_load(
        MxmWeightLoadMode mode,
        std::size_t inner_column)
    {
        switch (mode) {
        case MxmWeightLoadMode::Supercell:
            if (inner_column != 0) {
                throw std::invalid_argument(
                    "MXM full-supercell IW cannot select an inner column");
            }
            return;
        case MxmWeightLoadMode::Column:
            check_inner_column(inner_column);
            return;
        }
        throw std::invalid_argument("MXM weight load mode is invalid");
    }

    static void check_weight_input_mode(MxmWeightInputMode mode)
    {
        switch (mode) {
        case MxmWeightInputMode::Int8DequantBf16:
        case MxmWeightInputMode::Direct16:
            return;
        }
        throw std::invalid_argument("MXM weight input mode is invalid");
    }

    static void check_weight_stream_base(std::size_t stream_base,
        MxmWeightInputMode input_mode, MxmWeightLoadMode load_mode)
    {
        check_weight_input_mode(input_mode);
        const auto stream_count = load_mode == MxmWeightLoadMode::Column
            ? input_mode == MxmWeightInputMode::Int8DequantBf16 ? 1 : 2
            : input_mode == MxmWeightInputMode::Int8DequantBf16
            ? hw::kMxmInt8LoadStreamsPerCycle
            : hw::kMxmLoadStreamsPerCycle;
        if (stream_base + stream_count > hw::kEastStreams) {
            throw std::out_of_range(
                "MXM weight stream range is outside the east stream set");
        }
    }

    static void check_data_format(MxmDataFormat format)
    {
        switch (format) {
        case MxmDataFormat::Float16:
        case MxmDataFormat::BFloat16:
            return;
        }
        throw std::invalid_argument("MXM data format is invalid");
    }

    static void check_compute_mode(MxmComputeMode mode)
    {
        switch (mode) {
        case MxmComputeMode::Vector:
        case MxmComputeMode::Block8:
            return;
        }
        throw std::invalid_argument("MXM compute mode is invalid");
    }

    static void check_activation_stream_base(
        std::size_t activation_stream_base,
        MxmComputeMode mode)
    {
        check_compute_mode(mode);
        const auto stream_count = mode == MxmComputeMode::Block8
            ? hw::kMxmActivationStreamsPerBlock
            : hw::kMxmActivationStreamsPerVector;
        if (activation_stream_base + stream_count > hw::kEastStreams) {
            throw std::out_of_range(
                "MXM activation stream range is outside the east stream set");
        }
    }

    static void check_compute_destination(
        MxmComputeMode mode,
        MxmAccumulatorDestination destination)
    {
        (void)mode;
        (void)destination;
    }

    static void check_stream_base(
        std::size_t stream_base,
        MxmComputeMode mode,
        MxmAccumulatorOutputFormat output_format =
            MxmAccumulatorOutputFormat::Float32)
    {
        const auto stream_count = mode == MxmComputeMode::Block8
            ? hw::kMxmBlockRows * sizeof(std::uint16_t)
            : output_format == MxmAccumulatorOutputFormat::BFloat16
            ? sizeof(std::uint16_t)
            : sizeof(float);
        if (stream_base + stream_count > hw::kWestStreams) {
            throw std::out_of_range(
                "MXM output stream range is outside the west stream set");
        }
    }

    static void check_accumulator_read_stream_base(
        std::size_t stream_base,
        MxmComputeMode mode,
        MxmAccumulatorOutputFormat output_format =
            MxmAccumulatorOutputFormat::Float32)
    {
        const auto stream_count = mode == MxmComputeMode::Block8
            ? hw::kMxmBlockRows
                * (output_format == MxmAccumulatorOutputFormat::BFloat16
                       ? sizeof(std::uint16_t)
                       : sizeof(float))
            : output_format == MxmAccumulatorOutputFormat::BFloat16
            ? sizeof(std::uint16_t)
            : sizeof(float);
        if (stream_base + stream_count > hw::kWestStreams) {
            throw std::out_of_range(
                "MXM accumulator read stream range is outside the west stream set");
        }
    }

    static void check_accumulator_address(
        std::size_t address,
        MxmComputeMode mode = MxmComputeMode::Vector)
    {
        check_compute_mode(mode);
        const auto rows = mode == MxmComputeMode::Block8
            ? hw::kMxmBlockAccumulatorRows
            : hw::kMxmAccumulatorRows;
        if (address >= rows) {
            throw std::out_of_range(
                "MXM accumulator address " + std::to_string(address)
                + " is outside the configured " + std::to_string(rows)
                + "-row SRAM");
        }
    }

    static void check_decode_layout(MxmDecodeLayout layout)
    {
        if (layout != MxmDecodeLayout::Linear1x16
            && layout != MxmDecodeLayout::Native4x4) {
            throw std::out_of_range("MXM decode layout is invalid");
        }
    }

    static void check_decode_activation_stream_base(
        std::size_t activation_stream_base,
        MxmDecodeLayout layout = MxmDecodeLayout::Linear1x16)
    {
        check_decode_layout(layout);
        const auto kActivationStreams = layout == MxmDecodeLayout::Native4x4
            ? sizeof(std::uint16_t)
            : hw::kMxmSupercellsPerPlane * sizeof(std::uint16_t);
        if (activation_stream_base + kActivationStreams
            > hw::kEastStreams) {
            throw std::out_of_range(
                "MXM decode activation stream range is outside the east stream set");
        }
    }

    static void check_decode_output_stream_base(std::size_t stream_base)
    {
        constexpr auto kOutputStreams = sizeof(std::uint16_t);
        if (stream_base + kOutputStreams > hw::kWestStreams) {
            throw std::out_of_range(
                "MXM decode output stream range is outside the west stream set");
        }
    }
};

class MxmControlSlice {
public:
    using WeightInput = MxmWeightInput;
    using InstructionSlot = std::optional<MxmControlInstruction>;
    using DequantInstructionSlot = std::optional<MxmDequantInstruction>;
    using WeightInputSlot = std::optional<WeightInput>;
    using WeightInputProvider = std::function<WeightInput(std::size_t)>;

    struct ComputePulse {
        std::size_t weight_buffer{0};
        std::size_t activation_stream_base{0};
        std::size_t stream_base{0};
        std::size_t accumulator_address{0};
        std::size_t accumulator_row_stride{1};
        MxmAccumulatorDestination accumulator_destination{
            MxmAccumulatorDestination::Stream};
        bool accumulator_clear{true};
        MxmDataFormat data_format{MxmDataFormat::Float16};
        MxmComputeMode compute_mode{MxmComputeMode::Vector};
        MxmAccumulatorOutputFormat accumulator_output_format{
            MxmAccumulatorOutputFormat::Float32};
    };

    struct DecodeActivationLoadPulse {
        std::size_t activation_buffer{0};
        std::size_t stream_base{0};
        MxmDataFormat data_format{MxmDataFormat::BFloat16};
        MxmDecodeLayout layout{MxmDecodeLayout::Linear1x16};
    };

    struct DecodeStreamComputePulse {
        std::size_t activation_buffer{0};
        std::size_t output_stream_base{0};
        MxmDataFormat data_format{MxmDataFormat::BFloat16};
        std::size_t accumulator_address{0};
        std::size_t accumulator_column{0};
        MxmAccumulatorDestination accumulator_destination{
            MxmAccumulatorDestination::Stream};
        bool accumulator_clear{true};
        MxmDequantInstruction dequant{};
        std::uint64_t wave_id{0};
        MxmDecodeLayout layout{MxmDecodeLayout::Linear1x16};
    };

    struct AccumulatorReadPulse {
        std::size_t address{0};
        std::size_t stream_base{0};
        bool clear{true};
        MxmComputeMode compute_mode{MxmComputeMode::Vector};
        MxmAccumulatorOutputFormat output_format{
            MxmAccumulatorOutputFormat::Float32};
        MxmAccumulatorDestination destination{
            MxmAccumulatorDestination::Stream};
    };

    explicit MxmControlSlice(MxmArray& array)
        : array_(array)
    {
    }

    void reset()
    {
        load_instruction_queue_.clear();
        dequant_instruction_queue_.clear();
        compute_instruction_queue_.clear();
        for (auto& slot : load_instruction_rows_) {
            slot.reset();
        }
        for (auto& slot : compute_instruction_rows_) {
            slot.reset();
        }
        for (auto& slot : dequant_instruction_rows_) {
            slot.reset();
        }
        for (auto& slot : weight_inputs_) {
            slot.reset();
        }
        compute_pulses_.fill(false);
        for (auto& pulse : compute_pulse_details_) {
            pulse.reset();
        }
        for (auto& pulse : accumulator_read_pulse_details_) {
            pulse.reset();
        }
        for (auto& pulse : decode_activation_load_pulse_details_) {
            pulse.reset();
        }
        for (auto& pulse : decode_stream_compute_pulse_details_) {
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
        if (instruction.opcode == MxmControlOpcode::IW
            || (instruction.opcode == MxmControlOpcode::Decode
                && instruction.decode_operation
                    == MxmDecodeOperation::LoadActivation)) {
            load_instruction_queue_.push_back(instruction);
        } else {
            compute_instruction_queue_.push_back(instruction);
        }
    }

    void issue_dequant_south(MxmDequantInstruction instruction)
    {
        dequant_instruction_queue_.push_back(instruction);
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

    const DequantInstructionSlot& dequant_instruction_at(
        std::size_t tile) const
    {
        check_tile(tile);
        return dequant_instruction_rows_[tile];
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

    std::optional<std::size_t> compute_activation_stream_base(std::size_t tile) const
    {
        check_tile(tile);
        return compute_pulse_details_[tile].has_value()
            ? std::optional<std::size_t> {compute_pulse_details_[tile]->activation_stream_base}
            : std::nullopt;
    }

    std::optional<ComputePulse> compute_pulse(std::size_t tile) const
    {
        check_tile(tile);
        return compute_pulse_details_[tile];
    }

    std::optional<AccumulatorReadPulse> accumulator_read_pulse(
        std::size_t tile) const
    {
        check_tile(tile);
        return accumulator_read_pulse_details_[tile];
    }

    std::optional<DecodeActivationLoadPulse> decode_activation_load_pulse(
        std::size_t tile) const
    {
        check_tile(tile);
        return decode_activation_load_pulse_details_[tile];
    }

    std::optional<DecodeStreamComputePulse> decode_stream_compute_pulse(
        std::size_t tile) const
    {
        check_tile(tile);
        return decode_stream_compute_pulse_details_[tile];
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
        dispatch_dequant_instruction();
        dispatch_compute_instruction();
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
            throw std::out_of_range("MXM control tile is outside the configured array");
        }
    }

    static void check_column(std::size_t column)
    {
        if (column >= hw::kMxmSupercellsPerPlane) {
            throw std::out_of_range("MXM control supercell column is outside the configured array");
        }
    }

    static void check_instruction(const MxmControlInstruction& instruction)
    {
        MxmControlInstruction::check_weight_buffer(instruction.weight_buffer);
        if (instruction.opcode == MxmControlOpcode::Compute) {
            MxmControlInstruction::check_data_format(
                instruction.data_format);
            MxmControlInstruction::check_compute_mode(
                instruction.compute_mode);
            MxmControlInstruction::check_activation_stream_base(
                instruction.activation_stream_base,
                instruction.compute_mode);
            MxmControlInstruction::check_stream_base(
                instruction.stream_base,
                instruction.compute_mode,
                instruction.accumulator_output_format);
            MxmControlInstruction::check_compute_destination(
                instruction.compute_mode,
                instruction.accumulator_destination);
            MxmControlInstruction::check_accumulator_address(
                instruction.accumulator_address,
                instruction.compute_mode);
            if (instruction.accumulator_row_stride == 0) {
                throw std::invalid_argument(
                    "MXM accumulator row stride must be at least one");
            }
        } else if (instruction.opcode == MxmControlOpcode::Decode) {
            MxmControlInstruction::check_data_format(
                instruction.data_format);
            MxmControlInstruction::check_decode_layout(
                instruction.decode_layout);
            if (instruction.decode_operation
                == MxmDecodeOperation::LoadActivation) {
                MxmControlInstruction::check_decode_activation_stream_base(
                    instruction.activation_stream_base,
                    instruction.decode_layout);
            } else if (instruction.decode_operation
                == MxmDecodeOperation::StreamCompute) {
                MxmControlInstruction::check_accumulator_address(
                    instruction.accumulator_address,
                    MxmComputeMode::Vector);
                MxmControlInstruction::check_column(
                    instruction.weight_column);
                if (instruction.accumulator_destination
                    == MxmAccumulatorDestination::Stream) {
                    MxmControlInstruction::check_decode_output_stream_base(
                        instruction.stream_base);
                }
            } else {
                throw std::invalid_argument(
                    "MXM decode operation is invalid");
            }
        } else if (instruction.opcode == MxmControlOpcode::AccumulatorRead) {
            MxmControlInstruction::check_compute_mode(
                instruction.compute_mode);
            MxmControlInstruction::check_accumulator_address(
                instruction.accumulator_address,
                instruction.compute_mode);
            MxmControlInstruction::check_accumulator_read_stream_base(
                instruction.stream_base,
                instruction.compute_mode,
                instruction.accumulator_output_format);
        } else {
            MxmControlInstruction::check_column(instruction.weight_column);
            MxmControlInstruction::check_weight_load(
                instruction.weight_load_mode,
                instruction.weight_inner_column);
            MxmControlInstruction::check_weight_input_mode(
                instruction.weight_input_mode);
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

    void dispatch_dequant_instruction()
    {
        if (dequant_instruction_rows_[0].has_value()
            || dequant_instruction_queue_.empty()) {
            return;
        }
        dequant_instruction_rows_[0] =
            dequant_instruction_queue_.front();
        dequant_instruction_queue_.pop_front();
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
        for (auto& pulse : compute_pulse_details_) {
            pulse.reset();
        }
        for (auto& pulse : accumulator_read_pulse_details_) {
            pulse.reset();
        }
        for (auto& pulse : decode_activation_load_pulse_details_) {
            pulse.reset();
        }
        for (auto& pulse : decode_stream_compute_pulse_details_) {
            pulse.reset();
        }
        bool any = false;
        bool any_logged = false;
        for (std::size_t tile = 0; tile < hw::kMxmSupercellsPerPlane; ++tile) {
            const auto& instruction = load_instruction_rows_[tile];
            const auto& dequant_instruction =
                dequant_instruction_rows_[tile];
            bool dequant_consumed = false;
            if (instruction.has_value()) {
                any = true;
                const auto should_log = !log_tile.has_value() || tile == *log_tile;
                if (should_log) {
                    any_logged = true;
                    os << "  tile " << tile << " ";
                }
                if (instruction->opcode == MxmControlOpcode::IW) {
                    const auto needs_dequant =
                        instruction->weight_input_mode
                        == MxmWeightInputMode::Int8DequantBf16;
                    dequant_consumed = needs_dequant;
                    if (needs_dequant != dequant_instruction.has_value()) {
                        throw std::logic_error(
                            needs_dequant
                                ? "MXM INT8 IW reached tile without a matching Dequant instruction"
                                : "MXM Dequant instruction cannot accompany a Direct16 IW");
                    }
                    if (!weight_inputs_[tile].has_value()) {
                        throw std::logic_error("MXM IW reached tile without local weight input");
                    }

                    if (should_log) {
                        os << "IW b" << instruction->weight_buffer
                           << " col=" << instruction->weight_column;
                        if (instruction->weight_load_mode
                            == MxmWeightLoadMode::Column) {
                            os << " inner=" << instruction->weight_inner_column;
                        }
                        if (needs_dequant) {
                            os << " int8 scale="
                               << dequant_instruction->scale();
                        } else {
                            os << " direct16";
                        }
                        os << " inject ";
                    }
                    const auto column = instruction->weight_column;
                    const auto input = dequantizer_.convert(
                        *weight_inputs_[tile],
                        instruction->weight_load_mode,
                        instruction->weight_inner_column,
                        instruction->weight_input_mode,
                        dequant_instruction.value_or(
                            MxmDequantInstruction {}));
                    const auto load = instruction->weight_load_mode
                            == MxmWeightLoadMode::Column
                        ? MxmInstruction::IWColumn(
                              instruction->weight_buffer,
                              instruction->weight_inner_column)
                        : MxmInstruction::IW(instruction->weight_buffer);
                    weight_inputs_[tile].reset();

                    any = true;
                    if (should_log) {
                        any_logged = true;
                        os << "  tile " << tile << " weight b" << instruction->weight_buffer
                           << " col=" << column;
                        if (instruction->weight_load_mode
                            == MxmWeightLoadMode::Column) {
                            os << " inner=" << instruction->weight_inner_column;
                        }
                        os << " ";
                        array_.tick_cell_iw_load(
                            tile, column, load, input, os);
                    } else {
                        static NullStream null_stream;
                        array_.tick_cell_iw_load(
                            tile,
                            column,
                            load,
                            input,
                            null_stream.stream());
                    }
                    loaded_cells_[instruction->weight_buffer][tile][column]
                        = array_.cell(tile, column).weight_buffer_valid(
                            instruction->weight_buffer);
                } else if (
                    instruction->opcode == MxmControlOpcode::Decode
                    && instruction->decode_operation
                        == MxmDecodeOperation::LoadActivation) {
                    if (should_log) {
                        os << "DecodeLoadActivation b"
                           << instruction->weight_buffer
                           << " stream="
                           << instruction->activation_stream_base
                           << " format="
                           << mxm_data_format_name(
                                  instruction->data_format)
                           << " layout="
                           << (instruction->decode_layout
                                       == MxmDecodeLayout::Native4x4
                                   ? "4x4"
                                   : "1x16")
                           << '\n';
                    }
                    decode_activation_load_pulse_details_[tile] =
                        DecodeActivationLoadPulse {
                            instruction->weight_buffer,
                            instruction->activation_stream_base,
                            instruction->data_format,
                            instruction->decode_layout};
                }
            }

            const auto& compute_instruction = compute_instruction_rows_[tile];
            if (compute_instruction.has_value()) {
                any = true;
                const auto should_log = !log_tile.has_value() || tile == *log_tile;
                if (should_log) {
                    any_logged = true;
                }
                if (compute_instruction->opcode == MxmControlOpcode::Compute) {
                    if (should_log) {
                        os << "  tile " << tile << " Compute b"
                           << compute_instruction->weight_buffer
                           << " stream=" << compute_instruction->activation_stream_base
                           << " format="
                           << mxm_data_format_name(
                                  compute_instruction->data_format)
                           << " mode="
                           << (compute_instruction->compute_mode
                                      == MxmComputeMode::Block8
                                   ? "block8"
                                   : "vector")
                           << " acc=" << compute_instruction->accumulator_address
                           << " out=" << compute_instruction->stream_base << '\n';
                    }
                    compute_pulses_[tile] = true;
                    compute_pulse_details_[tile] = ComputePulse {
                        compute_instruction->weight_buffer,
                        compute_instruction->activation_stream_base,
                        compute_instruction->stream_base,
                        compute_instruction->accumulator_address,
                        compute_instruction->accumulator_row_stride,
                        compute_instruction->accumulator_destination,
                        compute_instruction->accumulator_clear,
                        compute_instruction->data_format,
                        compute_instruction->compute_mode,
                        compute_instruction->accumulator_output_format,
                    };
                } else if (
                    compute_instruction->opcode == MxmControlOpcode::Decode
                    && compute_instruction->decode_operation
                        == MxmDecodeOperation::StreamCompute) {
                    if (!dequant_instruction.has_value()) {
                        throw std::logic_error(
                            "MXM decode StreamCompute reached tile without a matching Dequant instruction");
                    }
                    dequant_consumed = true;
                    if (should_log) {
                        os << "  tile " << tile
                           << " DecodeStreamCompute b"
                           << compute_instruction->weight_buffer
                           << " weight_streams=E0..E31"
                           << " scale="
                           << dequant_instruction->scale()
                           << " layout="
                           << (compute_instruction->decode_layout
                                       == MxmDecodeLayout::Native4x4
                                   ? "4x4"
                                   : "1x16")
                           << " acc="
                           << compute_instruction->accumulator_address
                           << ':' << compute_instruction->weight_column
                           << (compute_instruction->accumulator_destination
                                   == MxmAccumulatorDestination::Sram
                               ? " dst=sram"
                               : " dst=stream");
                        if (compute_instruction->accumulator_destination
                            == MxmAccumulatorDestination::Stream) {
                            os << " out=W"
                               << compute_instruction->stream_base
                               << "..W"
                               << compute_instruction->stream_base
                                      + sizeof(std::uint16_t) - 1;
                        }
                        os << '\n';
                    }
                    decode_stream_compute_pulse_details_[tile] =
                        DecodeStreamComputePulse {
                            compute_instruction->weight_buffer,
                            compute_instruction->stream_base,
                            compute_instruction->data_format,
                            compute_instruction->accumulator_address,
                            compute_instruction->weight_column,
                            compute_instruction->accumulator_destination,
                            compute_instruction->accumulator_clear,
                            *dequant_instruction,
                            cycle_ - tile,
                            compute_instruction->decode_layout};
                } else if (
                    compute_instruction->opcode
                    == MxmControlOpcode::AccumulatorRead) {
                    if (should_log) {
                        os << "  tile " << tile << " AccumulatorRead address="
                           << compute_instruction->accumulator_address
                           << " out=" << compute_instruction->stream_base
                           << " mode="
                           << (compute_instruction->compute_mode
                                      == MxmComputeMode::Block8
                                   ? "block8"
                                   : "vector")
                           << (compute_instruction->accumulator_clear
                                   ? " clear" : " retain")
                           << (compute_instruction->accumulator_destination
                                       == MxmAccumulatorDestination::Stream
                                   ? " dst=stream" : " dst=sram")
                           << '\n';
                    }
                    accumulator_read_pulse_details_[tile] =
                        AccumulatorReadPulse {
                            compute_instruction->accumulator_address,
                            compute_instruction->stream_base,
                            compute_instruction->accumulator_clear,
                            compute_instruction->compute_mode,
                            compute_instruction->accumulator_output_format,
                            compute_instruction->accumulator_destination};
                }
            }
            if (dequant_instruction.has_value() && !dequant_consumed) {
                throw std::logic_error(
                    "MXM Dequant instruction reached tile without a matching INT8 weight consumer");
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
            dequant_instruction_rows_[tile] =
                dequant_instruction_rows_[tile - 1];
            compute_instruction_rows_[tile] = compute_instruction_rows_[tile - 1];
        }
        load_instruction_rows_[0].reset();
        dequant_instruction_rows_[0].reset();
        compute_instruction_rows_[0].reset();
    }

    MxmArray& array_;
    MxmWeightDequantizer dequantizer_{};
    std::deque<MxmControlInstruction> load_instruction_queue_{};
    std::deque<MxmDequantInstruction> dequant_instruction_queue_{};
    std::deque<MxmControlInstruction> compute_instruction_queue_{};
    std::array<InstructionSlot, hw::kMxmSupercellsPerPlane> load_instruction_rows_{};
    std::array<DequantInstructionSlot, hw::kMxmSupercellsPerPlane>
        dequant_instruction_rows_{};
    std::array<InstructionSlot, hw::kMxmSupercellsPerPlane> compute_instruction_rows_{};
    std::array<WeightInputSlot, hw::kMxmSupercellsPerPlane> weight_inputs_{};
    std::array<bool, hw::kMxmSupercellsPerPlane> compute_pulses_{};
    std::array<std::optional<ComputePulse>, hw::kMxmSupercellsPerPlane> compute_pulse_details_{};
    std::array<std::optional<AccumulatorReadPulse>, hw::kMxmSupercellsPerPlane>
        accumulator_read_pulse_details_{};
    std::array<
        std::optional<DecodeActivationLoadPulse>,
        hw::kMxmSupercellsPerPlane>
        decode_activation_load_pulse_details_{};
    std::array<
        std::optional<DecodeStreamComputePulse>,
        hw::kMxmSupercellsPerPlane>
        decode_stream_compute_pulse_details_{};
    std::array<
        std::array<std::array<bool, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane>,
        MxmSupercell::kWeightBuffers> loaded_cells_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
