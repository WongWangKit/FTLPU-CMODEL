#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/stream.hpp"
#include "ftlpu/mxm/data_format.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace ftlpu {

enum class MxmOpcode {
    IW,
};

enum class MxmWeightLoadMode {
    Supercell,
    Column,
};

struct MxmInstruction {
    MxmOpcode opcode{MxmOpcode::IW};
    std::size_t weight_buffer{0};
    MxmWeightLoadMode weight_load_mode{MxmWeightLoadMode::Supercell};
    std::size_t weight_inner_column{0};

    static MxmInstruction IW(std::size_t weight_buffer = 0)
    {
        return MxmInstruction {
            MxmOpcode::IW,
            weight_buffer,
            MxmWeightLoadMode::Supercell,
            0};
    }

    static MxmInstruction IWColumn(
        std::size_t weight_buffer,
        std::size_t inner_column)
    {
        return MxmInstruction {
            MxmOpcode::IW,
            weight_buffer,
            MxmWeightLoadMode::Column,
            inner_column};
    }
};

class MxmSupercell {
public:
    using Weight = float;
    using WeightBits = std::uint16_t;
    using InputWord = StreamWord<WeightBits>;
    using InputSlot = std::optional<InputWord>;
    using InputLane = std::array<InputSlot, hw::kMxmSupercellColumns>;
    using InputVector = std::array<InputLane, hw::kLanesPerTile>;
    using ActivationWord = StreamWord<float>;
    using ActivationSlot = std::optional<ActivationWord>;
    using ActivationVector = std::array<ActivationSlot, hw::kLanesPerTile>;
    using ActivationData = std::array<float, hw::kLanesPerTile>;
    using WeightRow = std::array<WeightBits, hw::kMxmSupercellColumns>;
    using WeightMatrix = std::array<WeightRow, hw::kMxmSupercellRows>;
    static constexpr std::size_t kWeightBuffers = 2;

    struct ComputeResult {
        std::size_t column{0};
        float value{0.0f};
    };

    struct ActivationPayload {
        ActivationData data{};
        std::size_t weight_buffer{0};
        MxmDataFormat data_format{MxmDataFormat::Float16};
    };

    void reset()
    {
        for (auto& matrix : weight_buffers_) {
            for (auto& row : matrix) {
                row.fill(0);
            }
        }
        weight_buffer_valid_.fill(false);
        for (auto& columns : weight_column_valid_) {
            columns.fill(false);
        }
        input_ = {};
        activation_input_.reset();
        for (auto& stage : activation_stages_) {
            stage.reset();
        }
        outputs_.clear();
        instruction_.reset();
    }

    void set_input(InputVector input)
    {
        input_ = input;
    }

    void set_activation_input(
        ActivationVector input,
        std::size_t weight_buffer = 0,
        MxmDataFormat data_format = MxmDataFormat::Float16)
    {
        check_buffer(weight_buffer);
        if (activation_input_.has_value()) {
            throw std::logic_error("MXM supercell already has an activation input for this cycle");
        }
        activation_input_ = ActivationPayload {
            decode_activation(input), weight_buffer, data_format};
    }

    void issue(MxmInstruction instruction)
    {
        if (instruction_.has_value()) {
            throw std::logic_error("MXM supercell already has an instruction for this cycle");
        }
        instruction_ = instruction;
    }

    const WeightMatrix& weight_buffer_bits(std::size_t buffer) const
    {
        check_buffer(buffer);
        return weight_buffers_[buffer];
    }

    const WeightMatrix& weight_buffer_bits() const
    {
        return weight_buffer_bits(0);
    }

    const WeightMatrix& weight_buffer(std::size_t buffer) const
    {
        return weight_buffer_bits(buffer);
    }

    const WeightMatrix& weight_buffer() const
    {
        return weight_buffer_bits();
    }

    Weight weight(std::size_t buffer, std::size_t row, std::size_t column) const
    {
        check_buffer(buffer);
        check_row(row);
        check_column(column);
        return decode_mxm_16bit(
            weight_buffers_[buffer][row][column],
            MxmDataFormat::Float16);
    }

    Weight weight(
        std::size_t buffer,
        std::size_t row,
        std::size_t column,
        MxmDataFormat format) const
    {
        check_buffer(buffer);
        check_row(row);
        check_column(column);
        return decode_mxm_16bit(weight_buffers_[buffer][row][column], format);
    }

    WeightBits weight_bits(
        std::size_t buffer,
        std::size_t row,
        std::size_t column) const
    {
        check_buffer(buffer);
        check_row(row);
        check_column(column);
        return weight_buffers_[buffer][row][column];
    }

    Weight weight(std::size_t row, std::size_t column) const
    {
        return weight(0, row, column);
    }

    Weight buffered_weight(std::size_t row, std::size_t column) const
    {
        return weight(0, row, column);
    }

    bool weight_buffer_valid(std::size_t buffer) const
    {
        check_buffer(buffer);
        return weight_buffer_valid_[buffer];
    }

    bool weight_buffer_valid() const
    {
        return weight_buffer_valid(0);
    }

    const std::vector<ComputeResult>& outputs() const
    {
        return outputs_;
    }

    void tick(std::ostream& os)
    {
        os << "mxm_supercell:";
        outputs_.clear();

        if (!instruction_.has_value()) {
            os << " no_weight_instruction";
        } else if (instruction_->opcode == MxmOpcode::IW) {
            load_weight_buffer(os);
        }

        compute(os);

        if (!instruction_.has_value() && outputs_.empty() && !activation_input_.has_value()) {
            os << " idle";
        }
        os << '\n';

        instruction_.reset();
        input_ = {};
        activation_input_.reset();
    }

private:
    static void check_row(std::size_t row)
    {
        if (row >= hw::kMxmSupercellRows) {
            throw std::out_of_range("MXM supercell row is outside 16x16 weights");
        }
    }

    static void check_column(std::size_t column)
    {
        if (column >= hw::kMxmSupercellColumns) {
            throw std::out_of_range("MXM supercell column is outside 16x16 weights");
        }
    }

    static void check_buffer(std::size_t buffer)
    {
        if (buffer >= kWeightBuffers) {
            throw std::out_of_range("MXM supercell weight buffer is outside the two-buffer set");
        }
    }

    static ActivationData decode_activation(const ActivationVector& input)
    {
        ActivationData data{};
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            if (!input[lane].has_value()) {
                throw std::logic_error("MXM compute requires all activation lanes to be valid");
            }
            data[lane] = input[lane]->data;
        }
        return data;
    }

    void compute(std::ostream& os)
    {
        if (activation_input_.has_value() && activation_stages_[0].has_value()) {
            throw std::logic_error("MXM activation input stage is occupied");
        }
        if (activation_input_.has_value()) {
            activation_stages_[0] = activation_input_;
        }

        for (std::size_t column = 0; column < hw::kMxmSupercellColumns; ++column) {
            if (!activation_stages_[column].has_value()) {
                continue;
            }

            const auto value = dot_product(*activation_stages_[column], column);
            outputs_.push_back(ComputeResult {column, value});
            os << " MAC col=" << column << " result=" << value;
        }

        advance_activation_stages();
    }

    float dot_product(const ActivationPayload& activation, std::size_t column) const
    {
        check_column(column);
        check_buffer(activation.weight_buffer);
        if (!weight_buffer_valid_[activation.weight_buffer]) {
            throw std::logic_error("MXM compute requires a valid selected weight buffer");
        }
        float sum = 0.0f;
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            sum += activation.data[lane]
                * decode_mxm_16bit(
                    weight_buffers_[activation.weight_buffer][lane][column],
                    activation.data_format);
        }
        return sum;
    }

    void advance_activation_stages()
    {
        for (std::size_t column = hw::kMxmSupercellColumns - 1; column > 0; --column) {
            activation_stages_[column] = activation_stages_[column - 1];
        }
        activation_stages_[0].reset();
    }

    void load_weight_buffer(std::ostream& os)
    {
        check_buffer(instruction_->weight_buffer);
        write_buffer_from_input();
        os << " IW buffer" << instruction_->weight_buffer << "=0x";
        print_matrix_hex(os, weight_buffers_[instruction_->weight_buffer]);
    }

    void write_buffer_from_input()
    {
        const auto buffer = instruction_->weight_buffer;
        if (instruction_->weight_load_mode == MxmWeightLoadMode::Column) {
            const auto column = instruction_->weight_inner_column;
            check_column(column);
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                if (!input_[lane][column].has_value()) {
                    throw std::logic_error(
                        "MXM column IW requires both weight streams on every lane");
                }
                weight_buffers_[buffer][lane][column]
                    = input_[lane][column]->data;
            }
            weight_column_valid_[buffer][column] = true;
            weight_buffer_valid_[buffer] = std::all_of(
                weight_column_valid_[buffer].begin(),
                weight_column_valid_[buffer].end(),
                [](bool valid) { return valid; });
            return;
        }
        if (instruction_->weight_load_mode != MxmWeightLoadMode::Supercell) {
            throw std::invalid_argument("MXM weight load mode is invalid");
        }

        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            for (std::size_t column = 0; column < hw::kMxmSupercellColumns; ++column) {
                if (!input_[lane][column].has_value()) {
                    throw std::logic_error("MXM IW requires every weight stream on every lane to be valid");
                }
                weight_buffers_[buffer][lane][column]
                    = input_[lane][column]->data;
            }
        }
        weight_column_valid_[buffer].fill(true);
        weight_buffer_valid_[buffer] = true;
    }

    static void print_matrix_hex(std::ostream& os, const WeightMatrix& matrix)
    {
        const auto old_flags = os.flags();
        const auto old_fill = os.fill();
        os << std::hex << std::setfill('0');
        for (const auto& row : matrix) {
            for (const auto bits : row) {
                os << std::setw(4) << static_cast<unsigned>(bits);
            }
        }
        os.flags(old_flags);
        os.fill(old_fill);
    }

    std::array<WeightMatrix, kWeightBuffers> weight_buffers_{};
    InputVector input_{};
    std::optional<ActivationPayload> activation_input_{};
    std::array<std::optional<ActivationPayload>, hw::kMxmSupercellColumns> activation_stages_{};
    std::vector<ComputeResult> outputs_{};
    std::optional<MxmInstruction> instruction_{};
    std::array<
        std::array<bool, hw::kMxmSupercellColumns>,
        kWeightBuffers> weight_column_valid_{};
    std::array<bool, kWeightBuffers> weight_buffer_valid_{};
};

} // namespace ftlpu
