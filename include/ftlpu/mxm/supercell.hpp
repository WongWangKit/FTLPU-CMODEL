#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/stream.hpp"

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

struct MxmInstruction {
    MxmOpcode opcode{MxmOpcode::IW};
    std::size_t weight_buffer{0};

    static MxmInstruction IW(std::size_t weight_buffer = 0)
    {
        return MxmInstruction {MxmOpcode::IW, weight_buffer};
    }
};

class MxmSupercell {
public:
    using Weight = std::int8_t;
    using InputWord = StreamWord<Weight>;
    using InputSlot = std::optional<InputWord>;
    using InputLane = std::array<InputSlot, hw::kMxmLoadStreamsPerCycle>;
    using InputVector = std::array<InputLane, hw::kLanesPerTile>;
    using ActivationVector = std::array<InputSlot, hw::kLanesPerTile>;
    using ActivationData = std::array<Weight, hw::kLanesPerTile>;
    using WeightRow = std::array<Weight, hw::kMxmSupercellColumns>;
    using WeightMatrix = std::array<WeightRow, hw::kMxmSupercellRows>;
    static constexpr std::size_t kWeightBuffers = 2;

    struct ComputeResult {
        std::size_t column{0};
        std::int32_t value{0};
    };

    struct ActivationPayload {
        ActivationData data{};
        std::size_t weight_buffer{0};
    };

    void reset()
    {
        for (auto& matrix : weight_buffers_) {
            for (auto& row : matrix) {
                row.fill(0);
            }
        }
        weight_buffer_valid_.fill(false);
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

    void set_activation_input(ActivationVector input, std::size_t weight_buffer = 0)
    {
        check_buffer(weight_buffer);
        if (activation_input_.has_value()) {
            throw std::logic_error("MXM supercell already has an activation input for this cycle");
        }
        activation_input_ = ActivationPayload {decode_activation(input), weight_buffer};
    }

    void issue(MxmInstruction instruction)
    {
        if (instruction_.has_value()) {
            throw std::logic_error("MXM supercell already has an instruction for this cycle");
        }
        instruction_ = instruction;
    }

    const WeightMatrix& weight_buffer(std::size_t buffer) const
    {
        check_buffer(buffer);
        return weight_buffers_[buffer];
    }

    const WeightMatrix& weight_buffer() const
    {
        return weight_buffer(0);
    }

    Weight weight(std::size_t buffer, std::size_t row, std::size_t column) const
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

    void load_weight_buffer(std::size_t buffer, const InputVector& input)
    {
        check_buffer(buffer);
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            for (std::size_t stream = 0; stream < hw::kMxmLoadStreamsPerCycle; ++stream) {
                if (!input[lane][stream].has_value()) {
                    throw std::logic_error("MXM IW requires 16 streams on all 16 lanes to be valid");
                }
                weight_buffers_[buffer][lane][stream] = input[lane][stream]->data;
            }
        }
        weight_buffer_valid_[buffer] = true;
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
                throw std::logic_error("MXM compute requires all 16 activation lanes to be valid");
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

    std::int32_t dot_product(const ActivationPayload& activation, std::size_t column) const
    {
        check_column(column);
        check_buffer(activation.weight_buffer);
        if (!weight_buffer_valid_[activation.weight_buffer]) {
            throw std::logic_error("MXM compute requires a valid selected weight buffer");
        }
        std::int32_t sum = 0;
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            sum += static_cast<std::int32_t>(activation.data[lane])
                * static_cast<std::int32_t>(weight_buffers_[activation.weight_buffer][lane][column]);
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
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            for (std::size_t stream = 0; stream < hw::kMxmLoadStreamsPerCycle; ++stream) {
                if (!input_[lane][stream].has_value()) {
                    throw std::logic_error("MXM IW requires 16 streams on all 16 lanes to be valid");
                }
                weight_buffers_[instruction_->weight_buffer][lane][stream] = input_[lane][stream]->data;
            }
        }
        weight_buffer_valid_[instruction_->weight_buffer] = true;
    }

    static void print_matrix_hex(std::ostream& os, const WeightMatrix& matrix)
    {
        const auto old_flags = os.flags();
        const auto old_fill = os.fill();
        os << std::hex << std::setfill('0');
        for (const auto& row : matrix) {
            for (const auto value : row) {
                os << std::setw(2) << static_cast<unsigned>(static_cast<std::uint8_t>(value));
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
    std::array<bool, kWeightBuffers> weight_buffer_valid_{};
};

} // namespace ftlpu
