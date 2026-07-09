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
    LW,
};

struct MxmInstruction {
    MxmOpcode opcode{MxmOpcode::IW};

    static MxmInstruction IW()
    {
        return MxmInstruction {MxmOpcode::IW};
    }

    static MxmInstruction LW()
    {
        return MxmInstruction {MxmOpcode::LW};
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

    struct ComputeResult {
        std::size_t column{0};
        std::int32_t value{0};
    };

    void reset()
    {
        for (auto& row : weights_) {
            row.fill(0);
        }
        for (auto& row : weight_buffer_) {
            row.fill(0);
        }
        weight_buffer_valid_ = false;
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

    void set_activation_input(ActivationVector input)
    {
        if (activation_input_.has_value()) {
            throw std::logic_error("MXM supercell already has an activation input for this cycle");
        }
        activation_input_ = decode_activation(input);
    }

    void issue(MxmInstruction instruction)
    {
        if (instruction_.has_value()) {
            throw std::logic_error("MXM supercell already has an instruction for this cycle");
        }
        instruction_ = instruction;
    }

    const WeightMatrix& weights() const
    {
        return weights_;
    }

    const WeightMatrix& weight_buffer() const
    {
        return weight_buffer_;
    }

    Weight weight(std::size_t row, std::size_t column) const
    {
        check_row(row);
        check_column(column);
        return weights_[row][column];
    }

    Weight buffered_weight(std::size_t row, std::size_t column) const
    {
        check_row(row);
        check_column(column);
        return weight_buffer_[row][column];
    }

    bool weight_buffer_valid() const
    {
        return weight_buffer_valid_;
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
        } else {
            load_weights(os);
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

    std::int32_t dot_product(const ActivationData& activation, std::size_t column) const
    {
        check_column(column);
        std::int32_t sum = 0;
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            sum += static_cast<std::int32_t>(activation[lane]) * static_cast<std::int32_t>(weights_[lane][column]);
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

    void load_weights(std::ostream& os)
    {
        if (!weight_buffer_valid_) {
            throw std::logic_error("LW requires a valid weight buffer from a prior IW");
        }

        weights_ = weight_buffer_;
        os << " LW matrix=0x";
        print_matrix_hex(os, weights_);
    }

    void load_weight_buffer(std::ostream& os)
    {
        write_buffer_from_input();
        os << " IW buffer=0x";
        print_matrix_hex(os, weight_buffer_);
    }

    void write_buffer_from_input()
    {
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            for (std::size_t stream = 0; stream < hw::kMxmLoadStreamsPerCycle; ++stream) {
                if (!input_[lane][stream].has_value()) {
                    throw std::logic_error("MXM IW requires 16 streams on all 16 lanes to be valid");
                }
                weight_buffer_[lane][stream] = input_[lane][stream]->data;
            }
        }
        weight_buffer_valid_ = true;
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

    WeightMatrix weights_{};
    WeightMatrix weight_buffer_{};
    InputVector input_{};
    std::optional<ActivationData> activation_input_{};
    std::array<std::optional<ActivationData>, hw::kMxmSupercellColumns> activation_stages_{};
    std::vector<ComputeResult> outputs_{};
    std::optional<MxmInstruction> instruction_{};
    bool weight_buffer_valid_{false};
};

} // namespace ftlpu
