#pragma once

#include "ftlpu/hardware_params.hpp"
#include "ftlpu/stream.hpp"

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
        input_ = {};
        activation_input_.reset();
        for (auto& stage : activation_stages_) {
            stage.reset();
        }
        outputs_.clear();
        instruction_.reset();
        load_enabled_ = false;
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

    Weight weight(std::size_t row, std::size_t column) const
    {
        check_row(row);
        check_column(column);
        return weights_[row][column];
    }

    bool load_enabled() const
    {
        return load_enabled_;
    }

    const std::vector<ComputeResult>& outputs() const
    {
        return outputs_;
    }

    void load_weights_with_iw(InputVector input, std::ostream& os)
    {
        input_ = input;
        write_weights_from_input();
        os << "mxm_supercell: IW load matrix=0x";
        print_matrix_hex(os);
        os << '\n';
        input_ = {};
        load_enabled_ = false;
    }

    void tick(std::ostream& os)
    {
        os << "mxm_supercell:";
        outputs_.clear();

        if (!instruction_.has_value()) {
            os << " no_weight_instruction";
        } else if (instruction_->opcode == MxmOpcode::IW) {
            load_enabled_ = true;
            os << " IW enable_load";
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
        if (!load_enabled_) {
            throw std::logic_error("LW requires a prior IW");
        }

        write_weights_from_input();
        os << " LW matrix=0x";
        print_matrix_hex(os);
        load_enabled_ = false;
    }

    void write_weights_from_input()
    {
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            for (std::size_t stream = 0; stream < hw::kMxmLoadStreamsPerCycle; ++stream) {
                if (!input_[lane][stream].has_value()) {
                    throw std::logic_error("MXM weight load requires 16 streams on all 16 lanes to be valid");
                }
                weights_[lane][stream] = input_[lane][stream]->data;
            }
        }
    }

    void print_matrix_hex(std::ostream& os) const
    {
        const auto old_flags = os.flags();
        const auto old_fill = os.fill();
        os << std::hex << std::setfill('0');
        for (const auto& row : weights_) {
            for (const auto value : row) {
                os << std::setw(2) << static_cast<unsigned>(static_cast<std::uint8_t>(value));
            }
        }
        os.flags(old_flags);
        os.fill(old_fill);
    }

    WeightMatrix weights_{};
    InputVector input_{};
    std::optional<ActivationData> activation_input_{};
    std::array<std::optional<ActivationData>, hw::kMxmSupercellColumns> activation_stages_{};
    std::vector<ComputeResult> outputs_{};
    std::optional<MxmInstruction> instruction_{};
    bool load_enabled_{false};
};

} // namespace ftlpu
