#pragma once

#include "ftlpu/core/timing.hpp"
#include "ftlpu/mxm/bf16_arithmetic.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ftlpu::mxm_timing {

template <typename T, std::size_t Stages>
inline void shift_pipeline(std::array<T, Stages>& next,
    const std::array<T, Stages>& current, const T& input)
{
    static_assert(Stages > 0);
    next[0] = input;
    for (std::size_t stage = 1; stage < Stages; ++stage)
        next[stage] = current[stage - 1];
}

template <typename Group, std::size_t Groups>
struct WeightShiftResult {
    std::array<Group, Groups> data{};
    std::array<bool, Groups> valid{};
    bool forward_valid{};
    Group forward_data{};
};

template <typename Group, std::size_t Groups>
inline WeightShiftResult<Group, Groups> shift_weight_buffer(
    const std::array<Group, Groups>& current_data,
    const std::array<bool, Groups>& current_valid, const Group& input)
{
    static_assert(Groups > 0);
    WeightShiftResult<Group, Groups> result{};
    result.forward_valid = current_valid.back();
    result.forward_data = current_data.back();
    result.data[0] = input;
    result.valid[0] = true;
    for (std::size_t group = 1; group < Groups; ++group) {
        result.data[group] = current_data[group - 1];
        result.valid[group] = current_valid[group - 1];
    }
    return result;
}

class Bf16DotCsaPipe final : public timing::ClockedModule {
public:
    static constexpr std::size_t kTerms = 8;
    static constexpr std::size_t kStages = 9;
    using Bf16Vector = std::array<std::uint16_t, kTerms>;

    struct Inputs {
        bool rst_ni{true};
        bool valid_i{};
        Bf16Vector activation_i{};
        Bf16Vector weight_i{};
        std::uint16_t tag_i{};
    };

    struct Outputs {
        bool valid_o{};
        std::uint32_t result_o{};
        std::uint16_t tag_o{};
        bool alignment_lost_o{};
        std::array<bool, kStages> stage_valid_q{};
        std::array<std::uint16_t, kStages> stage_tag_q{};
    };

    void drive(Inputs inputs) { inputs_ = std::move(inputs); }
    [[nodiscard]] const Inputs& inputs() const noexcept { return inputs_; }

    [[nodiscard]] Outputs outputs() const noexcept
    {
        Outputs result{};
        const auto& current = state_.current();
        for (std::size_t stage = 0; stage < kStages; ++stage) {
            result.stage_valid_q[stage] = current.pipeline[stage].valid;
            result.stage_tag_q[stage] = current.pipeline[stage].tag;
        }
        result.valid_o = current.pipeline.back().valid;
        result.tag_o = current.pipeline.back().tag;
        result.result_o = current.result;
        result.alignment_lost_o = current.alignment_lost;
        return result;
    }

    void low() override
    {
        state_.prepare();
        auto& next = state_.next_value();
        const auto& current = state_.current();
        if (!inputs_.rst_ni) {
            next = {};
            return;
        }

        next.pipeline[0].valid = inputs_.valid_i;
        next.pipeline[0].tag = inputs_.tag_i;
        if (inputs_.valid_i) {
            const auto value = mxm_numeric::bf16_dot(
                inputs_.activation_i, inputs_.weight_i);
            next.pipeline[0].data = value.bits;
            next.pipeline[0].alignment_lost = value.alignment_lost;
        }
        for (std::size_t stage = 1; stage < kStages; ++stage) {
            next.pipeline[stage].valid = current.pipeline[stage - 1].valid;
            next.pipeline[stage].tag = current.pipeline[stage - 1].tag;
            if (current.pipeline[stage - 1].valid) {
                next.pipeline[stage].data = current.pipeline[stage - 1].data;
                next.pipeline[stage].alignment_lost
                    = current.pipeline[stage - 1].alignment_lost;
            }
        }
        if (current.pipeline[kStages - 2].valid) {
            next.result = current.pipeline[kStages - 2].data;
            next.alignment_lost
                = current.pipeline[kStages - 2].alignment_lost;
        }
    }

    void validate_posedge() const override { state_.validate_commit(); }
    void posedge() override { state_.commit(); }

private:
    struct Token {
        bool valid{};
        std::uint16_t tag{};
        std::uint32_t data{};
        bool alignment_lost{};
    };
    struct State {
        std::array<Token, kStages> pipeline{};
        std::uint32_t result{};
        bool alignment_lost{};
    };

    Inputs inputs_{};
    timing::ClockedState<State> state_{};
};

class Fp32AddPipe final : public timing::ClockedModule {
public:
    struct Inputs {
        bool rst_ni{true};
        bool valid_i{};
        std::uint32_t lhs_i{};
        std::uint32_t rhs_i{};
        std::uint16_t tag_i{};
    };
    struct Outputs {
        bool a1_valid_q{};
        bool a2_valid_q{};
        bool valid_o{};
        std::uint16_t a1_tag_q{};
        std::uint16_t a2_tag_q{};
        std::uint16_t tag_o{};
        std::uint32_t result_o{};
    };

    void drive(Inputs inputs) { inputs_ = inputs; }
    [[nodiscard]] const Inputs& inputs() const noexcept { return inputs_; }
    [[nodiscard]] Outputs outputs() const noexcept
    {
        const auto& value = state_.current();
        return Outputs{value.a1_valid, value.a2_valid, value.valid_o,
            value.a1_tag, value.a2_tag, value.tag_o, value.result_o};
    }

    void low() override
    {
        state_.prepare();
        auto& next = state_.next_value();
        const auto& current = state_.current();
        if (!inputs_.rst_ni) {
            next = {};
            return;
        }

        next.a1_valid = inputs_.valid_i;
        next.a2_valid = current.a1_valid;
        next.valid_o = current.a2_valid;
        if (inputs_.valid_i) {
            next.a1_tag = inputs_.tag_i;
            next.a1_result = mxm_numeric::fp32_add_bits(inputs_.lhs_i, inputs_.rhs_i);
        }
        if (current.a1_valid) {
            next.a2_tag = current.a1_tag;
            next.a2_result = current.a1_result;
        }
        if (current.a2_valid) {
            next.tag_o = current.a2_tag;
            next.result_o = current.a2_result;
        }
    }

    void validate_posedge() const override { state_.validate_commit(); }
    void posedge() override { state_.commit(); }

private:
    struct State {
        bool a1_valid{};
        bool a2_valid{};
        bool valid_o{};
        std::uint16_t a1_tag{};
        std::uint16_t a2_tag{};
        std::uint16_t tag_o{};
        std::uint32_t a1_result{};
        std::uint32_t a2_result{};
        std::uint32_t result_o{};
    };
    Inputs inputs_{};
    timing::ClockedState<State> state_{};
};

template <std::size_t Lanes = 8, std::size_t Streams = 32>
class WeightDequantPipe final : public timing::ClockedModule {
public:
    static_assert(Lanes > 0 && Streams >= 4);
    static constexpr std::size_t kQStream = 5;
    static constexpr std::size_t kScaleLoStream = 9;
    static constexpr std::size_t kScaleHiStream = 12;
    static constexpr std::size_t kZeroStream = 17;
    static_assert(kZeroStream < Streams);
    using Weight = std::array<std::uint16_t, Lanes>;
    using StreamValid = std::array<std::array<bool, Lanes>, Streams>;
    using StreamData = std::array<std::array<std::uint8_t, Lanes>, Streams>;

    struct Inputs {
        bool rst_ni{true};
        bool run_i{true};
        bool request_i{};
        bool buffer_i{};
        bool param_write_i{};
        bool param_write_buffer_i{};
        bool param_buffer_i{};
        StreamValid stream_valid_i{};
        StreamData stream_i{};
    };
    struct Outputs {
        StreamValid consume_o{};
        bool accepted_o{};
        bool valid_o{};
        bool buffer_o{};
        bool param_accepted_o{};
        std::array<bool, 2> param_ready_o{};
        Weight weight_bf16_o{};
        bool fault_o{};
        std::array<bool, 2> valid_q{};
        std::array<bool, 2> buffer_q{};
    };

    void drive(Inputs inputs) { inputs_ = std::move(inputs); }
    [[nodiscard]] const Inputs& inputs() const noexcept { return inputs_; }

    [[nodiscard]] Outputs outputs() const noexcept
    {
        const auto& current = state_.current();
        Outputs result{};
        bool weight_available = true;
        bool param_available = true;
        for (std::size_t lane = 0; lane < Lanes; ++lane) {
            weight_available = weight_available
                && inputs_.stream_valid_i[kQStream][lane];
            param_available = param_available
                && inputs_.stream_valid_i[kScaleLoStream][lane]
                && inputs_.stream_valid_i[kScaleHiStream][lane]
                && inputs_.stream_valid_i[kZeroStream][lane];
        }
        const bool collision = inputs_.request_i && inputs_.param_write_i
            && inputs_.param_buffer_i == inputs_.param_write_buffer_i;
        result.accepted_o = inputs_.run_i && inputs_.request_i
            && weight_available
            && current.param_ready[static_cast<std::size_t>(inputs_.param_buffer_i)]
            && !collision;
        result.param_accepted_o = inputs_.run_i && inputs_.param_write_i
            && param_available && !collision;
        if (result.accepted_o)
            for (std::size_t lane = 0; lane < Lanes; ++lane)
                result.consume_o[kQStream][lane] = true;
        if (result.param_accepted_o) {
            for (std::size_t lane = 0; lane < Lanes; ++lane) {
                result.consume_o[kScaleLoStream][lane] = true;
                result.consume_o[kScaleHiStream][lane] = true;
                result.consume_o[kZeroStream][lane] = true;
            }
        }
        result.valid_o = current.valid[1];
        result.buffer_o = current.buffer[1];
        result.param_ready_o = current.param_ready;
        result.weight_bf16_o = current.weight;
        result.fault_o = current.fault;
        result.valid_q = current.valid;
        result.buffer_q = current.buffer;
        return result;
    }

    void low() override
    {
        state_.prepare();
        auto& next = state_.next_value();
        const auto& current = state_.current();
        if (!inputs_.rst_ni) {
            next = {};
            return;
        }
        const auto observed = outputs();
        if (observed.param_accepted_o) {
            const auto bank = static_cast<std::size_t>(inputs_.param_write_buffer_i);
            next.param_ready[bank] = true;
            for (std::size_t lane = 0; lane < Lanes; ++lane) {
                next.scale[bank][lane] = static_cast<std::uint16_t>(
                    (std::uint16_t(inputs_.stream_i[kScaleHiStream][lane]) << 8)
                    | inputs_.stream_i[kScaleLoStream][lane]);
                next.zero[bank][lane] = static_cast<std::int8_t>(
                    inputs_.stream_i[kZeroStream][lane]);
            }
        }
        if (!inputs_.run_i) {
            next.valid = {};
            next.buffer = {};
            next.fault = false;
            return;
        }
        next.valid = {observed.accepted_o, current.valid[0]};
        next.buffer = {inputs_.buffer_i, current.buffer[0]};
        if ((inputs_.request_i && !observed.accepted_o)
            || (inputs_.param_write_i && !observed.param_accepted_o))
            next.fault = true;
        if (observed.accepted_o) {
            const auto bank = static_cast<std::size_t>(inputs_.param_buffer_i);
            for (std::size_t lane = 0; lane < Lanes; ++lane) {
                next.d1_weight[lane] = mxm_numeric::int8_dequant_bf16(
                    static_cast<std::int8_t>(inputs_.stream_i[kQStream][lane]),
                    current.zero[bank][lane], current.scale[bank][lane]);
            }
        }
        if (current.valid[0]) next.weight = current.d1_weight;
    }

    void validate_posedge() const override { state_.validate_commit(); }
    void posedge() override { state_.commit(); }

private:
    struct State {
        std::array<bool, 2> param_ready{};
        std::array<Weight, 2> scale{};
        std::array<std::array<std::int8_t, Lanes>, 2> zero{};
        std::array<bool, 2> valid{};
        std::array<bool, 2> buffer{};
        Weight d1_weight{};
        Weight weight{};
        bool fault{};
    };
    Inputs inputs_{};
    timing::ClockedState<State> state_{};
};

template <std::size_t Terms = 8, std::size_t Groups = 8>
class WeightBufferPipe final : public timing::ClockedModule {
public:
    static_assert(Terms > 0 && Groups > 0);
    using WeightGroup = std::array<std::uint16_t, Terms>;
    using WeightGroups = std::array<WeightGroup, Groups>;

    struct Inputs {
        bool rst_ni{true};
        bool activation_valid_i{};
        WeightGroup activation_i{};
        bool weight_buffer_i{};
        std::uint16_t tag_i{};
        bool weight_write_valid_i{};
        bool weight_write_buffer_i{};
        WeightGroup weight_write_data_i{};
    };
    struct Outputs {
        bool weight_write_accepted_o{};
        bool weight_forward_valid_o{};
        bool weight_forward_buffer_o{};
        WeightGroup weight_forward_data_o{};
        std::array<bool, 2> weight_buffer_ready_o{};
        std::array<bool, 2> weight_buffer_busy_o{};
        bool activation_valid_o{};
        WeightGroup activation_o{};
        bool weight_buffer_o{};
        std::uint16_t activation_tag_o{};
        bool fault_o{};
        std::array<std::array<bool, Groups>, 2> weight_valid_q{};
        std::array<WeightGroups, 2> weights_q{};
    };

    explicit WeightBufferPipe(bool last_column = false)
        : last_column_(last_column)
    {}
    void drive(Inputs inputs) { inputs_ = std::move(inputs); }
    [[nodiscard]] const Inputs& inputs() const noexcept { return inputs_; }

    [[nodiscard]] Outputs outputs() const noexcept
    {
        const auto& current = state_.current();
        Outputs result{};
        for (std::size_t bank = 0; bank < 2; ++bank) {
            result.weight_buffer_ready_o[bank] = true;
            for (const bool valid : current.valid[bank])
                result.weight_buffer_ready_o[bank]
                    = result.weight_buffer_ready_o[bank] && valid;
        }
        if (inputs_.activation_valid_i)
            result.weight_buffer_busy_o[static_cast<std::size_t>(
                inputs_.weight_buffer_i)] = true;
        result.weight_write_accepted_o = inputs_.weight_write_valid_i
            && !result.weight_buffer_busy_o[static_cast<std::size_t>(
                inputs_.weight_write_buffer_i)];
        result.weight_forward_valid_o = current.forward_valid;
        result.weight_forward_buffer_o = current.forward_buffer;
        result.weight_forward_data_o = current.forward_data;
        result.activation_valid_o = current.activation_valid;
        result.activation_o = current.activation;
        result.weight_buffer_o = current.activation_buffer;
        result.activation_tag_o = current.activation_tag;
        result.fault_o = current.fault;
        result.weight_valid_q = current.valid;
        result.weights_q = current.weights;
        return result;
    }

    void low() override
    {
        state_.prepare();
        auto& next = state_.next_value();
        const auto& current = state_.current();
        if (!inputs_.rst_ni) {
            next = {};
            return;
        }
        const auto observed = outputs();
        next.forward_valid = !last_column_ && observed.weight_write_accepted_o
            && current.valid[static_cast<std::size_t>(
                inputs_.weight_write_buffer_i)].back();
        if (next.forward_valid) {
            next.forward_buffer = inputs_.weight_write_buffer_i;
            next.forward_data = current.weights[static_cast<std::size_t>(
                inputs_.weight_write_buffer_i)].back();
        }
        next.activation_valid = inputs_.activation_valid_i;
        if (inputs_.activation_valid_i) {
            next.activation_buffer = inputs_.weight_buffer_i;
            next.activation = inputs_.activation_i;
            next.activation_tag = inputs_.tag_i;
        }
        if (observed.weight_write_accepted_o) {
            const auto bank = static_cast<std::size_t>(inputs_.weight_write_buffer_i);
            const auto shifted = shift_weight_buffer(
                current.weights[bank], current.valid[bank],
                inputs_.weight_write_data_i);
            next.weights[bank] = shifted.data;
            next.valid[bank] = shifted.valid;
        }
        if (inputs_.weight_write_valid_i && !observed.weight_write_accepted_o)
            next.fault = true;
    }

    void validate_posedge() const override { state_.validate_commit(); }
    void posedge() override { state_.commit(); }

private:
    struct State {
        std::array<WeightGroups, 2> weights{};
        std::array<std::array<bool, Groups>, 2> valid{};
        bool forward_valid{};
        bool forward_buffer{};
        WeightGroup forward_data{};
        bool activation_valid{};
        WeightGroup activation{};
        bool activation_buffer{};
        std::uint16_t activation_tag{};
        bool fault{};
    };
    bool last_column_{};
    Inputs inputs_{};
    timing::ClockedState<State> state_{};
};

template <std::size_t Groups = 8>
class SupercellPipe final : public timing::ClockedModule {
public:
    static_assert(Groups > 0);
    using Bf16Vector = Bf16DotCsaPipe::Bf16Vector;
    using WeightBuffer = WeightBufferPipe<Bf16DotCsaPipe::kTerms, Groups>;

    struct Inputs {
        bool rst_ni{true};
        bool activation_valid_i{};
        Bf16Vector activation_i{};
        bool weight_buffer_i{};
        std::uint16_t tag_i{};
        bool weight_write_valid_i{};
        bool weight_write_buffer_i{};
        Bf16Vector weight_write_data_i{};
        std::array<bool, Groups> below_valid_i{};
        std::array<std::uint32_t, Groups> below_partial_i{};
        std::array<std::uint16_t, Groups> below_tag_i{};
    };

    struct Outputs {
        bool weight_write_accepted_o{};
        bool weight_forward_valid_o{};
        bool weight_forward_buffer_o{};
        Bf16Vector weight_forward_data_o{};
        std::array<bool, 2> weight_buffer_ready_o{};
        std::array<bool, 2> weight_buffer_busy_o{};
        bool activation_valid_o{};
        Bf16Vector activation_o{};
        bool weight_buffer_o{};
        std::uint16_t activation_tag_o{};
        std::array<bool, Groups> dot_valid_o{};
        std::array<std::uint32_t, Groups> dot_result_o{};
        std::array<std::uint16_t, Groups> dot_tag_o{};
        std::array<bool, Groups> alignment_lost_o{};
        std::array<bool, Groups> partial_valid_o{};
        std::array<std::uint32_t, Groups> partial_o{};
        std::array<std::uint16_t, Groups> partial_tag_o{};
        bool fault_o{};
        std::array<bool, Groups> input_weight_fault{};
        std::array<bool, Groups> partial_protocol_fault{};
        std::array<std::array<bool, Groups>, 2> weight_valid_q{};
        std::array<typename WeightBuffer::WeightGroups, 2> weights_q{};
    };

    explicit SupercellPipe(bool bottom = true, bool last_column = false)
        : bottom_(bottom), weight_buffer_(last_column)
    {}

    void drive(Inputs inputs) { inputs_ = std::move(inputs); }
    [[nodiscard]] const Inputs& inputs() const noexcept { return inputs_; }

    [[nodiscard]] Outputs outputs() const noexcept
    {
        Outputs result{};
        const auto weight = weight_buffer_.outputs();
        result.weight_write_accepted_o = weight.weight_write_accepted_o;
        result.weight_forward_valid_o = weight.weight_forward_valid_o;
        result.weight_forward_buffer_o = weight.weight_forward_buffer_o;
        result.weight_forward_data_o = weight.weight_forward_data_o;
        result.weight_buffer_ready_o = weight.weight_buffer_ready_o;
        result.weight_buffer_busy_o = weight.weight_buffer_busy_o;
        result.activation_valid_o = weight.activation_valid_o;
        result.activation_o = weight.activation_o;
        result.weight_buffer_o = weight.weight_buffer_o;
        result.activation_tag_o = weight.activation_tag_o;
        result.weight_valid_q = weight.weight_valid_q;
        result.weights_q = weight.weights_q;
        result.fault_o = fault_.current() || weight.fault_o;
        const auto bank = static_cast<std::size_t>(inputs_.weight_buffer_i);
        for (std::size_t group = 0; group < Groups; ++group) {
            const auto dot = dots_[group].outputs();
            const auto add = partial_adds_[group].outputs();
            result.input_weight_fault[group] = inputs_.activation_valid_i
                && !weight.weight_valid_q[bank][group];
            if (!bottom_) {
                result.partial_protocol_fault[group]
                    = dot.valid_o != inputs_.below_valid_i[group]
                    || (dot.valid_o && inputs_.below_valid_i[group]
                        && dot.tag_o != inputs_.below_tag_i[group]);
            }
            result.dot_valid_o[group] = dot.valid_o;
            result.dot_result_o[group] = dot.result_o;
            result.dot_tag_o[group] = dot.tag_o;
            result.alignment_lost_o[group] = dot.alignment_lost_o;
            result.partial_valid_o[group] = add.valid_o;
            result.partial_o[group] = add.result_o;
            result.partial_tag_o[group] = add.tag_o;
        }
        return result;
    }

    void low() override
    {
        typename WeightBuffer::Inputs weight_inputs{};
        weight_inputs.rst_ni = inputs_.rst_ni;
        weight_inputs.activation_valid_i = inputs_.activation_valid_i;
        weight_inputs.activation_i = inputs_.activation_i;
        weight_inputs.weight_buffer_i = inputs_.weight_buffer_i;
        weight_inputs.tag_i = inputs_.tag_i;
        weight_inputs.weight_write_valid_i = inputs_.weight_write_valid_i;
        weight_inputs.weight_write_buffer_i = inputs_.weight_write_buffer_i;
        weight_inputs.weight_write_data_i = inputs_.weight_write_data_i;
        weight_buffer_.drive(weight_inputs);
        weight_buffer_.low();

        const auto before = outputs();
        const auto bank = static_cast<std::size_t>(inputs_.weight_buffer_i);
        for (std::size_t group = 0; group < Groups; ++group) {
            Bf16DotCsaPipe::Inputs dot_inputs{};
            dot_inputs.rst_ni = inputs_.rst_ni;
            dot_inputs.valid_i = inputs_.activation_valid_i
                && !before.input_weight_fault[group];
            dot_inputs.activation_i = inputs_.activation_i;
            dot_inputs.weight_i = before.weights_q[bank][group];
            dot_inputs.tag_i = inputs_.tag_i;
            dots_[group].drive(dot_inputs);
            dots_[group].low();

            const auto dot = dots_[group].outputs();
            const bool partial_fault = !bottom_
                && (dot.valid_o != inputs_.below_valid_i[group]
                    || (dot.valid_o && inputs_.below_valid_i[group]
                        && dot.tag_o != inputs_.below_tag_i[group]));
            Fp32AddPipe::Inputs add_inputs{};
            add_inputs.rst_ni = inputs_.rst_ni;
            add_inputs.valid_i = dot.valid_o && !partial_fault;
            add_inputs.lhs_i = dot.result_o;
            add_inputs.rhs_i = bottom_ ? 0u : inputs_.below_partial_i[group];
            add_inputs.tag_i = dot.tag_o;
            partial_adds_[group].drive(add_inputs);
            partial_adds_[group].low();
        }

        fault_.prepare();
        auto& next_fault = fault_.next_value();
        if (!inputs_.rst_ni) {
            next_fault = false;
        } else {
            for (std::size_t group = 0; group < Groups; ++group)
                if (before.input_weight_fault[group]
                    || before.partial_protocol_fault[group])
                    next_fault = true;
        }
    }

    void validate_posedge() const override
    {
        weight_buffer_.validate_posedge();
        for (const auto& dot : dots_) dot.validate_posedge();
        for (const auto& add : partial_adds_) add.validate_posedge();
        fault_.validate_commit();
    }

    void posedge() override
    {
        weight_buffer_.posedge();
        for (auto& dot : dots_) dot.posedge();
        for (auto& add : partial_adds_) add.posedge();
        fault_.commit();
    }

private:
    Inputs inputs_{};
    bool bottom_{};
    WeightBuffer weight_buffer_;
    std::array<Bf16DotCsaPipe, Groups> dots_{};
    std::array<Fp32AddPipe, Groups> partial_adds_{};
    timing::ClockedState<bool> fault_{};
};

template <typename Token, std::size_t Rows, std::size_t Columns>
inline void advance_result_return(
    std::array<std::array<Token, Columns>, Rows>& next_down,
    std::array<std::array<Token, Columns>, Rows>& next_left,
    const std::array<std::array<Token, Columns>, Rows>& current_down,
    const std::array<std::array<Token, Columns>, Rows>& current_left,
    const std::array<Token, Columns>& top)
{
    static_assert(Rows > 0 && Rows == Columns);
    const auto capture = [](Token& next, const Token& current, const Token& source) {
        next = current;
        next.valid = source.valid;
        if (source.valid) {
            next.data = source.data;
            next.tag = source.tag;
        }
    };
    for (std::size_t row = 0; row < Rows; ++row) {
        for (std::size_t column = 0; column < Columns; ++column) {
            if (row > column) {
                const auto& source = row == Rows - 1
                    ? top[column] : current_down[row + 1][column];
                capture(next_down[row][column], current_down[row][column], source);
            } else {
                next_down[row][column] = Token{};
            }
            if (row < column) {
                next_left[row][column] = Token{};
            } else if (row == column) {
                const auto& source = row == Rows - 1
                    ? top[column] : current_down[row + 1][column];
                capture(next_left[row][column], current_left[row][column], source);
            } else {
                capture(next_left[row][column], current_left[row][column],
                    current_left[row][column + 1]);
            }
        }
    }
}

template <std::size_t Rows = 4, std::size_t Lanes = 8>
class ResultReturnPipe final : public timing::ClockedModule {
public:
    static_assert(Rows > 0 && Lanes > 0);
    using Fp32Vector = std::array<std::uint32_t, Lanes>;
    struct Inputs {
        bool rst_ni{true};
        bool run_i{true};
        std::array<bool, Rows> top_valid_i{};
        std::array<Fp32Vector, Rows> top_i{};
        std::array<std::uint16_t, Rows> top_tag_i{};
    };
    struct Outputs {
        std::array<bool, Rows> left_valid_o{};
        std::array<Fp32Vector, Rows> left_o{};
        std::array<std::uint16_t, Rows> left_tag_o{};
        std::array<std::array<bool, Rows>, Rows> down_valid_o{};
        std::array<std::array<Fp32Vector, Rows>, Rows> down_o{};
        std::array<std::array<std::uint16_t, Rows>, Rows> down_tag_o{};
        std::array<std::array<bool, Rows>, Rows> left_link_valid_o{};
        std::array<std::array<Fp32Vector, Rows>, Rows> left_link_o{};
        std::array<std::array<std::uint16_t, Rows>, Rows> left_link_tag_o{};
    };

    void drive(Inputs inputs) { inputs_ = std::move(inputs); }
    [[nodiscard]] const Inputs& inputs() const noexcept { return inputs_; }
    [[nodiscard]] Outputs outputs() const noexcept
    {
        Outputs result{};
        const auto& current = state_.current();
        for (std::size_t row = 0; row < Rows; ++row) {
            result.left_valid_o[row] = current.left[row][0].valid;
            result.left_o[row] = current.left[row][0].data;
            result.left_tag_o[row] = current.left[row][0].tag;
            for (std::size_t column = 0; column < Rows; ++column) {
                result.down_valid_o[row][column] = current.down[row][column].valid;
                result.down_o[row][column] = current.down[row][column].data;
                result.down_tag_o[row][column] = current.down[row][column].tag;
                result.left_link_valid_o[row][column] = current.left[row][column].valid;
                result.left_link_o[row][column] = current.left[row][column].data;
                result.left_link_tag_o[row][column] = current.left[row][column].tag;
            }
        }
        return result;
    }

    void low() override
    {
        state_.prepare();
        auto& next = state_.next_value();
        const auto& current = state_.current();
        if (!inputs_.rst_ni || !inputs_.run_i) {
            next = {};
            return;
        }
        std::array<Token, Rows> top{};
        for (std::size_t column = 0; column < Rows; ++column) {
            top[column].valid = inputs_.top_valid_i[column];
            if (top[column].valid) {
                top[column].data = inputs_.top_i[column];
                top[column].tag = inputs_.top_tag_i[column];
            }
        }
        advance_result_return(next.down, next.left,
            current.down, current.left, top);
    }

    void validate_posedge() const override { state_.validate_commit(); }
    void posedge() override { state_.commit(); }

private:
    struct Token {
        bool valid{};
        std::uint16_t tag{};
        Fp32Vector data{};
    };
    struct State {
        std::array<std::array<Token, Rows>, Rows> down{};
        std::array<std::array<Token, Rows>, Rows> left{};
    };
    Inputs inputs_{};
    timing::ClockedState<State> state_{};
};

enum class BackendColumnOperation : std::uint8_t {
    DirectOut = 0,
    Store = 1,
    AccStore = 2,
    AccOut = 3,
};

template <std::size_t Lanes = 8, std::size_t Depth = 16>
class BackendColumnPipe final : public timing::ClockedModule {
    static_assert(Lanes > 0 && Depth >= 2 && Depth % 2 == 0);
public:
    using Fp32Vector = std::array<std::uint32_t, Lanes>;
    using Operation = BackendColumnOperation;
    static constexpr std::size_t kBankDepth = Depth / 2;

    struct Inputs {
        bool rst_ni{true};
        bool run_i{true};
        bool slot_valid_i{};
        Operation operation_i{Operation::DirectOut};
        bool output_bf16_i{};
        std::uint16_t address_i{};
        bool partial_valid_i{};
        Fp32Vector partial_i{};
        std::uint16_t partial_tag_i{};
    };
    struct Outputs {
        bool slot_valid_o{};
        bool slot_bf16_o{};
        Operation slot_operation_o{Operation::DirectOut};
        std::uint16_t slot_address_o{};
        bool result_valid_o{};
        bool result_bf16_o{};
        Fp32Vector result_o{};
        std::uint16_t result_address_o{};
        std::uint16_t result_tag_o{};
        std::array<bool, 2> bank_request_valid_o{};
        std::array<bool, 2> bank_write_o{};
        std::array<std::uint16_t, 2> bank_address_o{};
        std::array<Fp32Vector, 2> bank_write_data_o{};
        bool alu_valid_o{};
        Fp32Vector operand_a_o{};
        Fp32Vector operand_b_o{};
        std::uint16_t alu_address_o{};
        std::uint16_t alu_tag_o{};
        bool fault_o{};
        std::array<bool, 2> context_valid_q{};
        std::array<bool, 2> context_ok_q{};
        std::array<bool, 3> bypass_valid_q{};
        std::array<bool, 3> sum_valid_q{};
        std::array<bool, 5> pending_valid_q{};
        std::array<bool, 2> bank_capture_valid_q{};
        std::array<bool, 2> bank_read_valid_o{};
        std::array<Fp32Vector, 2> bank_read_data_o{};
        std::array<std::uint16_t, 2> bank_read_address_o{};
        std::array<bool, 2> invalid_read_o{};
        bool accepted_read{};
    };

    void drive(Inputs inputs) { inputs_ = std::move(inputs); }
    [[nodiscard]] const Inputs& inputs() const noexcept { return inputs_; }

    [[nodiscard]] Outputs outputs() const noexcept
    {
        Outputs out{};
        const auto& current = state_.current();
        const auto& slot = current.context[1];
        out.slot_valid_o = inputs_.run_i && slot.valid;
        out.slot_bf16_o = slot.bf16;
        out.slot_operation_o = slot.operation;
        out.slot_address_o = slot.address;
        out.context_valid_q = {current.context[0].valid, current.context[1].valid};
        out.context_ok_q = {current.context[0].ok, current.context[1].ok};
        for (std::size_t index = 0; index < 3; ++index) {
            out.bypass_valid_q[index] = current.bypass[index].valid;
            out.sum_valid_q[index] = current.sum[index].valid;
        }
        out.pending_valid_q = current.pending_valid;

        const auto& sum = current.sum.back();
        const auto& bypass = current.bypass.back();
        const bool accumulated_write = sum.valid
            && sum.operation == Operation::AccStore;
        const bool direct_write = bypass.valid
            && bypass.operation == Operation::Store;
        const bool writeback_valid = accumulated_write || direct_write;
        const auto& writeback = accumulated_write ? sum : bypass;
        const bool first_read = inputs_.slot_valid_i && operation_reads(inputs_.operation_i);
        bool address_hazard = false;
        for (std::size_t index = 0; index < current.pending_valid.size(); ++index)
            address_hazard = address_hazard
                || (current.pending_valid[index]
                    && current.pending_address[index] == inputs_.address_i
                    && (first_read || operation_writes(inputs_.operation_i)));
        const bool bank_collision = first_read && writeback_valid
            && ((inputs_.address_i ^ writeback.address) & 1u) == 0;
        const bool address_in_range = inputs_.address_i < Depth;
        const bool first_ok = inputs_.run_i && !bank_collision
            && !address_hazard && address_in_range;
        const auto issue_bank = static_cast<std::size_t>(inputs_.address_i & 1u);
        const auto issue_row = static_cast<std::size_t>(inputs_.address_i >> 1u);
        for (std::size_t bank = 0; bank < 2; ++bank) {
            const auto& bank_state = current.banks[bank];
            out.bank_capture_valid_q[bank] = bank_state.capture_valid;
            out.bank_read_valid_o[bank] = bank_state.read_valid;
            out.bank_read_data_o[bank] = bank_state.read_data;
            out.bank_read_address_o[bank] = bank_state.read_address;
            const bool read_here = first_read && first_ok && issue_bank == bank;
            const bool write_here = writeback_valid && !bank_collision
                && ((writeback.address & 1u) == bank);
            out.bank_request_valid_o[bank] = read_here || write_here;
            out.bank_write_o[bank] = write_here;
            out.bank_address_o[bank] = static_cast<std::uint16_t>(
                (write_here ? writeback.address : inputs_.address_i) >> 1u);
            out.bank_write_data_o[bank] = writeback.data;
            out.invalid_read_o[bank] = inputs_.run_i && read_here
                && (issue_row >= kBankDepth || !initialized_[bank][issue_row]);
        }
        out.accepted_read = first_read && first_ok && !out.invalid_read_o[issue_bank];

        const auto slot_bank = static_cast<std::size_t>(slot.address & 1u);
        const auto& read = current.banks[slot_bank];
        const bool response_match = read.read_valid
            && read.read_address == (slot.address >> 1u);
        out.operand_a_o = inputs_.partial_i;
        out.operand_b_o = read.read_data;
        out.alu_address_o = slot.address;
        out.alu_tag_o = inputs_.partial_tag_i;
        out.alu_valid_o = inputs_.run_i && slot.valid && slot.ok
            && inputs_.partial_valid_i && operation_reads(slot.operation)
            && response_match;

        out.result_valid_o = inputs_.run_i && current.output.valid;
        out.result_bf16_o = current.output.bf16;
        out.result_o = current.output.data;
        out.result_address_o = current.output.address;
        out.result_tag_o = current.output.tag;
        out.fault_o = current.fault;
        return out;
    }

    void low() override
    {
        state_.prepare();
        auto& next = state_.next_value();
        const auto& current = state_.current();
        clear_initialized_ = false;
        pending_write_ = {};
        if (!inputs_.rst_ni) {
            next = {};
            clear_initialized_ = true;
            return;
        }
        if (!inputs_.run_i) {
            next = current;
            for (auto& context : next.context) {
                context.valid = false;
                context.ok = false;
            }
            for (auto& token : next.bypass) token.valid = false;
            next.sum = {};
            next.pending_valid = {};
            for (auto& bank : next.banks) {
                bank.capture_valid = false;
                bank.read_valid = false;
            }
            next.output = {};
            next.fault = false;
            return;
        }

        const auto observed = outputs();
        for (std::size_t bank = 0; bank < 2; ++bank) {
            auto& next_bank = next.banks[bank];
            const auto& bank_state = current.banks[bank];
            next_bank.capture_valid = observed.bank_request_valid_o[bank]
                && !observed.bank_write_o[bank] && !observed.invalid_read_o[bank];
            next_bank.read_valid = bank_state.capture_valid;
            if (bank_state.capture_valid) {
                next_bank.read_data = bank_state.capture_data;
                next_bank.read_address = bank_state.capture_address;
            }
            if (next_bank.capture_valid) {
                const auto row = static_cast<std::size_t>(observed.bank_address_o[bank]);
                next_bank.capture_data = memory_[bank][row];
                next_bank.capture_address = observed.bank_address_o[bank];
            }
            if (observed.bank_request_valid_o[bank] && observed.bank_write_o[bank]) {
                pending_write_.valid = true;
                pending_write_.bank = bank;
                pending_write_.row = observed.bank_address_o[bank];
                pending_write_.data = observed.bank_write_data_o[bank];
            }
        }

        ContextToken context_input{};
        context_input.valid = inputs_.slot_valid_i;
        const bool first_ok = inputs_.run_i && inputs_.address_i < Depth
            && !bank_collision(current) && !has_address_hazard(current);
        context_input.ok = inputs_.slot_valid_i && (
            ((inputs_.operation_i == Operation::DirectOut
                || inputs_.operation_i == Operation::Store)
                && first_ok)
            || observed.accepted_read);
        context_input.bf16 = inputs_.output_bf16_i;
        context_input.operation = inputs_.operation_i;
        context_input.address = inputs_.address_i;
        shift_pipeline(next.context, current.context, context_input);

        const auto& slot = current.context[1];
        const auto slot_bank = static_cast<std::size_t>(slot.address & 1u);
        const auto& read = current.banks[slot_bank];
        const bool response_match = read.read_valid
            && read.read_address == (slot.address >> 1u);
        const bool alu_valid = slot.valid && slot.ok && inputs_.partial_valid_i
            && operation_reads(slot.operation) && response_match;
        DataToken sum_input{};
        sum_input.valid = alu_valid;
        sum_input.bf16 = slot.bf16;
        sum_input.operation = slot.operation;
        sum_input.address = slot.address;
        sum_input.tag = inputs_.partial_tag_i;
        if (alu_valid)
            for (std::size_t lane = 0; lane < Lanes; ++lane)
                sum_input.data[lane] = mxm_numeric::fp32_add_bits(
                    inputs_.partial_i[lane], read.read_data[lane]);
        shift_pipeline(next.sum, current.sum, sum_input);

        DataToken bypass_input{};
        bypass_input.valid = slot.valid && slot.ok && inputs_.partial_valid_i
            && !operation_reads(slot.operation);
        bypass_input.bf16 = slot.bf16;
        bypass_input.operation = slot.operation;
        bypass_input.address = slot.address;
        bypass_input.tag = inputs_.partial_tag_i;
        bypass_input.data = inputs_.partial_i;
        shift_pipeline(next.bypass, current.bypass, bypass_input);

        const auto& sum = current.sum.back();
        const auto& bypass = current.bypass.back();
        const bool accumulated_output = sum.valid
            && sum.operation == Operation::AccOut;
        const bool direct_output = bypass.valid
            && bypass.operation == Operation::DirectOut;
        const auto& output_source = accumulated_output ? sum : bypass;
        next.output.valid = accumulated_output || direct_output;
        next.output.bf16 = next.output.valid && output_source.bf16;
        if (next.output.valid) {
            next.output = output_source;
            next.output.valid = true;
            next.output.bf16 = output_source.bf16;
            if (output_source.bf16)
                for (auto& word : next.output.data)
                    word = mxm_numeric::fp32_to_bf16(word);
        }

        const bool scheduled_write = inputs_.slot_valid_i
            && (inputs_.operation_i == Operation::Store
                || inputs_.operation_i == Operation::AccStore);
        next.pending_valid[0] = scheduled_write && first_ok
            && (inputs_.operation_i == Operation::Store || observed.accepted_read);
        next.pending_address[0] = inputs_.address_i;
        for (std::size_t index = 1; index < next.pending_valid.size(); ++index) {
            next.pending_valid[index] = current.pending_valid[index - 1];
            next.pending_address[index] = current.pending_address[index - 1];
        }

        const bool response_mismatch = slot.valid && slot.ok
            && operation_reads(slot.operation) && !response_match;
        const bool address_hazard = has_address_hazard(current);
        next.fault = current.fault
            || (slot.valid != inputs_.partial_valid_i)
            || bank_collision(current) || address_hazard
            || observed.invalid_read_o[0] || observed.invalid_read_o[1]
            || response_mismatch || (inputs_.slot_valid_i && inputs_.address_i >= Depth)
            || (inputs_.slot_valid_i && inputs_.output_bf16_i
                && inputs_.operation_i != Operation::DirectOut
                && inputs_.operation_i != Operation::AccOut);
    }

    void validate_posedge() const override
    {
        state_.validate_commit();
        if (pending_write_.valid && pending_write_.row >= kBankDepth)
            throw std::logic_error("MXM accumulator write outside bank depth");
    }

    void posedge() override
    {
        state_.commit();
        if (clear_initialized_)
            for (auto& bank : initialized_) bank.fill(false);
        if (pending_write_.valid) {
            memory_[pending_write_.bank][pending_write_.row] = pending_write_.data;
            initialized_[pending_write_.bank][pending_write_.row] = true;
        }
        clear_initialized_ = false;
        pending_write_ = {};
    }

private:
    struct ContextToken {
        bool valid{};
        bool ok{};
        bool bf16{};
        Operation operation{Operation::DirectOut};
        std::uint16_t address{};
    };
    struct DataToken {
        bool valid{};
        bool bf16{};
        Operation operation{Operation::DirectOut};
        std::uint16_t address{};
        std::uint16_t tag{};
        Fp32Vector data{};
    };
    struct BankState {
        bool capture_valid{};
        Fp32Vector capture_data{};
        std::uint16_t capture_address{};
        bool read_valid{};
        Fp32Vector read_data{};
        std::uint16_t read_address{};
    };
    struct State {
        std::array<ContextToken, 2> context{};
        std::array<DataToken, 3> bypass{};
        std::array<DataToken, 3> sum{};
        std::array<bool, 5> pending_valid{};
        std::array<std::uint16_t, 5> pending_address{};
        std::array<BankState, 2> banks{};
        DataToken output{};
        bool fault{};
    };
    struct PendingWrite {
        bool valid{};
        std::size_t bank{};
        std::size_t row{};
        Fp32Vector data{};
    };

    static bool operation_reads(Operation operation) noexcept
    { return operation == Operation::AccStore || operation == Operation::AccOut; }
    static bool operation_writes(Operation operation) noexcept
    { return operation == Operation::Store || operation == Operation::AccStore; }
    bool has_address_hazard(const State& current) const noexcept
    {
        const bool first_read = inputs_.slot_valid_i && operation_reads(inputs_.operation_i);
        for (std::size_t index = 0; index < current.pending_valid.size(); ++index)
            if (current.pending_valid[index]
                && current.pending_address[index] == inputs_.address_i
                && (first_read || operation_writes(inputs_.operation_i))) return true;
        return false;
    }
    bool bank_collision(const State& current) const noexcept
    {
        const auto& sum = current.sum.back();
        const auto& bypass = current.bypass.back();
        const bool accumulated_write = sum.valid && sum.operation == Operation::AccStore;
        const bool direct_write = bypass.valid && bypass.operation == Operation::Store;
        const auto& writeback = accumulated_write ? sum : bypass;
        return inputs_.slot_valid_i && operation_reads(inputs_.operation_i)
            && (accumulated_write || direct_write)
            && ((inputs_.address_i ^ writeback.address) & 1u) == 0;
    }

    Inputs inputs_{};
    timing::ClockedState<State> state_{};
    std::array<std::array<Fp32Vector, kBankDepth>, 2> memory_{};
    std::array<std::array<bool, kBankDepth>, 2> initialized_{};
    PendingWrite pending_write_{};
    bool clear_initialized_{};
};

template <std::size_t Lanes = 8, typename Meta = std::uint32_t>
class OutputPipe final : public timing::ClockedModule {
public:
    static_assert(Lanes > 0);
    struct Inputs {
        bool rst_ni{true};
        bool valid_i{};
        bool bf16_i{};
        std::array<std::uint32_t, Lanes> data_i{};
        Meta meta_i{};
    };
    struct Outputs {
        bool valid_o{};
        bool bf16_o{};
        std::array<std::uint32_t, Lanes> data_o{};
        Meta meta_o{};
    };

    void drive(Inputs inputs) { inputs_ = std::move(inputs); }
    [[nodiscard]] const Inputs& inputs() const noexcept { return inputs_; }
    [[nodiscard]] Outputs outputs() const noexcept
    {
        const auto& current = state_.current();
        return Outputs{current.valid, current.bf16, current.data, current.meta};
    }

    void low() override
    {
        state_.prepare();
        auto& next = state_.next_value();
        if (!inputs_.rst_ni) {
            next = {};
            return;
        }
        next.valid = inputs_.valid_i;
        next.bf16 = inputs_.valid_i && inputs_.bf16_i;
        if (inputs_.valid_i) {
            next.meta = inputs_.meta_i;
            for (std::size_t lane = 0; lane < Lanes; ++lane) {
                next.data[lane] = inputs_.bf16_i
                    ? static_cast<std::uint32_t>(
                        mxm_numeric::fp32_to_bf16(inputs_.data_i[lane]))
                    : inputs_.data_i[lane];
            }
        }
    }

    void validate_posedge() const override { state_.validate_commit(); }
    void posedge() override { state_.commit(); }

private:
    struct State {
        bool valid{};
        bool bf16{};
        std::array<std::uint32_t, Lanes> data{};
        Meta meta{};
    };
    Inputs inputs_{};
    timing::ClockedState<State> state_{};
};

} // namespace ftlpu::mxm_timing
