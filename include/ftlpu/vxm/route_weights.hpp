#pragma once

#include "ftlpu/core/bf16.hpp"
#include "ftlpu/vxm/special_alu.hpp"
#include "ftlpu/vxm/topk_instruction.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace ftlpu {

// Scalar VXM control port for route weights. It uses the regular VXM LUT
// contents and its registered lookup stages. At most one request is in flight.
class VxmRouteWeightArithmetic {
public:
    void begin_softmax(std::uint32_t max_bits)
    {
        require_idle();
        maximum_ = std::bit_cast<float>(max_bits);
        if (!std::isfinite(maximum_))
            throw std::invalid_argument("selected softmax requires finite logits");
        sum_ = 0;
    }

    void selected_exp(std::uint32_t bits, unsigned slot)
    {
        submit(VxmSpecialAluOpcode::Exp, std::bit_cast<float>(bits) - maximum_,
               {Operation::SelectedExp, slot});
    }

    void selected_reciprocal()
    {
        submit(VxmSpecialAluOpcode::Reciprocal, sum_, {Operation::SelectedInverse, 0});
    }

    std::uint32_t selected_weight(unsigned slot) const
    {
        return weight_bits(values_.at(slot) * inverse_);
    }

    void shared_exp(std::uint32_t bits)
    {
        const auto logit = Bf16::from_float(std::bit_cast<float>(bits)).to_float();
        submit(VxmSpecialAluOpcode::Exp, -logit, {Operation::SharedExp, 0});
    }

    void shared_reciprocal()
    {
        submit(VxmSpecialAluOpcode::Reciprocal, 1.0f + shared_, {Operation::SharedInverse, 0});
    }

    std::uint32_t shared_weight() const { return weight_bits(shared_); }
    bool idle() const { return !request_ && lookup_.empty() && !ready_; }

    bool take_ready()
    {
        const bool ready = ready_;
        ready_ = false;
        return ready;
    }

    void tick(const VxmSpecialAlu& alu)
    {
        const auto result = lookup_.tick(alu, request_);
        request_.reset();
        if (!result) return;
        const auto& [op, slot] = result->metadata;
        switch (op) {
        case Operation::SelectedExp:
            values_.at(slot) = result->value;
            sum_ += result->value;
            break;
        case Operation::SelectedInverse:
            inverse_ = result->value;
            break;
        case Operation::SharedExp:
        case Operation::SharedInverse:
            shared_ = result->value;
            break;
        }
        ready_ = true;
    }

private:
    enum class Operation { SelectedExp, SelectedInverse, SharedExp, SharedInverse };
    struct Tag { Operation op{}; unsigned slot{}; };
    using Lookup = VxmSpecialAlu::Pipeline<Tag>;

    void require_idle() const
    {
        if (!idle())
            throw std::logic_error("VXM route weight lookup still in flight");
    }

    void submit(VxmSpecialAluOpcode opcode, float value, Tag tag)
    {
        require_idle();
        request_ = Lookup::Request{opcode, value, tag};
    }

    static std::uint32_t weight_bits(float value)
    {
        return std::bit_cast<std::uint32_t>(Bf16::from_float(value).to_float());
    }

    Lookup lookup_;
    std::optional<Lookup::Request> request_;
    std::array<float, VxmTopKInstruction::max_k> values_{};
    float maximum_{}, sum_{}, inverse_{}, shared_{};
    bool ready_{};
};
} // namespace ftlpu
