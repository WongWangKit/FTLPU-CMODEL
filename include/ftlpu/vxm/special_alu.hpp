#pragma once

#include "ftlpu/vxm/data_format.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ftlpu {

enum class VxmSpecialAluOpcode {
    Exp,
    Reciprocal,
    Rsqrt,
};

struct VxmLutEntry {
    std::uint16_t k_bits{0};
    std::uint16_t b_bits{0};

    static VxmLutEntry from_float(float k, float b)
    {
        return {VxmDataFormat::float_to_fp16_bits(k),
                VxmDataFormat::float_to_fp16_bits(b)};
    }

    float k() const { return VxmDataFormat::fp16_bits_to_float(k_bits); }
    float b() const { return VxmDataFormat::fp16_bits_to_float(b_bits); }
};

struct VxmLutConfig {
    float input_min{0.0f};
    float segment_width{1.0f};
};

struct VxmLutLookupRequest {
    VxmSpecialAluOpcode opcode{VxmSpecialAluOpcode::Exp};
    std::size_t bank{0};
    std::size_t index{0};
    float dx{0.0f};
    float multiplier{1.0f};
    int result_exponent{0};
    bool lookup_valid{true};
    float direct_result{0.0f};
};//请求的信息数据结构

struct VxmLutReadResult {
    VxmLutLookupRequest request{};
    float k{0.0f};
    float b{0.0f};
};

struct VxmLutLinearResult {
    VxmLutLookupRequest request{};
    float local{0.0f};
};

// One instance represents the shared LUT SRAM and its address-generation
// hardware.  Requests contain all lane-local state, so lanes may use different
// indices in the same cycle.  SRAM read-port conflicts are intentionally not
// timed in this functional C model.
class VxmSpecialAlu {
public:
    static constexpr std::size_t kBankCount = 3;
    static constexpr std::size_t kPipelineLatency = 5;
    static constexpr std::size_t kInitiationInterval = 1;
//对SRAM进行config
    void configure_lut(VxmSpecialAluOpcode opcode, VxmLutConfig config,
                       std::vector<VxmLutEntry> entries)
    {
        if (!(config.segment_width > 0.0f) || entries.empty()) {
            throw std::invalid_argument("VXM LUT requires positive width and non-empty entries");
        }
        auto& bank = banks_[bank_for_opcode(opcode)];
        bank.config = config;
        bank.entries = std::move(entries);
        bank.configured = true;
    }

    bool configured(VxmSpecialAluOpcode opcode) const
    {
        return banks_[bank_for_opcode(opcode)].configured;
    }

    VxmLutLookupRequest make_lookup(VxmSpecialAluOpcode opcode, float input) const
    {
        input = VxmDataFormat::round_fp16_ftz(input);
        auto request = VxmLutLookupRequest {};
        request.opcode = opcode;
        request.bank = bank_for_opcode(opcode);

        if (opcode == VxmSpecialAluOpcode::Exp) {
            if (std::isnan(input)) return direct(request, input);
            if (input == std::numeric_limits<float>::infinity()) return direct(request, input);
            if (input == -std::numeric_limits<float>::infinity()) return direct(request, 0.0f);
            constexpr float kInvLn2 = 1.4426950408889634f;
            constexpr float kLn2 = 0.6931471805599453f;
            const auto exponent = static_cast<int>(std::nearbyint(input * kInvLn2));//乘以倒数
            request.result_exponent = exponent;
            return address(request, input - static_cast<float>(exponent) * kLn2);
        }

        if (opcode == VxmSpecialAluOpcode::Reciprocal) {
            if (std::isnan(input)) return direct(request, input);
            if (input == 0.0f) {
                return direct(request, 1.0f / input);
            }
            if (std::isinf(input)) {
                return direct(request, 1.0f / input);
            }
            int exponent = 0;
            const auto mantissa = std::frexp(std::fabs(input), &exponent) * 2.0f;
            request.result_exponent = -(exponent - 1);
            request.multiplier = std::copysign(1.0f, input);
            return address(request, mantissa);
        }

        if (std::isnan(input) || input < 0.0f) {
            return direct(request, std::numeric_limits<float>::quiet_NaN());
        }
        if (input == 0.0f) {
            return direct(request, std::numeric_limits<float>::infinity());
        }
        if (std::isinf(input)) {
            return direct(request, 0.0f);
        }
        int exponent = 0;
        auto mantissa = std::frexp(input, &exponent) * 2.0f;
        --exponent;
        if ((exponent & 1) != 0) {
            mantissa *= 2.0f;
            --exponent;
        }
        request.result_exponent = -exponent / 2;
        return address(request, mantissa);
    }

    VxmLutReadResult read_lut(const VxmLutLookupRequest& request) const
    {
        if (!request.lookup_valid) {
            return {request, 0.0f, request.direct_result};
        }
        const auto& bank = banks_[request.bank];
        if (!bank.configured) {
            throw std::logic_error("VXM special ALU LUT bank is not configured");
        }
        const auto& entry = bank.entries.at(request.index);
        return {request, entry.k(), entry.b()};
    }

    VxmLutLinearResult linear(const VxmLutReadResult& read) const
    {
        if (!read.request.lookup_valid) {
            return {read.request, read.request.direct_result};
        }
        return {read.request, std::fma(read.k, read.request.dx, read.b)};
    }

    float restore(const VxmLutLinearResult& linear_result) const
    {
        if (!linear_result.request.lookup_valid) {
            return VxmDataFormat::round_fp16_ftz(
                linear_result.request.direct_result);
        }
        const auto restored = std::ldexp(
            linear_result.local * linear_result.request.multiplier,
            linear_result.request.result_exponent);
        return VxmDataFormat::round_fp16_ftz(restored);
    }

    float complete_lookup(const VxmLutLookupRequest& request) const
    {
        return restore(linear(read_lut(request)));
    }

    float execute(VxmSpecialAluOpcode opcode, float input) const
    {
        return complete_lookup(make_lookup(opcode, input));
    }

    // Five registered stages: address, SRAM, two FMA stages, and restore.
    // Latency is five cycles; a new lookup may enter every cycle.
    template<typename Metadata>
    class Pipeline {
    public:
        struct Request {
            VxmSpecialAluOpcode opcode{VxmSpecialAluOpcode::Exp};
            float input{0.0f};
            Metadata metadata{};
        };

        struct Result {
            float value{0.0f};
            Metadata metadata{};
        };

        std::optional<Result> tick(
            const VxmSpecialAlu& alu,
            std::optional<Request> input = std::nullopt)
        {
            auto output = std::optional<Result>{};
            restore_active_ = linear_stage_.has_value();
            if (linear_stage_) {
                output = Result{
                    alu.restore(linear_stage_->value),
                    std::move(linear_stage_->metadata)};
                linear_stage_.reset();
            }
            if (fma_stage_) {
                linear_stage_ = LinearState{
                    alu.linear(fma_stage_->value),
                    std::move(fma_stage_->metadata)};
                fma_stage_.reset();
            }
            if (sram_stage_) {
                fma_stage_ = std::move(sram_stage_);
                sram_stage_.reset();
            }
            if (address_stage_) {
                sram_stage_ = ReadState{
                    alu.read_lut(address_stage_->value),
                    std::move(address_stage_->metadata)};
                address_stage_.reset();
            }
            if (input) {
                address_stage_ = LookupState{
                    alu.make_lookup(input->opcode, input->input),
                    std::move(input->metadata)};
            }
            return output;
        }

        void reset()
        {
            address_stage_.reset();
            sram_stage_.reset();
            fma_stage_.reset();
            linear_stage_.reset();
            restore_active_ = false;
        }

        bool empty() const
        {
            return !address_stage_ && !sram_stage_ && !fma_stage_
                && !linear_stage_;
        }

        std::array<bool, kPipelineLatency> occupancy() const
        {
            return {address_stage_.has_value(), sram_stage_.has_value(),
                    fma_stage_.has_value(), linear_stage_.has_value(),
                    restore_active_};
        }

    private:
        template<typename Value>
        struct State {
            Value value{};
            Metadata metadata{};
        };

        using LookupState = State<VxmLutLookupRequest>;
        using ReadState = State<VxmLutReadResult>;
        using LinearState = State<VxmLutLinearResult>;

        std::optional<LookupState> address_stage_{};
        std::optional<ReadState> sram_stage_{};
        std::optional<ReadState> fma_stage_{};
        std::optional<LinearState> linear_stage_{};
        bool restore_active_{false};
    };

private:
    struct Bank {
        VxmLutConfig config{};
        std::vector<VxmLutEntry> entries{};
        bool configured{false};
    };

    static constexpr std::size_t bank_for_opcode(VxmSpecialAluOpcode opcode)
    {
        switch (opcode) {
        case VxmSpecialAluOpcode::Exp: return 0;
        case VxmSpecialAluOpcode::Reciprocal: return 1;
        case VxmSpecialAluOpcode::Rsqrt: return 2;
        }
        return 0;
    }

    static VxmLutLookupRequest direct(VxmLutLookupRequest request, float result)
    {
        request.lookup_valid = false;
        request.direct_result = result;
        return request;
    }

    VxmLutLookupRequest address(VxmLutLookupRequest request, float local_input) const
    {
        const auto& bank = banks_[request.bank];
        if (!bank.configured) {
            throw std::logic_error("VXM special ALU LUT bank is not configured");
        }
        const auto position = (local_input - bank.config.input_min)
                            / bank.config.segment_width;
        const auto raw_index = static_cast<long>(std::floor(position));
        request.index = static_cast<std::size_t>(std::clamp<long>(
            raw_index, 0, static_cast<long>(bank.entries.size() - 1)));
        const auto x0 = bank.config.input_min
                      + static_cast<float>(request.index) * bank.config.segment_width;
        request.dx = local_input - x0;
        return request;
    }

    std::array<Bank, kBankCount> banks_{};
};

} // namespace ftlpu
