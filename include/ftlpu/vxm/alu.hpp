#pragma once

#include "ftlpu/vxm/data_format.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>

namespace ftlpu {

// Only scalar operations implemented by every Basic ALU live here.
enum class VxmAluOpcode {
    Bypass,
    Add,
    Subtract,
    Multiply,
    Negate,
    Max,
};

enum class VxmAluPrecision {
    Float16,
    Float32,
};

struct VxmAluInstruction {
    VxmAluOpcode opcode{VxmAluOpcode::Bypass};
    VxmAluPrecision precision{VxmAluPrecision::Float16};
};

// Scalar arithmetic and the internal timing of one Basic ALU.
// Functional evaluation remains available separately for reference tests.
class VxmAlu {
public:
    static constexpr std::size_t kDefaultLatency = 1;
    static constexpr std::size_t kMultiplyLatency = 2;
    static constexpr std::size_t kInitiationInterval = 1;

    static constexpr std::size_t latency(const VxmAluInstruction& instruction)
    {
        return instruction.opcode == VxmAluOpcode::Multiply
            ? kMultiplyLatency : kDefaultLatency;
    }

    static float execute(const VxmAluInstruction& instruction,
                         float lhs, float rhs = 0.0f)
    {
        if (instruction.precision == VxmAluPrecision::Float16) {
            lhs = VxmDataFormat::round_fp16_ftz(lhs);
            rhs = VxmDataFormat::round_fp16_ftz(rhs);
        }

        float result = 0.0f;
        switch (instruction.opcode) {
        case VxmAluOpcode::Bypass: result = lhs; break;
        case VxmAluOpcode::Add: result = lhs + rhs; break;
        case VxmAluOpcode::Subtract: result = lhs - rhs; break;
        case VxmAluOpcode::Multiply: result = lhs * rhs; break;
        case VxmAluOpcode::Negate: result = -lhs; break;
        case VxmAluOpcode::Max: result = std::max(lhs, rhs); break;
        }

        return instruction.precision == VxmAluPrecision::Float16
            ? VxmDataFormat::round_fp16_ftz(result)
            : result;
    }

    // Metadata follows the numerical result through the ALU. Lane-level
    // routing state remains aligned without being interpreted here.
    template<typename Metadata>
    class Pipeline {
    public:
        struct Request {
            VxmAluInstruction instruction{};
            float lhs{0.0f};
            float rhs{0.0f};
            Metadata metadata{};
        };

        struct Result {
            float value{0.0f};
            Metadata metadata{};
        };

        std::optional<Result> tick(std::optional<Request> request = std::nullopt)
        {
            auto output = std::move(multiply_register_);
            multiply_register_.reset();

            if (!request) return output;

            auto result = Result{
                VxmAlu::execute(request->instruction,
                                request->lhs, request->rhs),
                std::move(request->metadata)};

            if (VxmAlu::latency(request->instruction) == kDefaultLatency) {
                if (output) {
                    throw std::logic_error(
                        "VXM Basic ALU output collision between a completing "
                        "Multiply and a one-cycle operation");
                }
                output = std::move(result);
            } else {
                // The register supplies the extra cycle of a two-cycle,
                // fully-pipelined multiplier. A new Multiply may enter every
                // cycle while the preceding one completes.
                multiply_register_ = std::move(result);
            }
            return output;
        }

        void reset() { multiply_register_.reset(); }
        bool empty() const { return !multiply_register_.has_value(); }

    private:
        std::optional<Result> multiply_register_{};
    };
};

} // namespace ftlpu
