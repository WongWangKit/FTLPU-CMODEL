#pragma once

#include "ftlpu/core/bf16.hpp"
#include "ftlpu/core/fp16.hpp"

#include <cstdint>
#include <stdexcept>

namespace ftlpu {

enum class MxmDataFormat : std::uint8_t {
    Float16 = 0,
    BFloat16 = 1,
};

inline const char* mxm_data_format_name(MxmDataFormat format)
{
    switch (format) {
    case MxmDataFormat::Float16:
        return "fp16";
    case MxmDataFormat::BFloat16:
        return "bf16";
    }
    throw std::invalid_argument("MXM data format is invalid");
}

inline float decode_mxm_16bit(std::uint16_t bits, MxmDataFormat format)
{
    switch (format) {
    case MxmDataFormat::Float16:
        return Fp16::from_bits(bits).to_float();
    case MxmDataFormat::BFloat16:
        return Bf16::from_bits(bits).to_float();
    }
    throw std::invalid_argument("MXM data format is invalid");
}

inline float quantize_mxm_16bit(float value, MxmDataFormat format)
{
    switch (format) {
    case MxmDataFormat::Float16:
        return Fp16::from_float(value).to_float();
    case MxmDataFormat::BFloat16:
        return Bf16::from_float(value).to_float();
    }
    throw std::invalid_argument("MXM data format is invalid");
}

inline std::uint16_t encode_mxm_16bit(float value, MxmDataFormat format)
{
    switch (format) {
    case MxmDataFormat::Float16:
        return Fp16::from_float(value).bits();
    case MxmDataFormat::BFloat16:
        return Bf16::from_float(value).bits();
    }
    throw std::invalid_argument("MXM data format is invalid");
}

} // namespace ftlpu
