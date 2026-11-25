#include "ftlpu/mxm/bf16_arithmetic.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Vector = std::array<std::uint16_t, 8>;

void require(std::uint32_t actual, std::uint32_t expected, const char* message)
{
    if (actual != expected) throw std::runtime_error(message);
}

std::vector<std::string> fields(const std::string& line)
{
    auto result = std::vector<std::string> {};
    auto stream = std::istringstream(line);
    auto field = std::string {};
    while (std::getline(stream, field, ',')) result.push_back(field);
    return result;
}

std::uint32_t hex_value(const std::string& value)
{
    return static_cast<std::uint32_t>(std::stoul(value, nullptr, 16));
}

void compare_dot_oracle(const char* path)
{
    auto input = std::ifstream(path);
    if (!input) throw std::runtime_error("cannot open BF16 dot oracle vectors");
    auto line = std::string {};
    std::size_t checked = 0;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') continue;
        const auto values = fields(line);
        if (values.size() != 18) {
            throw std::runtime_error("invalid BF16 dot oracle row");
        }
        auto activation = Vector {};
        auto weight = Vector {};
        for (std::size_t lane = 0; lane < activation.size(); ++lane) {
            activation[lane] = static_cast<std::uint16_t>(hex_value(values[lane]));
            weight[lane] = static_cast<std::uint16_t>(
                hex_value(values[activation.size() + lane]));
        }
        const auto actual = ftlpu::mxm_numeric::bf16_dot(
            activation, weight);
        if (actual.bits != hex_value(values[16])
            || actual.alignment_lost != (hex_value(values[17]) != 0)) {
            throw std::runtime_error("BF16 dot oracle mismatch");
        }
        ++checked;
    }
    if (checked == 0) throw std::runtime_error("empty BF16 dot oracle");
}

void compare_dequant_oracle(const char* path)
{
    auto input = std::ifstream(path);
    if (!input) throw std::runtime_error("cannot open dequant oracle vectors");
    auto line = std::string {};
    std::size_t checked = 0;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') continue;
        const auto values = fields(line);
        if (values.size() != 4) {
            throw std::runtime_error("invalid dequant oracle row");
        }
        const auto actual = ftlpu::mxm_numeric::int8_dequant_bf16(
            static_cast<std::int8_t>(
                static_cast<std::uint8_t>(hex_value(values[0]))),
            static_cast<std::int8_t>(
                static_cast<std::uint8_t>(hex_value(values[1]))),
            static_cast<std::uint16_t>(hex_value(values[2])));
        if (actual != static_cast<std::uint16_t>(hex_value(values[3]))) {
            throw std::runtime_error("INT8 dequant oracle mismatch");
        }
        ++checked;
    }
    if (checked == 0) throw std::runtime_error("empty dequant oracle");
}

} // namespace

int main(int argc, char** argv)
{
    using ftlpu::mxm_numeric::bf16_dot;
    using ftlpu::mxm_numeric::fp32_add_bits;
    using ftlpu::mxm_numeric::fp32_to_bf16;

    auto activation = Vector {};
    auto weight = Vector {};
    activation.fill(0x3f80); // +1
    weight.fill(0x3f80);
    require(bf16_dot(activation, weight).bits, 0x41000000,
        "eight-term BF16 dot mismatch");

    activation.fill(0);
    weight.fill(0x3f80);
    activation[0] = 0x007f; // positive subnormal: DAZ to +0
    activation[1] = 0x807f; // negative subnormal: DAZ to -0
    require(bf16_dot(activation, weight).bits, 0,
        "BF16 DAZ subnormal mismatch");

    activation.fill(0);
    weight.fill(0x3f80);
    activation[0] = 0x0080; // minimum normal
    weight[0] = 0x3f00; // 0.5
    require(bf16_dot(activation, weight).bits, 0x00400000,
        "BF16 dot gradual-underflow mismatch");

    activation.fill(0);
    weight.fill(0x3f80);
    activation[0] = 0x3f80;
    activation[1] = 0xbf80;
    require(bf16_dot(activation, weight).bits, 0,
        "BF16 dot cancellation mismatch");

    activation.fill(0);
    weight.fill(0x3f80);
    activation[0] = 0x7fc1;
    require(bf16_dot(activation, weight).bits, 0x7fc00000,
        "BF16 dot NaN canonicalization mismatch");

    activation.fill(0);
    weight.fill(0x3f80);
    activation[0] = 0x7f80;
    require(bf16_dot(activation, weight).bits, 0x7f800000,
        "BF16 dot infinity mismatch");

    activation.fill(0);
    weight.fill(0x3f80);
    activation[0] = 0;
    weight[0] = 0x7f80;
    require(bf16_dot(activation, weight).bits, 0x7fc00000,
        "BF16 dot zero-times-infinity mismatch");

    require(fp32_add_bits(0x3f800000, 0x33800000), 0x3f800000,
        "FP32 RNE tie-even mismatch");
    require(fp32_add_bits(0x3f800001, 0x33800000), 0x3f800002,
        "FP32 RNE tie-odd mismatch");
    require(fp32_add_bits(0x00800000, 0x807fffff), 0x00000001,
        "FP32 gradual-underflow add mismatch");
    require(fp32_add_bits(0x7f800000, 0xff800000), 0x7fc00000,
        "FP32 infinity cancellation mismatch");

    require(fp32_to_bf16(0x3f808000u), 0x3f80,
        "FP32 to BF16 even tie mismatch");
    require(fp32_to_bf16(0x3f818000u), 0x3f82,
        "FP32 to BF16 odd tie mismatch");
    require(fp32_to_bf16(0x7f800001u), 0x7fc0,
        "FP32 to BF16 NaN quieting mismatch");
    require(fp32_to_bf16(0x00008001u), 0x0001,
        "FP32 to BF16 subnormal RNE mismatch");
    if (argc > 1) compare_dot_oracle(argv[1]);
    if (argc > 2) compare_dequant_oracle(argv[2]);
    return 0;
}
