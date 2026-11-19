#pragma once
#include "ftlpu/core/hardware_params.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ftlpu::rtl {
// Native replacement blocks, matching lpu_native_config_pkg and
// lpu_mxm_config_pkg. Legacy whole-chip codecs have a separate ABI.
struct Config {
  std::size_t rows = hw::kTileRows, columns = hw::kTileRows,
              terms = hw::kLanesPerTile, outputs = hw::kLanesPerTile;
  std::size_t mem_depth = hw::kSramDepthRows,
              accumulator_depth = hw::kMxmAccumulatorRows, queue_depth = 16;
  static constexpr std::size_t lanes = 8, streams = 32;
  // Byte stream bindings at each MXM row ingress. The last five bindings
  // are shared by the different parameter layouts.
  std::size_t weight_q_stream = 0;
  std::array<std::size_t, 5> weight_parameter_streams{1, 2, 3, 4, 5};
  bool weight_zero_point_enable = true;
  static constexpr std::size_t compute_hop = 3, result_hop = 1;
  static constexpr std::size_t dot_stages = 9, partial_add_stages = 3;
  static constexpr std::size_t dequant_stages = 5, add_stages = 3;
  static constexpr std::size_t read_to_write = 5, backend_latency = 6;
  static Config profile(unsigned width) {
    Config c;
    c.accumulator_depth = 1024;
    if (width == 16) {
      c.rows = c.columns = 2;
      c.mem_depth = 1024;
    } else if (width == 32) {
      c.rows = c.columns = 4;
      c.mem_depth = 2048;
    } else if (width == 64) {
      c.rows = c.columns = 8;
      c.mem_depth = 2048;
    } else
      throw std::invalid_argument("RTL profile must be 16/32/64");
    return c;
  }
  void validate() const {
    if (!rows || rows > 64 || columns != rows || !terms || !outputs ||
        columns * outputs > 65535 || terms > 32 || mem_depth < 16 ||
        mem_depth > 32768 || (mem_depth & (mem_depth - 1)) ||
        accumulator_depth < 2 || accumulator_depth > 65536 ||
        accumulator_depth % 2 || !queue_depth)
      throw std::invalid_argument("unsupported native RTL geometry");
    if (weight_q_stream >= streams)
      throw std::invalid_argument("invalid MXM weight stream binding");
    std::array<bool, streams> used{};
    used[weight_q_stream] = true;
    for (auto stream : weight_parameter_streams) {
      if (stream >= streams || used[stream])
        throw std::invalid_argument("invalid MXM weight stream binding");
      used[stream] = true;
    }
  }
  unsigned address_bits() const {
    unsigned b = 0;
    for (auto n = accumulator_depth - 1; n; n >>= 1)
      ++b;
    return b ? b : 1;
  }
};
struct MemInstruction {
  bool write = false, preserve = false, delay3 = false, west = false;
  std::uint16_t row = 0;
  std::uint8_t stream = 0;
  std::uint32_t encode(std::size_t depth) const {
    if (row >= depth || stream >= 32 || (preserve && !write))
      throw std::invalid_argument("illegal native MEM instruction");
    return (std::uint32_t(preserve) << 31) | (std::uint32_t(delay3) << 30) |
           (std::uint32_t(row) << 15) | (std::uint32_t(west) << 8) |
           (std::uint32_t(stream) << 3) | write;
  }
  static MemInstruction decode(std::uint32_t word, std::size_t depth,
                               bool raw = true) {
    if ((word & 7) > 1 || (!raw && (word & 0x7e00)) ||
        ((word >> 15) & 0x7fff) >= depth || ((word >> 31) && !(word & 1)))
      throw std::invalid_argument("illegal native MEM word");
    return {bool(word & 1),
            bool(word >> 31),
            bool((word >> 30) & 1),
            bool((word >> 8) & 1),
            std::uint16_t((word >> 15) & 0x7fff),
            std::uint8_t((word >> 3) & 31)};
  }
};
struct ComputeInstruction {
  std::uint8_t activation_stream = 0;
  bool weight_buffer = false;
  std::uint8_t encode() const {
    if (activation_stream >= 64)
      throw std::invalid_argument("stream");
    return (activation_stream << 1) | weight_buffer;
  }
  static ComputeInstruction decode(std::uint8_t v) {
    if (v & 0x80)
      throw std::invalid_argument("compute word");
    return {std::uint8_t(v >> 1), bool(v & 1)};
  }
};
struct LoadInstruction {
  bool weight_buffer = false, column = false;
  std::uint8_t encode() const { return (weight_buffer << 1) | column; }
  static LoadInstruction decode(std::uint8_t v) {
    if (v > 3)
      throw std::invalid_argument("load word");
    return {bool(v & 2), bool(v & 1)};
  }
};
enum class WeightFormat : std::uint8_t { Int8, Fp4E2M1, Fp8E4M3Fn, Reserved };
enum class WeightScaleMode : std::uint8_t { Bf16, Fp32, Ue8m0, NvFp4 };
enum class WeightZeroMode : std::uint8_t { Implicit, Int8, Reserved2, Reserved3 };

struct ParamLoadInstruction {
  WeightScaleMode scale_mode = WeightScaleMode::Bf16;
  WeightZeroMode zero_mode = WeightZeroMode::Implicit;
  bool parameter_buffer = false;
  static constexpr unsigned bits = 5;
  std::uint8_t encode() const {
    if (unsigned(scale_mode) > 3 || unsigned(zero_mode) > 3)
      throw std::invalid_argument("parameter load word");
    return (unsigned(zero_mode) << 3) | (unsigned(scale_mode) << 1) |
           parameter_buffer;
  }
  static ParamLoadInstruction decode(std::uint8_t word) {
    if (word >= (1u << bits))
      throw std::invalid_argument("parameter load word");
    return {WeightScaleMode((word >> 1) & 3), WeightZeroMode(word >> 3),
            bool(word & 1)};
  }
};

struct DequantInstruction {
  WeightFormat format = WeightFormat::Int8;
  bool parameter_buffer = false;
  static constexpr unsigned bits = 3;
  std::uint8_t encode() const {
    if (unsigned(format) > 3)
      throw std::invalid_argument("dequant word");
    return (unsigned(format) << 1) | parameter_buffer;
  }
  static DequantInstruction decode(std::uint8_t word) {
    if (word >= (1u << bits))
      throw std::invalid_argument("dequant word");
    return {WeightFormat(word >> 1), bool(word & 1)};
  }
};

enum class BackendOperation : std::uint8_t {
  DirectOut,
  Store,
  AccStore,
  AccOut
};
struct BackendInstruction {
  BackendOperation operation = BackendOperation::DirectOut;
  std::uint16_t address = 0;
  bool bf16 = false;
  std::uint32_t encode(const Config &c) const {
    if (address >= c.accumulator_depth || unsigned(operation) > 3 ||
        (bf16 && (operation == BackendOperation::Store ||
                  operation == BackendOperation::AccStore)))
      throw std::invalid_argument("backend word");
    return unsigned(operation) | (std::uint32_t(address) << 2) |
           (std::uint32_t(bf16) << (c.address_bits() + 2));
  }
  static BackendInstruction decode(std::uint32_t word, const Config &c) {
    const unsigned b = c.address_bits();
    if (word >> (b + 3))
      throw std::invalid_argument("backend reserved bits");
    BackendInstruction r{BackendOperation(word & 3),
                         std::uint16_t((word >> 2) & ((1u << b) - 1)),
                         bool((word >> (b + 2)) & 1)};
    r.encode(c);
    return r;
  }
};
} // namespace ftlpu::rtl
