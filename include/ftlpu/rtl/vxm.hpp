#pragma once
#include "ftlpu/rtl/float_bits.hpp"
#include <algorithm>
#include <array>
#include <optional>
#include <vector>

namespace ftlpu::rtl {
enum class Format : std::uint8_t { Fp16, Bf16, Fp32 };
enum class VxmOperation : std::uint8_t {
  Bypass,
  Add,
  Subtract,
  Multiply,
  Negate,
  Max,
  Special0,
  Special1,
  Log
};
struct VxmGlobalConfig {
  bool right_to_left = false;
  unsigned chain = 0, active_width = 0;
  Format compute = Format::Fp16;
  unsigned lhs_bits = 0;
  Format lhs = Format::Fp16;
  unsigned rhs_bits = 0;
  Format rhs = Format::Fp16;
  std::uint16_t encode() const {
    if (chain > 2 || active_width > 2 || unsigned(compute) > 2 ||
        lhs_bits > 2 || rhs_bits > 2 || unsigned(lhs) > 2 || unsigned(rhs) > 2)
      throw std::invalid_argument("VXM global fields");
    return (right_to_left << 14) | (chain << 12) | (active_width << 10) |
           (unsigned(compute) << 8) | (lhs_bits << 6) | (unsigned(lhs) << 4) |
           (rhs_bits << 2) | unsigned(rhs);
  }
  static VxmGlobalConfig decode(std::uint16_t v) {
    if (v & 0x8000)
      throw std::invalid_argument("global reserved bit");
    VxmGlobalConfig c{bool(v & 0x4000),        unsigned((v >> 12) & 3),
                      unsigned((v >> 10) & 3), Format((v >> 8) & 3),
                      unsigned((v >> 6) & 3),  Format((v >> 4) & 3),
                      unsigned((v >> 2) & 3),  Format(v & 3)};
    c.encode();
    return c;
  }
};
enum class OperandSource {
  Stream,
  Immediate,
  Feedback,
  Previous,
  Original,
  Auxiliary,
  Accumulator
};
struct VxmLocal {
  VxmOperation operation;
  OperandSource lhs, rhs;
  bool head, tail;
  static VxmLocal decode(unsigned stage, unsigned chain, std::uint8_t word) {
    if (stage >= 16 || chain > 2)
      throw std::invalid_argument("VXM position/chain");
    const unsigned q = stage % 8, length = 2u << chain;
    VxmLocal d{VxmOperation(word & 7), OperandSource::Previous,
               OperandSource::Original, stage % length == 0,
               stage % length == length - 1};
    unsigned lc = 0, rc = 0;
    if (q == 0) {
      if (word & 0x40)
        throw std::invalid_argument("Q0 width");
      lc = (word >> 3) & 3;
      rc = (word >> 5) & 1;
    } else if (q == 1 || q == 5) {
      if (word & 0x60)
        throw std::invalid_argument("Q1/5 width");
      rc = (word >> 3) & 3;
    } else if (q == 3 || q == 7) {
      if (word & 0x40)
        throw std::invalid_argument("Q3/7 width");
      rc = (word >> 3) & 3;
      if (word & 0x20) {
        if (word != 0x26)
          throw std::invalid_argument("noncanonical Log");
        d.operation = VxmOperation::Log;
      }
    } else {
      if (word & 0x80)
        throw std::invalid_argument("local width");
      lc = (word >> 3) & 3;
      rc = (word >> 5) & 3;
    }
    if (d.head) {
      if (lc > 2 || rc > 1)
        throw std::invalid_argument("head operand");
      d.lhs = lc == 0   ? OperandSource::Stream
              : lc == 1 ? OperandSource::Immediate
                        : OperandSource::Feedback;
      d.rhs = rc == 0 ? OperandSource::Stream : OperandSource::Immediate;
    } else {
      if ((q == 2 || q == 4 || q == 6) && lc)
        throw std::invalid_argument("internal lhs");
      d.rhs = rc == 0   ? OperandSource::Original
              : rc == 1 ? OperandSource::Auxiliary
              : rc == 2 ? OperandSource::Immediate
                        : OperandSource::Accumulator;
      if (rc == 3 && !(q & 1))
        throw std::invalid_argument("even accumulator source");
    }
    const auto op = unsigned(d.operation);
    if ((q % 4 == 0 || q % 4 == 2) && op > 5)
      throw std::invalid_argument("basic-only ALU");
    if (q % 4 == 1 && op > 6)
      throw std::invalid_argument("Exp ALU");
    return d;
  }
};
namespace vxmath {
struct Conversion {
  bool valid = false, active = false, fault = false;
  std::uint32_t bits = 0;
};
inline Conversion convert(bool valid, Format from, Format to,
                          std::uint32_t raw) {
  Conversion c;
  if (!valid)
    return c;
  if (unsigned(from) > 2 || unsigned(to) > 2) {
    c.fault = true;
    return c;
  }
  if (from == to) {
    c.valid = true;
    c.bits = to == Format::Fp32 ? raw : raw & 65535;
    return c;
  }
  if (from == Format::Fp16) {
    c.bits = fp::from_fp16(std::uint16_t(raw), true);
    if (fp::nan(c.bits))
      c.bits = (c.bits & 0x80000000u) | fp::qnan;
    if (to == Format::Bf16)
      c.bits = fp::nan(c.bits) ? (c.bits >> 16) : fp::bf16(c.bits, true);
  } else if (from == Format::Bf16 && to == Format::Fp32) {
    const auto wide = std::uint32_t(std::uint16_t(raw)) << 16;
    c.bits = fp::nan(wide) ? (wide & 0x80000000u) | fp::qnan : fp::daz(wide);
  } else if (from == Format::Fp32 && to == Format::Bf16)
    c.bits = fp::nan(raw) ? ((raw >> 16) & 0x8000) | 0x7fc0 : fp::bf16(raw, true);
  else {
    c.fault = true;
    return c;
  }
  c.valid = c.active = true;
  return c;
}
inline std::uint32_t widen(std::uint32_t a, Format f, bool flush = true) {
  if (f == Format::Fp16)
    return fp::from_fp16(std::uint16_t(a), flush);
  if (f == Format::Bf16)
    return flush ? fp::daz(a << 16) : a << 16;
  return flush ? fp::daz(a) : a;
}
inline std::uint32_t narrow(std::uint32_t a, Format f, bool flush = true) {
  if (fp::nan(a))
    return f == Format::Fp16 ? 0x7e00 : f == Format::Bf16 ? 0x7fc0 : fp::qnan;
  return f == Format::Fp16   ? fp::fp16(a, flush)
         : f == Format::Bf16 ? fp::bf16(a, flush)
         : flush             ? fp::ftz(a)
                             : a;
}
inline std::uint32_t round(std::uint32_t a, Format f) {
  return widen(narrow(a, f), f);
}
inline std::uint32_t basic(VxmOperation op, Format f, std::uint32_t raw_a,
                           std::uint32_t raw_b) {
  const auto mask = f == Format::Fp32 ? 0xffffffffu : 0xffffu,
             sign = f == Format::Fp32 ? 0x80000000u : 0x8000u;
  if (op == VxmOperation::Bypass)
    return raw_a & mask;
  if (op == VxmOperation::Negate)
    return (raw_a ^ sign) & mask;
  const auto a = widen(raw_a, f), b = widen(raw_b, f);
  std::uint32_t result = 0;
  if (op == VxmOperation::Add)
    result = fp::add(a, b);
  else if (op == VxmOperation::Subtract)
    result = fp::add(a, b ^ 0x80000000);
  else if (op == VxmOperation::Multiply)
    result = fp::multiply(a, b);
  else if (op == VxmOperation::Max) {
    if (fp::nan(a) || fp::nan(b))
      result = fp::qnan;
    else if (!(a & 0x7fffffff) && !(b & 0x7fffffff))
      result = 0;
    else {
      const auto ka = (a >> 31) ? ~a : a ^ 0x80000000,
                 kb = (b >> 31) ? ~b : b ^ 0x80000000;
      result = ka >= kb ? a : b;
    }
  } else
    throw std::invalid_argument("not a basic operation");
  return narrow(result, f);
}
inline std::uint64_t rshift(std::uint64_t v, unsigned n) {
  if (!n)
    return v;
  if (n >= 64)
    return 0;
  const auto r = v >> n, rem = v & ((std::uint64_t(1) << n) - 1),
             half = std::uint64_t(1) << (n - 1);
  return r + (rem > half || (rem == half && (r & 1)));
}
inline std::uint32_t scale(std::uint32_t v, int e) {
  v = fp::daz(v);
  if (fp::nan(v))
    return fp::qnan;
  if (fp::inf(v) || !(v & 0x7fffffff))
    return v;
  const int ex = int((v >> 23) & 255) + e;
  return ex <= 0     ? v & 0x80000000
         : ex >= 255 ? (v & 0x80000000) | 0x7f800000
                     : (v & 0x807fffff) | (unsigned(ex) << 23);
}
inline std::int64_t to_fixed(std::uint32_t v, unsigned fraction) {
  const auto x = fp::unpack(v);
  if (!x.sig)
    return 0;
  const int shift = x.e - 23 + int(fraction);
  const auto m = shift >= 0 ? std::uint64_t(x.sig) << shift
                            : rshift(x.sig, unsigned(-shift));
  return v >> 31 ? -std::int64_t(m) : std::int64_t(m);
}
inline int integer_rne(std::uint32_t a) {
  const auto x = fp::unpack(a);
  if (!x.sig || x.e < -1)
    return 0;
  if (x.e >= 31)
    return a >> 31 ? -0x7fffffff : 0x7fffffff;
  const auto m = x.e >= 23 ? std::uint64_t(x.sig) << (x.e - 23)
                           : rshift(x.sig, unsigned(23 - x.e));
  return a >> 31 ? -int(m) : int(m);
}
inline std::uint32_t coefficient_multiply(std::uint32_t stored,
                                          std::uint32_t value, Format f) {
  const unsigned cf = f == Format::Bf16 ? 9 : f == Format::Fp16 ? 12 : 25;
  const auto c = rshift(stored, 25 - cf);
  const auto x = fp::unpack(round(value, f));
  if (!x.sig)
    return value & 0x80000000;
  const unsigned vf = f == Format::Fp16 ? 10 : f == Format::Bf16 ? 7 : 23;
  const auto sig = x.sig >> (23 - vf);
  const auto p = c * sig;
  if (!p)
    return value & 0x80000000;
  const auto l = fp::lead(p);
  auto result = fp::ftz(
      fp::pack(value >> 31, x.e - int(vf) - int(cf) + int(l), p << (63 - l)));
  return f == Format::Fp16 ? round(result, f) : result;
}
} // namespace vxmath
struct LutBank {
  bool configured = false;
  std::uint16_t minimum = 0, width = 0x3c00;
  std::array<std::uint64_t, 64> row{};
};
// Physically independent per-lane storage. Tile configuration may broadcast
// writes; no shared_ptr or shared request port across different lanes.
using LaneLuts = std::array<LutBank, 4>;
class VxmArithmetic {
public:
  static bool requires_lut(unsigned function, Format f, std::uint32_t raw) {
    const auto a = vxmath::widen(raw, f);
    if (fp::nan(a) || fp::inf(a))
      return false;
    if (function == 0)
      return true;
    if (!(a & 0x7fffffff) || (function >= 2 && (a >> 31)))
      return false;
    if (function != 3)
      return true;
    const bool near = f == Format::Bf16 ? (a >= 0x3f7e0000 && a <= 0x3f810000)
                                        : (a >= 0x3f7c0000 && a <= 0x3f820000);
    return !near;
  }
  static std::uint32_t log(Format f, std::uint32_t raw, const LutBank &bank,
                           bool &fault) {
    using namespace vxmath;
    const auto a = widen(raw, f);
    if (fp::nan(a))
      return narrow(fp::qnan, f);
    if (!(a & 0x7fffffff))
      return narrow(0xff800000, f);
    if (a >> 31)
      return narrow(fp::qnan, f);
    if (fp::inf(a))
      return narrow(a, f);
    const unsigned fb = f == Format::Fp16 ? 10 : f == Format::Bf16 ? 7 : 23;
    const unsigned fraction = raw & ((1u << fb) - 1),
                   index = fraction >> (fb - 6);
    int exponent = int((a >> 23) & 255) - 127;
    const bool near = f == Format::Bf16 ? (a >= 0x3f7e0000 && a <= 0x3f810000)
                                        : (a >= 0x3f7c0000 && a <= 0x3f820000);
    if (!near && !bank.configured) {
      fault = true;
      return narrow(fp::qnan, f);
    }
    std::uint32_t delta =
        fp::signed_fixed(fraction & ((1u << (fb - 6)) - 1), fb);
    if (near)
      delta = fp::add(a, 0xbf800000);
    const auto row = bank.row[index];
    const unsigned bs = f == Format::Bf16   ? 14
                        : f == Format::Fp16 ? 13
                                            : 0,
                   is = f == Format::Bf16   ? 12
                        : f == Format::Fp16 ? 9
                                            : 0;
    const auto base = near ? 0 : rshift(row >> 22, bs) << bs,
               inv = near ? 1u << 21 : rshift(row & 0x3fffff, is) << is;
    if (near)
      exponent = 0;
    const auto u =
        round(fp::multiply(delta, fp::signed_fixed(std::int64_t(inv), 21)), f);
    std::uint32_t h = 0x3f800000;
    if (f == Format::Fp16)
      h = round(fp::add(round(fp::multiply(0xbf000000, u), f), 0x3f800000), f);
    if (f == Format::Fp32) {
      h = fp::add(fp::multiply(0xbe800000, u), 0x3eaaaaab);
      h = fp::add(fp::multiply(h, u), 0xbf000000);
      h = fp::add(fp::multiply(h, u), 0x3f800000);
    }
    const auto p = round(fp::multiply(h, u), f);
    const auto result =
        near ? p
             : fp::signed_fixed(std::int64_t(exponent) * 0x00b17217f7d2ll +
                                    std::int64_t(base << 12) + to_fixed(p, 40),
                                40);
    return narrow(result, f);
  }
  static std::uint32_t reciprocal_or_rsqrt(bool rsqrt, Format f,
                                           std::uint32_t raw,
                                           const LutBank &bank, bool &fault) {
    using namespace vxmath;
    const auto a = widen(raw, f);
    if (fp::nan(a) || (rsqrt && (a >> 31) && (a & 0x7fffffff)))
      return narrow(fp::qnan, f);
    if (!(a & 0x7fffffff))
      return narrow((rsqrt ? 0 : a & 0x80000000) | 0x7f800000, f);
    if (fp::inf(a))
      return narrow(rsqrt ? 0 : a & 0x80000000, f);
    if (!bank.configured) {
      fault = true;
      return narrow(fp::qnan, f);
    }
    const unsigned fb = f == Format::Fp16 ? 10 : f == Format::Bf16 ? 7 : 23;
    const unsigned fraction = raw & ((1u << fb) - 1);
    const int original = int((a >> 23) & 255) - 127;
    const bool odd = bool(original & 1);
    const unsigned address = rsqrt ? ((odd ? 32u : 0u) | (fraction >> (fb - 5)))
                                   : fraction >> (fb - 6);
    const unsigned residual = fraction & ((1u << (fb - (rsqrt ? 5 : 6))) - 1);
    const unsigned cf = f == Format::Bf16 ? 9 : f == Format::Fp16 ? 12 : 15;
    const auto row = bank.row[address], k = rshift(row >> 16, 15 - cf),
               b = rshift(row & 65535, 15 - cf);
    const auto difference =
        (std::int64_t(b) << fb) - std::int64_t(k * residual);
    if (difference < 0) {
      fault = true;
      return 0;
    }
    const auto retained = rshift(std::uint64_t(difference), fb);
    auto y = fp::signed_fixed(std::int64_t(retained), cf);
    if (f == Format::Fp16)
      y = round(y, f);
    const auto m = (a & 0x7fffff) | ((rsqrt && odd ? 128u : 127u) << 23);
    if (f == Format::Fp32) {
      if (!rsqrt)
        y = fp::multiply(y,
                         fp::add(0x40000000, fp::multiply(m, y) ^ 0x80000000));
      else {
        const auto sq = fp::multiply(y, y);
        const auto ms = fp::unpack(m), ys = fp::unpack(sq);
        const auto product = std::uint64_t(ms.sig) * ys.sig;
        const int shift = ms.e + ys.e + 8;
        const auto q55 = shift >= 0 && shift <= 16   ? product << shift
                         : shift < 0 && shift >= -47 ? product >> -shift
                                                     : 0;
        const auto one = std::uint64_t(3) << 54;
        if (!ys.sig || q55 >= one) {
          fault = true;
          return 0;
        }
        y = fp::multiply(y, fp::signed_fixed(std::int64_t(one - q55), 55));
      }
    }
    if (rsqrt)
      return narrow(scale(y, -(original - (odd ? 1 : 0)) / 2), f);
    const int selected = (f == Format::Fp16 ? 30 - int((raw >> 10) & 31)
                                            : 254 - int((raw >> fb) & 255)) -
                         (((y >> 23) & 255) >= 127 ? 0 : 1);
    std::uint32_t result;
    if (f == Format::Fp16) {
      const auto h = fp::fp16(y, true);
      result = selected <= 0    ? 0
               : selected >= 31 ? 0x7c00
                                : (unsigned(selected) << 10) | (h & 1023);
      return result | (raw & 0x8000);
    }
    result = selected <= 0     ? 0
             : selected >= 255 ? 0x7f800000
                               : (unsigned(selected) << 23) | (y & 0x7fffff);
    return narrow(result | (a & 0x80000000), f);
  }
  static std::uint32_t exp(Format f, std::uint32_t raw, const LutBank &bank,
                           bool &fault) {
    using namespace vxmath;
    const auto a = widen(raw, f);
    if (fp::nan(a))
      return narrow(fp::qnan, f);
    if (fp::inf(a))
      return narrow(a >> 31 ? 0 : 0x7f800000, f);
    if (!bank.configured) {
      fault = true;
      return narrow(fp::qnan, f);
    }
    auto product =
        fp::multiply(a, f == Format::Fp16 ? fp::from_fp16(0x3dc5) : 0x3fb8aa3b);
    if (f == Format::Fp16)
      product = round(product, f);
    int exponent =
        std::clamp(integer_rne(product), f == Format::Fp16 ? -31 : -127,
                   f == Format::Fp16 ? 31 : 127);
    auto ex = fp::signed_fixed(exponent, 0);
    auto ex_ln = fp::multiply(ex, f == Format::Fp16 ? fp::from_fp16(0x398c)
                                                    : 0x3f317218);
    if (f == Format::Fp16)
      ex_ln = round(ex_ln, f);
    auto local = fp::add(a, ex_ln ^ 0x80000000);
    if (f == Format::Fp16)
      local = round(local, f);
    const auto minimum = fp::from_fp16(bank.minimum, true),
               width = fp::from_fp16(bank.width, true);
    auto numerator = fp::add(local, minimum ^ 0x80000000);
    if (f == Format::Fp16)
      numerator = round(numerator, f);
    auto pos = fp::divide(numerator, width);
    if (f == Format::Fp16)
      pos = round(pos, f);
    const auto px = fp::unpack(pos);
    unsigned index = 0;
    if (!(pos >> 31) && px.e >= 0)
      index = px.e >= 23
                  ? unsigned(std::min<std::uint64_t>(
                        std::uint64_t(px.sig) << (std::min(px.e - 23, 31)), 63))
                  : px.sig >> (23 - px.e);
    index = std::min(index, 63u);
    auto x0 = fp::add(minimum, fp::multiply(fp::signed_fixed(index, 0), width));
    if (f == Format::Fp16)
      x0 = round(x0, f);
    auto dx = fp::add(local, x0 ^ 0x80000000);
    if (f == Format::Fp16)
      dx = round(dx, f);
    const auto row = bank.row[index];
    const std::uint32_t base = std::uint32_t(row) & 0x3ffffff;
    std::uint32_t y;
    if (f == Format::Fp32) {
      y = fp::add(fp::multiply(0x3e2aaaab, dx), 0x3f000000);
      y = fp::add(fp::multiply(y, dx), 0x3f800000);
      y = fp::add(fp::multiply(y, dx), 0x3f800000);
      y = coefficient_multiply(base, y, f);
    } else {
      const auto cf = f == Format::Fp16 ? 12 : 9;
      auto b = fp::signed_fixed(std::int64_t(rshift(base, 25 - cf)), cf);
      if (f == Format::Fp16)
        b = round(b, f);
      y = fp::add(coefficient_multiply(base, dx, f), b);
      if (f == Format::Fp16)
        y = round(y, f);
    }
    return narrow(scale(y, exponent), f);
  }
};

class VxmAlu {
public:
  struct Request {
    VxmOperation operation = VxmOperation::Bypass;
    Format format = Format::Fp16;
    std::uint32_t lhs = 0, rhs = 0;
    std::uint16_t tag = 0;
  };
  struct Result {
    std::uint32_t bits = 0;
    std::uint16_t tag = 0;
    bool fault = false;
  };
  explicit VxmAlu(unsigned physical_stage = 0) : stage_(physical_stage) {
    if (stage_ >= 16)
      throw std::invalid_argument("VXM stage");
  }
  void reset() {
    pending_.reset();
    remaining_ = 0;
    special_pending_ = false;
  }
  bool ready() const {
    return !pending_ || (!special_pending_ && remaining_ == 0);
  }
  std::optional<Result> tick(std::optional<Request> request,
                             const LaneLuts &luts) {
    if (request && !ready())
      throw std::logic_error("VXM request while ALU not ready");
    std::optional<Result> result;
    if (pending_) {
      if (!remaining_) {
        result = pending_;
        pending_.reset();
      } else
        --remaining_;
    }
    if (request) {
      if (pending_)
        throw std::logic_error("VXM request while ALU not ready");
      const auto &r = *request;
      Result y{0, r.tag, false};
      unsigned latency = 1;
      if (unsigned(r.operation) <= 5)
        y.bits = vxmath::basic(r.operation, r.format, r.lhs, r.rhs);
      else {
        latency = 8;
        if (stage_ % 4 == 1 && r.operation == VxmOperation::Special0)
          y.bits = VxmArithmetic::exp(r.format, r.lhs, luts[0], y.fault);
        else if (stage_ % 4 == 3 && r.operation == VxmOperation::Special0)
          y.bits = VxmArithmetic::reciprocal_or_rsqrt(false, r.format, r.lhs,
                                                      luts[1], y.fault);
        else if (stage_ % 4 == 3 && r.operation == VxmOperation::Special1)
          y.bits = VxmArithmetic::reciprocal_or_rsqrt(true, r.format, r.lhs,
                                                      luts[2], y.fault);
        else if (stage_ % 4 == 3 && r.operation == VxmOperation::Log)
          y.bits = VxmArithmetic::log(r.format, r.lhs, luts[3], y.fault);
        else
          throw std::invalid_argument(
              "unsupported special opcode at physical stage");
      }
      if (r.operation == VxmOperation::Multiply)
        latency = 2;
      if (latency == 1) {
        if (result)
          throw std::logic_error("VXM completing output collision");
        result = y;
      } else {
        pending_ = y;
        remaining_ = latency - 2;
        special_pending_ = unsigned(r.operation) > 5;
      }
    }
    return result;
  }

private:
  unsigned stage_;
  std::optional<Result> pending_;
  unsigned remaining_ = 0;
  bool special_pending_ = false;
};

// A chain head's fixed two byte streams, independent per LHS/RHS. A completed
// input may execute on the FOLLOWING edge. FP32 is low-half/high-half/execute.
class VxmInput {
public:
  void reset() {
    high_ = false;
    ready_ = false;
    value_ = 0;
  }
  bool ready() const { return ready_; }
  std::uint32_t value() const { return value_; }
  void consume() {
    if (!ready_)
      throw std::logic_error("incomplete VXM input");
    reset();
  }
  bool tick(std::optional<std::uint16_t> pair, unsigned read_bits) {
    if (read_bits > 2)
      throw std::invalid_argument("reserved input width");
    if (ready_ || !pair)
      return false;
    if (read_bits == 1) {
      if (!high_) {
        value_ = *pair;
        high_ = true;
      } else {
        value_ |= std::uint32_t(*pair) << 16;
        ready_ = true;
      }
    } else {
      value_ = read_bits == 2 ? *pair & 255 : *pair;
      ready_ = true;
    }
    return true;
  }

private:
  bool high_ = false, ready_ = false;
  std::uint32_t value_ = 0;
};
} // namespace ftlpu::rtl
