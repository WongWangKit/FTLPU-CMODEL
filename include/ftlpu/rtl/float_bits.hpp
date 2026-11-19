#pragma once
#include "ftlpu/rtl/config.hpp"
#include <bit>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace ftlpu::rtl::fp {
constexpr std::uint32_t qnan = 0x7fc00000;
inline unsigned lead(std::uint64_t v) {
  return v ? 63u - std::countl_zero(v) : 0;
}
inline std::uint64_t jam(std::uint64_t v, unsigned n) {
  if (!n)
    return v;
  if (n >= 64)
    return v != 0;
  return (v >> n) | bool(v & ((std::uint64_t(1) << n) - 1));
}
inline std::uint32_t pack(bool sign, int e, std::uint64_t m) {
  if (e < -126) {
    m = jam(m, unsigned(-126 - e));
    e = -126;
  }
  std::uint32_t r = std::uint32_t(m >> 40);
  if ((m & (std::uint64_t(1) << 39)) &&
      ((m & ((std::uint64_t(1) << 39) - 1)) || (r & 1)))
    ++r;
  if (r & 0x1000000) {
    r >>= 1;
    ++e;
  }
  if (!r)
    return std::uint32_t(sign) << 31;
  if (e > 127)
    return (std::uint32_t(sign) << 31) | 0x7f800000;
  const unsigned exponent =
      (e == -126 && !(r & 0x800000)) ? 0 : unsigned(e + 127);
  return (std::uint32_t(sign) << 31) | (exponent << 23) | (r & 0x7fffff);
}
inline bool nan(std::uint32_t v) { return (v & 0x7fffffff) > 0x7f800000; }
inline bool inf(std::uint32_t v) { return (v & 0x7fffffff) == 0x7f800000; }
inline std::uint32_t daz(std::uint32_t v) {
  return (v & 0x7f800000) ? v : v & 0x80000000;
}
inline std::uint32_t ftz(std::uint32_t v) { return daz(v); }
struct Finite {
  std::uint32_t sig;
  int e;
};
inline Finite unpack(std::uint32_t v) {
  const unsigned ex = (v >> 23) & 255;
  std::uint32_t s = (v & 0x7fffff) | (ex ? 0x800000 : 0);
  int e = ex ? int(ex) - 127 : -126;
  if (s && !(s & 0x800000)) {
    const unsigned d = 23 - lead(s);
    s <<= d;
    e -= int(d);
  }
  return {s, e};
}
inline std::uint32_t add(std::uint32_t a, std::uint32_t b) {
  const bool sa = a >> 31, sb = b >> 31;
  if (nan(a) || nan(b) || (inf(a) && inf(b) && sa != sb))
    return qnan;
  if (inf(a))
    return a;
  if (inf(b))
    return b;
  const auto x = unpack(a), y = unpack(b);
  if (!x.sig && !y.sig)
    return std::uint32_t(sa && sb) << 31;
  if (!x.sig)
    return b;
  if (!y.sig)
    return a;
  const int e = x.e > y.e ? x.e : y.e;
  const auto xm = jam(std::uint64_t(x.sig) << 37, unsigned(e - x.e));
  const auto ym = jam(std::uint64_t(y.sig) << 37, unsigned(e - y.e));
  std::uint64_t m;
  bool sign;
  if (sa == sb) {
    m = xm + ym;
    sign = sa;
  } else if (xm >= ym) {
    m = xm - ym;
    sign = sa;
  } else {
    m = ym - xm;
    sign = sb;
  }
  if (!m)
    return 0;
  const unsigned l = lead(m);
  return pack(sign, e + int(l) - 60, m << (63 - l));
}
inline std::uint32_t multiply(std::uint32_t a, std::uint32_t b) {
  const bool sign = (a ^ b) >> 31;
  if (nan(a) || nan(b) || (inf(a) && !(b & 0x7fffffff)) ||
      (inf(b) && !(a & 0x7fffffff)))
    return qnan;
  if (inf(a) || inf(b))
    return (std::uint32_t(sign) << 31) | 0x7f800000;
  const auto x = unpack(a), y = unpack(b);
  const auto p = std::uint64_t(x.sig) * y.sig;
  if (!p)
    return std::uint32_t(sign) << 31;
  const auto l = lead(p);
  return pack(sign, x.e + y.e + int(l) - 46, p << (63 - l));
}
inline std::uint16_t bf16(std::uint32_t a, bool flush = false) {
  if (nan(a))
    return 0x7fc0;
  const auto r = std::uint16_t((a + 0x7fff + ((a >> 16) & 1)) >> 16);
  return flush && !(r & 0x7f80) ? r & 0x8000 : r;
}
// MXM output quantization preserves the NaN sign and retained payload, with
// the quiet bit forced on. VXM arithmetic uses its separate canonical policy.
inline std::uint16_t mxm_bf16(std::uint32_t a) {
  if ((a & 0x7f800000u) == 0x7f800000u)
    return std::uint16_t(a >> 16) | (nan(a) ? 0x40 : 0);
  return bf16(a);
}
inline std::uint32_t divide(std::uint32_t a, std::uint32_t b) {
  const bool sign = (a ^ b) >> 31;
  const auto signed_zero = std::uint32_t(sign) << 31;
  if (nan(a) || nan(b) || (inf(a) && inf(b)) ||
      (!(a & 0x7fffffff) && !(b & 0x7fffffff)))
    return qnan;
  if (inf(a) || !(b & 0x7fffffff))
    return signed_zero | 0x7f800000;
  if (inf(b) || !(a & 0x7fffffff))
    return signed_zero;
  const auto x = unpack(a), y = unpack(b);
  const unsigned extra = x.sig < y.sig;
  const auto numerator = std::uint64_t(x.sig) << (26 + extra);
  auto q = numerator / y.sig;
  if (numerator % y.sig)
    q |= 1;
  return pack(sign, x.e - y.e - int(extra), q << 37);
}
inline std::uint32_t from_fp16(std::uint16_t a, bool flush = false) {
  const bool sign = a >> 15;
  const unsigned ex = (a >> 10) & 31, f = a & 1023;
  if (ex == 31)
    return (std::uint32_t(sign) << 31) | 0x7f800000 | (f << 13);
  if (!ex) {
    if (!f || flush)
      return std::uint32_t(sign) << 31;
    const auto l = lead(f);
    return pack(sign, int(l) - 24, std::uint64_t(f) << (63 - l));
  }
  return (std::uint32_t(sign) << 31) | ((ex + 112) << 23) | (f << 13);
}
inline std::uint16_t fp16(std::uint32_t a, bool flush = false) {
  const auto sign = std::uint16_t(a >> 16) & 0x8000;
  if (nan(a))
    return 0x7e00;
  if (inf(a))
    return sign | 0x7c00;
  const auto x = unpack(a);
  if (!x.sig)
    return sign;
  int e = x.e;
  std::uint64_t m = std::uint64_t(x.sig) << 40;
  if (e < -14) {
    m = jam(m, unsigned(-14 - e));
    e = -14;
  }
  unsigned r = unsigned(m >> 53);
  if ((m & (std::uint64_t(1) << 52)) &&
      ((m & ((std::uint64_t(1) << 52) - 1)) || (r & 1)))
    ++r;
  if (r & 0x800) {
    r >>= 1;
    ++e;
  }
  if (e > 15)
    return sign | 0x7c00;
  const unsigned ex = e == -14 && !(r & 0x400) ? 0 : unsigned(e + 15);
  if (!ex && flush)
    return sign;
  return sign | (ex << 10) | (r & 1023);
}
inline std::uint32_t signed_fixed(std::int64_t value, unsigned fraction) {
  const bool s = value < 0;
  const auto m =
      s ? std::uint64_t(0) - std::uint64_t(value) : std::uint64_t(value);
  if (!m)
    return 0;
  const unsigned l = lead(m);
  return pack(s, int(l) - int(fraction), m << (63 - l));
}
struct Dot {
  std::uint32_t bits = 0;
  bool alignment_lost = false;
};
// Common exponent and signed 64-bit sum. Out-of-window magnitudes are
// truncated before applying sign, exactly as the RTL (no sticky injection).
inline Dot dot(std::span<const std::uint16_t> a,
               std::span<const std::uint16_t> b) {
  if (a.empty() || a.size() != b.size() || a.size() > 32)
    throw std::invalid_argument("dot shape");
  bool positive_inf = false, negative_inf = false, invalid = false;
  int maximum = -1024;
  unsigned growth = 0;
  for (auto n = a.size() - 1; n; n >>= 1)
    ++growth;
  const unsigned position = 62 - growth;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const auto x = daz(std::uint32_t(a[i]) << 16),
               y = daz(std::uint32_t(b[i]) << 16);
    if (nan(x) || nan(y) || (inf(x) && !(y & 0x7fffffff)) ||
        (inf(y) && !(x & 0x7fffffff))) {
      invalid = true;
      continue;
    }
    if (inf(x) || inf(y)) {
      if ((x ^ y) >> 31)
        negative_inf = true;
      else
        positive_inf = true;
      continue;
    }
    const unsigned ex = (a[i] >> 7) & 255, ey = (b[i] >> 7) & 255;
    if (!ex || !ey)
      continue;
    const unsigned p = (128 + (a[i] & 127)) * (128 + (b[i] & 127));
    const int e = int(ex) + int(ey) - 254 - 14 + int(lead(p));
    if (e > maximum)
      maximum = e;
  }
  std::uint64_t sum = 0;
  bool lost = false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const unsigned ex = (a[i] >> 7) & 255, ey = (b[i] >> 7) & 255;
    if (!ex || !ey || ex == 255 || ey == 255)
      continue;
    const unsigned p = (128 + (a[i] & 127)) * (128 + (b[i] & 127)), l = lead(p);
    const int e = int(ex) + int(ey) - 254 - 14 + int(l);
    const unsigned distance = unsigned(maximum - e);
    auto m = std::uint64_t(p) << (position - l);
    if (distance >= 64) {
      lost = lost || m;
      m = 0;
    } else if (distance) {
      lost = lost || bool(m & ((std::uint64_t(1) << distance) - 1));
      m >>= distance;
    }
    sum += ((a[i] ^ b[i]) & 0x8000) ? std::uint64_t(0) - m : m;
  }
  if (invalid || (positive_inf && negative_inf))
    return {qnan, lost};
  if (positive_inf || negative_inf)
    return {(std::uint32_t(negative_inf) << 31) | 0x7f800000, lost};
  if (!sum)
    return {0, lost};
  const bool sign = sum >> 63;
  const auto m = sign ? std::uint64_t(0) - sum : sum;
  const auto l = lead(m);
  return {pack(sign, maximum + int(l) - int(position), m << (63 - l)), lost};
}
// An exact integer significand with a power-of-two exponent.
struct WeightOperand {
  std::uint32_t magnitude;
  int exponent;
  bool sign, nan = false, inf = false;
};

inline WeightOperand e4m3fn(std::uint8_t v) {
  const unsigned e = (v >> 3) & 15;
  return {(e ? 8u : 0u) + (v & 7u), e ? int(e) - 10 : -9,
          bool(v & 0x80), (v & 0x7f) == 0x7f};
}

inline std::uint16_t dequant(WeightFormat format, std::uint8_t q,
                             std::uint8_t zero, WeightScaleMode mode,
                             std::uint32_t scale, std::uint8_t block_scale) {
  WeightOperand weight{};
  switch (format) {
  case WeightFormat::Int8: {
    const int difference = int(std::int8_t(q)) - int(std::int8_t(zero));
    weight = {unsigned(difference < 0 ? -difference : difference), 0,
              difference < 0};
    break;
  }
  case WeightFormat::Fp4E2M1: {
    const unsigned e = (q >> 1) & 3;
    weight = {(e ? 2u : 0u) + (q & 1u), e ? int(e) - 2 : -1,
              bool(q & 8)};
    break;
  }
  case WeightFormat::Fp8E4M3Fn:
    weight = e4m3fn(q);
    break;
  default:
    throw std::invalid_argument("reserved weight format");
  }
  const auto block = mode == WeightScaleMode::NvFp4
                         ? e4m3fn(block_scale)
                         : WeightOperand{1, 0, false};
  WeightOperand factor;
  if (mode == WeightScaleMode::Ue8m0) {
    factor = {1, int(scale & 255) - 127, false, (scale & 255) == 255};
  } else {
    const unsigned e = (scale >> 23) & 255;
    factor = {(e ? 0x800000u : 0) + (scale & 0x7fffff),
              e ? int(e) - 150 : -149, bool(scale >> 31),
              nan(scale), inf(scale)};
  }
  const auto sign = std::uint16_t((weight.sign ^ block.sign ^ factor.sign) << 15);
  const auto magnitude =
      std::uint64_t(weight.magnitude) * block.magnitude * factor.magnitude;
  if (weight.nan || block.nan || factor.nan || (factor.inf && !magnitude))
    return 0x7fc0;
  if (factor.inf)
    return sign | 0x7f80;
  if (!magnitude)
    return sign;

  // Round directly to BF16, including the subnormal range. Keep the exact
  // integer product until this step to avoid FP32/BF16 double rounding.
  const int exponent = weight.exponent + block.exponent + factor.exponent;
  const int leading = int(lead(magnitude));
  int biased = leading + exponent + 127;
  const int shift = biased > 0 ? leading - 7 : -133 - exponent;
  if (shift >= 64)
    return sign;
  auto rounded = shift <= 0 ? magnitude << -shift : magnitude >> shift;
  if (shift > 0) {
    const auto half = std::uint64_t(1) << (shift - 1);
    const auto remainder = magnitude & ((half << 1) - 1);
    if (remainder > half || (remainder == half && (rounded & 1)))
      ++rounded;
  }
  if (biased <= 0)
    return sign | std::uint16_t(rounded);
  if (rounded == 256) {
    rounded >>= 1;
    ++biased;
  }
  return sign | (biased >= 255 ? 0x7f80 :
                                 std::uint16_t((biased << 7) | (rounded & 127)));
}

inline std::uint16_t dequant(std::uint8_t q, std::uint8_t zero,
                             std::uint16_t scale) {
  return dequant(WeightFormat::Int8, q, zero, WeightScaleMode::Bf16,
                 std::uint32_t(scale) << 16, 0);
}
} // namespace ftlpu::rtl::fp
