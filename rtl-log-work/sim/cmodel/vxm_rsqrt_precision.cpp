#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>

namespace {

double quantize_rne(double value, int fraction_bits) {
  return std::nearbyint(std::ldexp(value, fraction_bits)) /
         std::ldexp(1.0, fraction_bits);
}

struct Coefficient {
  double k;
  double b;
};

Coefficient coefficient_for(int parity, int segment, int active_fraction) {
  const double scale = parity ? 2.0 : 1.0;
  const double x0 = scale * (1.0 + static_cast<double>(segment) / 32.0);
  const double x1 = scale *
                    (1.0 + static_cast<double>(segment + 1) / 32.0);
  const double b = 1.0 / std::sqrt(x0);
  const double f1 = 1.0 / std::sqrt(x1);
  // dx_encoded always has the original format's fraction denominator.
  // Folding scale into k lets both parity halves share the fixed datapath.
  const double k = (b - f1) * 32.0;
  const double stored_k = quantize_rne(k, 15);
  const double stored_b = quantize_rne(b, 15);
  return {
    quantize_rne(stored_k, active_fraction),
    quantize_rne(stored_b, active_fraction)
  };
}

void check_low_precision(const char* name, int fraction_bits,
                         int coefficient_fraction) {
  double max_seed_relative_error = 0.0;
  for (int parity = 0; parity < 2; ++parity) {
    const std::uint32_t fraction_count = 1u << fraction_bits;
    const std::uint32_t residual_mask =
      (1u << (fraction_bits - 5)) - 1u;
    for (std::uint32_t fraction = 0; fraction < fraction_count; ++fraction) {
      const int segment = static_cast<int>(fraction >> (fraction_bits - 5));
      const std::uint32_t residual = fraction & residual_mask;
      const auto coefficient =
        coefficient_for(parity, segment, coefficient_fraction);
      const double dx = static_cast<double>(residual) /
                        static_cast<double>(fraction_count);
      const double y0 = quantize_rne(
        coefficient.b - coefficient.k * dx, coefficient_fraction);
      const double x = (parity ? 2.0 : 1.0) *
                       (1.0 + static_cast<double>(fraction) /
                              static_cast<double>(fraction_count));
      const double reference = 1.0 / std::sqrt(x);
      max_seed_relative_error = std::max(
        max_seed_relative_error, std::abs(y0-reference) / reference);
    }
  }
  std::cout << name << " max linear relative error = "
            << std::scientific << max_seed_relative_error << '\n';
}

int check_fp32_exhaustive() {
  constexpr int fraction_bits = 23;
  constexpr std::uint32_t fraction_count = 1u << fraction_bits;
  constexpr std::uint32_t residual_mask = (1u << 18) - 1u;
  double max_seed_relative_error = 0.0;
  double max_newton_relative_error = 0.0;
  std::uint32_t max_ulp_error = 0;
  std::uint64_t over_one_ulp = 0;

  for (int parity = 0; parity < 2; ++parity) {
    for (std::uint32_t fraction = 0; fraction < fraction_count; ++fraction) {
      const int segment = static_cast<int>(fraction >> 18);
      const std::uint32_t residual = fraction & residual_mask;
      const auto coefficient = coefficient_for(parity, segment, 15);
      const double dx = static_cast<double>(residual) /
                        static_cast<double>(fraction_count);
      const float y0 = static_cast<float>(quantize_rne(
        coefficient.b - coefficient.k * dx, 15));
      const float m = static_cast<float>((parity ? 2.0 : 1.0) *
        (1.0 + static_cast<double>(fraction) /
               static_cast<double>(fraction_count)));
      const float y_squared = y0 * y0;
      const float correction = std::fma(-0.5f * m, y_squared, 1.5f);
      const float result = y0 * correction;
      const float reference = 1.0f / std::sqrt(m);
      const double exact_reference = 1.0 / std::sqrt(static_cast<double>(m));
      max_seed_relative_error = std::max(max_seed_relative_error,
        std::abs(static_cast<double>(y0)-exact_reference) / exact_reference);
      max_newton_relative_error = std::max(max_newton_relative_error,
        std::abs(static_cast<double>(result)-exact_reference) /
        exact_reference);
      const std::uint32_t result_bits = std::bit_cast<std::uint32_t>(result);
      const std::uint32_t reference_bits =
        std::bit_cast<std::uint32_t>(reference);
      const std::uint32_t ulp_error = result_bits > reference_bits ?
        result_bits-reference_bits : reference_bits-result_bits;
      max_ulp_error = std::max(max_ulp_error, ulp_error);
      if (ulp_error > 1)
        ++over_one_ulp;
    }
  }

  std::cout << "FP32 max linear relative error = " << std::scientific
            << max_seed_relative_error << '\n'
            << "FP32 max Newton relative error = "
            << max_newton_relative_error << '\n'
            << "FP32 max ULP error = " << std::dec << max_ulp_error << '\n'
            << "FP32 cases over 1 ULP = " << over_one_ulp << '\n';
  // The inference target is a faithful FP32 approximation, not a
  // correctly-rounded libm replacement. The accepted architectural bound is
  // two ULP after one Newton step over every normalized FP32 mantissa.
  return max_ulp_error <= 2 ? 0 : 1;
}

}  // namespace

int main() {
  std::fesetround(FE_TONEAREST);
  check_low_precision("BF16", 7, 9);
  check_low_precision("FP16", 10, 12);
  return check_fp32_exhaustive();
}
