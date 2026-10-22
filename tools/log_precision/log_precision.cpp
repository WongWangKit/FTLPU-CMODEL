// Standalone architecture experiment. Does not change the RTL or CModel.
// Compile without FP contraction: every float multiply/add is a separate RNE.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

constexpr double LN2 = 0.693147180559945309417232121458176568;

float half_round(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto sign = bits & 0x80000000u;
    auto mag = bits & 0x7fffffffu;
    if (mag >= 0x47800000u) return std::bit_cast<float>(sign | 0x7f800000u);
    if (mag < 0x38800000u) { // Match FTZ after RNE, including carry into normal.
        const double scaled = std::ldexp(std::fabs(static_cast<double>(value)), 24);
        const auto rounded = static_cast<std::uint32_t>(std::nearbyint(scaled));
        return rounded >= 1024 ? std::bit_cast<float>(sign | 0x38800000u)
                               : std::bit_cast<float>(sign);
    }
    mag += 0xfffu + ((mag >> 13) & 1u);
    mag &= ~0x1fffu;
    if (mag >= 0x47800000u) mag = 0x7f800000u;
    return std::bit_cast<float>(sign | mag);
}

float bf_round(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    if ((bits & 0x7f800000u) == 0) bits &= 0x80000000u;
    return std::bit_cast<float>(bits);
}

std::uint32_t ordered(float value, int format) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits = (bits & 0x80000000u) ? ~bits : bits | 0x80000000u;
    if (format == 16) return bits >> 13;
    if (format == 7) return bits >> 16;
    return bits;
}

float output_round(float value, int format) {
    return format == 16 ? half_round(value) : format == 7 ? bf_round(value) : value;
}

struct Coefficient { float logc; float invc; };
struct Plan {
    std::string name;
    bool centered;
    int degree;
    int coefficient_fraction; // -1: FP32, -2: FP16, -3: BF16; otherwise fixed.
    bool split_ln2;
    std::array<Coefficient, 64> table;
    bool fused_final = false;
    int arithmetic_format = 32;
    bool near_one = false;
};

float coefficient_round(double x, int fraction) {
    if (fraction == -2) return half_round(static_cast<float>(x));
    if (fraction == -3) return bf_round(static_cast<float>(x));
    if (fraction < 0) return static_cast<float>(x);
    return static_cast<float>(std::ldexp(std::nearbyint(std::ldexp(x, fraction)), -fraction));
}

Plan make_plan(std::string name, bool centered, int degree, int fraction, bool split) {
    Plan p{name, centered, degree, fraction, split, {}};
    for (int i = 0; i < 64; ++i) {
        double c = 1.0 + static_cast<double>(i) / 64;
        if (centered && i >= 32) c *= 0.5;
        p.table[i] = {coefficient_round(std::log(c), fraction),
                      coefficient_round(1.0 / c, fraction)};
    }
    return p;
}

float evaluate(float x, const Plan& p) {
    const auto bits = std::bit_cast<std::uint32_t>(x);
    const auto fraction = bits & 0x7fffffu;
    int e = static_cast<int>((bits >> 23) & 255) - 127;
    int index = static_cast<int>(fraction >> 17);
    float m = std::bit_cast<float>(0x3f800000u | fraction);
    float c;
    if (p.centered) {
        // Round the six-bit segment index using the first residual bit.
        // Map anchors >=1.5 to half their value. Index 64 wraps to c=1.
        index += (fraction >> 16) & 1u;
        c = 1.0f + static_cast<float>(index) / 64;
        if (index >= 32) { c *= 0.5f; m *= 0.5f; ++e; }
        index &= 63;
    } else {
        c = 1.0f + static_cast<float>(index) / 64;
    }
    if (p.near_one && x >= 0.984375f && x <= 1.015625f) {
        // Reuse the same Horner evaluator with c=1 and inv(c)=1. This
        // interval uses no inexact LUT base or inverse and no exponent term.
        m = x;
        c = 1.0f;
        index = 0;
        e = 0;
    }
    const float delta = m - c; // Exact for these nearby binary numbers.
    const float u = output_round(delta * p.table[index].invc, p.arithmetic_format);
    float h = (p.degree % 2 ? 1.0f : -1.0f) / static_cast<float>(p.degree);
    for (int k = p.degree - 1; k >= 1; --k) {
        const float product = output_round(h * u, p.arithmetic_format);
        h = output_round(product + output_round(
            (k % 2 ? 1.0f : -1.0f) / static_cast<float>(k), p.arithmetic_format), p.arithmetic_format);
    }
    const float polynomial = output_round(u * h, p.arithmetic_format);
    const float local = p.fused_final ? std::fma(u, h, p.table[index].logc)
        : output_round(p.table[index].logc + polynomial, p.arithmetic_format);
    if (p.split_ln2) {
        constexpr float hi = 0x1.62e000p-1f;
        const float lo = static_cast<float>(LN2 - static_cast<double>(hi));
        const float tail = static_cast<float>(e) * lo;
        const float small = local + tail;
        const float large = static_cast<float>(e) * hi;
        return large + small;
    }
    const float correction = static_cast<float>(e) * static_cast<float>(LN2);
    return local + correction;
}

struct Stats {
    std::uint64_t count = 0, over1 = 0, over2 = 0, exact = 0;
    std::uint32_t maxulp = 0, worst = 0;
    double maxabs = 0, maxrel = 0;
    void observe(float x, float result, double reference, int format) {
        result = output_round(result, format);
        const float target = output_round(static_cast<float>(reference), format);
        const auto a = ordered(result, format), b = ordered(target, format);
        const auto ulp = a > b ? a - b : b - a;
        ++count; exact += ulp == 0; over1 += ulp > 1; over2 += ulp > 2;
        if (ulp > maxulp) { maxulp = ulp; worst = std::bit_cast<std::uint32_t>(x); }
        const double err = std::fabs(static_cast<double>(result) - reference);
        maxabs = std::max(maxabs, err);
        if (reference != 0) maxrel = std::max(maxrel, err / std::fabs(reference));
    }
};

void print(const char* scope, const Plan& p, const Stats& s) {
    std::cout << scope << ',' << p.name << ',' << s.count << ',' << s.maxulp
              << ',' << s.over1 << ',' << s.over2 << ',' << s.exact
              << ',' << std::scientific << std::setprecision(9) << s.maxabs
              << ',' << s.maxrel << ",0x" << std::hex << s.worst << std::dec << '\n';
}

void low_precision(int format) {
    std::vector<Plan> plans;
    for (int degree : {1, 2, 3})
        for (int fraction : {format == 7 ? -3 : -2, 15, 20, 25, -1}) {
            const std::string precision = fraction == -1 ? "fp32" : fraction == -2 ? "fp16" : fraction == -3 ? "bf16" : "q" + std::to_string(fraction);
            plans.push_back(make_plan("center_d" + std::to_string(degree) + "_" + precision, true, degree, fraction, true));
        }
    for (int degree : {1, 2, 3}) {
        auto p = make_plan("center_d" + std::to_string(degree) + "_native_coeff_native_local", true,
            degree, format == 7 ? -3 : -2, true);
        p.arithmetic_format = format;
        plans.push_back(p);
    }
    std::vector<Stats> stats(plans.size());
    const int fb = format == 16 ? 10 : 7;
    const int maxe = format == 16 ? 30 : 254;
    for (int be = 1; be <= maxe; ++be)
        for (int f = 0; f < (1 << fb); ++f) {
            const float x = format == 16
                ? std::ldexp(1.0f + static_cast<float>(f) / 1024, be - 15)
                : std::bit_cast<float>(static_cast<std::uint32_t>((be << 23) | (f << 16)));
            const double reference = std::log(static_cast<double>(x));
            for (std::size_t i = 0; i < plans.size(); ++i)
                stats[i].observe(x, evaluate(x, plans[i]), reference, format);
        }
    for (std::size_t i = 0; i < plans.size(); ++i) print(format == 16 ? "FP16_ALL_NORMAL" : "BF16_ALL_NORMAL", plans[i], stats[i]);
}

int main(int argc, char**) {
    std::cout << "scope,plan,count,max_ulp,over_1ulp,over_2ulp,correctly_rounded,max_abs,max_rel,worst_x_fp32_bits\n";
    if (argc > 1) {
        std::vector<Plan> full_plans;
        full_plans.push_back(make_plan("center_d4_fp32_split", true, 4, -1, true));
        auto fused = full_plans[0];
        fused.name = "center_d4_fp32_split_final_fma";
        fused.fused_final = true;
        full_plans.push_back(fused);
        if (argc > 2) {
            full_plans.erase(full_plans.begin(), full_plans.end());
            auto near = make_plan("center_d4_fp32_split_near1", true, 4, -1, true);
            near.near_one = true;
            full_plans.push_back(near);
        }
        std::vector<Stats> stats(full_plans.size());
        for (std::uint32_t be = 1; be < 255; ++be) {
            for (std::uint32_t f = 0; f < (1u << 23); ++f) {
                const float x = std::bit_cast<float>((be << 23) | f);
                const double reference = std::log(static_cast<double>(x));
                for (std::size_t i = 0; i < full_plans.size(); ++i)
                    stats[i].observe(x, evaluate(x, full_plans[i]), reference, 32);
            }
            if (be % 32 == 0) std::cerr << "Completed normal exponent " << be << "/254\n";
        }
        for (std::size_t i = 0; i < full_plans.size(); ++i)
            print("FP32_ALL_POSITIVE_NORMAL", full_plans[i], stats[i]);
        return 0;
    }
    low_precision(7);
    low_precision(16);
    std::vector<Plan> plans;
    for (int d : {1, 2, 3, 4}) plans.push_back(make_plan("left_d" + std::to_string(d) + "_fp32", false, d, -1, false));
    for (int d : {1, 2, 3, 4}) plans.push_back(make_plan("center_d" + std::to_string(d) + "_fp32_split", true, d, -1, true));
    for (int q : {15, 20, 25, 30}) plans.push_back(make_plan("center_d4_q" + std::to_string(q) + "_split", true, 4, q, true));
    auto fused = make_plan("center_d4_fp32_split_final_fma", true, 4, -1, true);
    fused.fused_final = true;
    plans.push_back(fused);
    auto near = make_plan("center_d4_fp32_split_near1", true, 4, -1, true);
    near.near_one = true;
    plans.push_back(near);
    std::vector<Stats> exhaustive(plans.size()), exponent_scan(plans.size());
    for (std::uint32_t f = 0; f < (1u << 23); ++f) {
        for (int e : {-1, 0, 1}) {
            const float x = std::bit_cast<float>((static_cast<std::uint32_t>(127 + e) << 23) | f);
            const double reference = std::log(static_cast<double>(x));
            for (std::size_t i = 0; i < plans.size(); ++i)
                exhaustive[i].observe(x, evaluate(x, plans[i]), reference, 32);
        }
    }
    // Cover every normal exponent with a deterministic fraction grid plus
    // immediate neighbors of each segment boundary, midpoint and wrap point.
    std::vector<std::uint32_t> fractions;
    for (std::uint32_t f = 0; f < (1u << 23); f += 1024) fractions.push_back(f);
    for (std::uint32_t anchor = 0; anchor <= (1u << 23); anchor += (1u << 16))
        for (int offset : {-1, 0, 1}) {
            const auto f = static_cast<std::int64_t>(anchor) + offset;
            if (f >= 0 && f < (1u << 23)) fractions.push_back(static_cast<std::uint32_t>(f));
        }
    std::sort(fractions.begin(), fractions.end());
    fractions.erase(std::unique(fractions.begin(), fractions.end()), fractions.end());
    for (std::uint32_t be = 1; be < 255; ++be)
        for (auto f : fractions) {
            const float x = std::bit_cast<float>((be << 23) | f);
            const double reference = std::log(static_cast<double>(x));
            for (std::size_t i = 0; i < plans.size(); ++i)
                exponent_scan[i].observe(x, evaluate(x, plans[i]), reference, 32);
        }
    for (std::size_t i = 0; i < plans.size(); ++i) {
        print("FP32_ALL_MANTISSAS_E_MINUS1_0_1", plans[i], exhaustive[i]);
        print("FP32_ALL_EXPONENTS_STRATIFIED", plans[i], exponent_scan[i]);
    }
    return 0;
}
