// Reuse the independent numerical evaluator, not the production CModel.
#define main baseline_main
#include "log_precision.cpp"
#undef main

Plan fixed_plan(int log_fraction, int inverse_fraction, int degree, int arithmetic = 32) {
    auto p = make_plan("B" + std::to_string(log_fraction) + "_I" + std::to_string(inverse_fraction)
        + "_d" + std::to_string(degree), true, degree, log_fraction, true);
    for (int i = 0; i < 64; ++i) {
        double c = 1.0 + static_cast<double>(i) / 64;
        if (i >= 32) c *= 0.5;
        p.table[i].invc = coefficient_round(1.0 / c, inverse_fraction);
    }
    p.arithmetic_format = arithmetic;
    p.near_one = arithmetic == 32;
    return p;
}

int main(int argc, char**) {
    std::cout << "scope,plan,count,max_ulp,over_1ulp,over_2ulp,correctly_rounded,max_abs,max_rel,worst_x_fp32_bits\n";
    if (argc > 1) {
        const auto p = fixed_plan(28, 21, 4);
        Stats s;
        for (std::uint32_t be = 1; be < 255; ++be) {
            for (std::uint32_t f = 0; f < (1u << 23); ++f) {
                const float x = std::bit_cast<float>((be << 23) | f);
                s.observe(x, evaluate(x, p), std::log(static_cast<double>(x)), 32);
            }
            if (be % 32 == 0) std::cerr << "Completed exponent " << be << "/254\n";
        }
        print("FP32_ALL_POSITIVE_NORMAL", p, s);
        return 0;
    }
    for (int format : {7, 16}) {
        std::vector<Plan> plans;
        for (int b : {9, 12, 13, 14, 15, 16, 17, 18, 20})
            for (int inv : {9, 12, 15}) plans.push_back(fixed_plan(b, inv, format == 7 ? 1 : 2, format));
        std::vector<Stats> stats(plans.size());
        const int fb = format == 7 ? 7 : 10;
        const int maxe = format == 7 ? 254 : 30;
        for (int be = 1; be <= maxe; ++be)
            for (int f = 0; f < (1 << fb); ++f) {
                const float x = format == 7 ? std::bit_cast<float>(static_cast<std::uint32_t>((be << 23) | (f << 16)))
                    : std::ldexp(1.0f + static_cast<float>(f) / 1024, be - 15);
                const double reference = std::log(static_cast<double>(x));
                for (std::size_t i = 0; i < plans.size(); ++i) stats[i].observe(x, evaluate(x, plans[i]), reference, format);
            }
        for (std::size_t i = 0; i < plans.size(); ++i) print(format == 7 ? "BF16_ALL_NORMAL" : "FP16_ALL_NORMAL", plans[i], stats[i]);
    }
    std::vector<Plan> plans;
    for (int b : {26, 27, 28, 29, 30, 31})
        for (int inv : {18, 19, 20, 21, 22, 23, 24, 25}) plans.push_back(fixed_plan(b, inv, 4));
    std::vector<Stats> stats(plans.size());
    for (std::uint32_t f = 0; f < (1u << 23); ++f)
        for (int e : {-1, 0, 1}) {
            const float x = std::bit_cast<float>((static_cast<std::uint32_t>(127 + e) << 23) | f);
            const double reference = std::log(static_cast<double>(x));
            for (std::size_t i = 0; i < plans.size(); ++i) stats[i].observe(x, evaluate(x, plans[i]), reference, 32);
        }
    for (std::size_t i = 0; i < plans.size(); ++i) print("FP32_ALL_MANTISSAS_E_MINUS1_0_1", plans[i], stats[i]);
}
