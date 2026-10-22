#define main baseline_main
#include "log_precision.cpp"
#undef main

int main() {
    std::cout << "scope,plan,count,max_ulp,over_1ulp,over_2ulp,correctly_rounded,max_abs,max_rel,worst_x_fp32_bits\n";
    std::vector<Plan> plans;
    for (int b : {22, 24, 25, 26, 27, 28, 29, 30})
        for (int inv : {18, 20, 22}) {
            auto p = make_plan("left_B" + std::to_string(b) + "_I" + std::to_string(inv), false, 4, b, true);
            for (int i = 0; i < 64; ++i)
                p.table[i].invc = coefficient_round(1.0 / (1.0 + static_cast<double>(i) / 64), inv);
            p.near_one = true;
            plans.push_back(p);
        }
    std::vector<Stats> stats(plans.size());
    for (std::uint32_t f = 0; f < (1u << 23); ++f)
        for (int e : {-1, 0, 1}) {
            const float x = std::bit_cast<float>((static_cast<std::uint32_t>(127 + e) << 23) | f);
            const double ref = std::log(static_cast<double>(x));
            for (std::size_t i = 0; i < plans.size(); ++i) stats[i].observe(x, evaluate(x, plans[i]), ref, 32);
        }
    for (std::size_t i = 0; i < plans.size(); ++i) print("FP32_MANTISSAS_E_MINUS1_0_1", plans[i], stats[i]);
}
