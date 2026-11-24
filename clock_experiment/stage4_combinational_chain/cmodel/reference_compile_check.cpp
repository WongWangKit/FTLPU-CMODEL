#include "cpp_timing_modeling.hpp"

#include <cstdint>

int main() {
    ftlpu::timing::FourPhaseClock clock;
    clock.require(ftlpu::timing::ClockPhase::LowEvaluate);
    const auto result = ftlpu::timing::evaluate_mac(std::uint8_t{7});
    ftlpu::timing::CurrentNext<std::uint8_t> state;
    state.prepare_next(result.result8, clock.phase());
    clock.advance_from(ftlpu::timing::ClockPhase::LowEvaluate);
    state.commit(clock.phase());
    return state.current() == result.result8 ? 0 : 1;
}
