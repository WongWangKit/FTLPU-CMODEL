#include "ftlpu/core/timing.hpp"

#include <array>
#include <cassert>
#include <stdexcept>
#include <vector>

namespace {

using ftlpu::timing::Clock;
using ftlpu::timing::ClockPhase;
using ftlpu::timing::ClockedState;
using ftlpu::timing::Register;
using ftlpu::timing::RegisterBank;
using ftlpu::timing::SystemPhase;

template <typename Function>
void rejects(Function&& function)
{
    bool rejected = false;
    try { function(); } catch (const std::logic_error&) { rejected = true; }
    assert(rejected);
}

class Scenario {
public:
    explicit Scenario(bool reverse_commit_registration)
    {
        if (reverse_commit_registration) {
            clock_.register_module(destination_);
            clock_.register_module(bank_);
            clock_.register_module(source_);
        } else {
            clock_.register_module(source_);
            clock_.register_module(bank_);
            clock_.register_module(destination_);
        }

        phases_.add_node("source register LOW", [this] {
            source_.drive(input_, enable_, reset_);
            source_.low();
            bank_.drive({input_, input_ + 1}, enable_, reset_);
            bank_.low();
            ++low_side_effects_;
        });
        phases_.add_node("combinational A", [this] {
            low_order_.push_back(1);
            a_ = source_.q() * 3;
        }, [this] {
            high_order_.push_back(1);
            high_a_ = source_.q() * 3;
        });
        phases_.add_node("combinational B", [this] {
            low_order_.push_back(2);
            b_ = a_ + 7;
        }, [this] {
            high_order_.push_back(2);
            high_b_ = high_a_ + 7;
        });
        phases_.add_node("destination register LOW", [this] {
            destination_.drive(b_, true, reset_);
            destination_.low();
        });
    }

    struct Result {
        int source{};
        int destination{};
        std::array<int, 2> bank{};
        int high_combination{};
        std::vector<int> low_order{};
        std::vector<int> high_order{};
        std::uint64_t cycle{};
        bool operator==(const Result&) const = default;
    };

    Result run(int input, bool enable = true, bool reset = false)
    {
        input_ = input;
        enable_ = enable;
        reset_ = reset;
        low_order_.clear();
        high_order_.clear();

        const auto source_before = source_.q();
        phases_.low();
        assert(clock_.phase() == ClockPhase::LOW);
        assert(source_.q() == source_before);
        const auto destination_prepared = destination_.next();

        clock_.complete_low();
        clock_.validate_all();
        clock_.commit_all();
        assert(clock_.phase() == ClockPhase::HIGH);
        const auto destination_committed = destination_.q();

        phases_.high();
        assert(destination_.q() == destination_committed);
        assert(destination_.next() == destination_prepared);
        clock_.complete_high();
        clock_.negedge_all();

        assert(low_order_ == std::vector<int>({1, 2}));
        assert(high_order_ == std::vector<int>({1, 2}));
        assert(low_side_effects_ == clock_.cycle());
        return {source_.q(), destination_.q(), bank_.q(), high_b_,
            low_order_, high_order_, clock_.cycle()};
    }

private:
    Register<int> source_{};
    Register<int> destination_{};
    RegisterBank<int, 2> bank_{};
    Clock clock_{};
    SystemPhase phases_{};
    int input_{};
    int a_{};
    int b_{};
    int high_a_{};
    int high_b_{};
    std::uint64_t low_side_effects_{};
    bool enable_{true};
    bool reset_{};
    std::vector<int> low_order_{};
    std::vector<int> high_order_{};
};

} // namespace

int main()
{
    ClockedState<int> state(4);
    rejects([&] { state.commit(); });
    state.prepare();
    assert(state.current() == 4 && state.next() == 4);
    state.next_value() = 9;
    state.commit();
    assert(state.current() == 9 && state.commit_count() == 1);
    rejects([&] { state.commit(); });

    Register<int> registration_probe;
    Clock registration_clock;
    registration_clock.register_module(registration_probe);
    rejects([&] { registration_clock.register_module(registration_probe); });
    rejects([&] { registration_clock.validate_all(); });

    Scenario forward(false);
    Scenario reverse(true);
    const auto first_forward = forward.run(5);
    const auto first_reverse = reverse.run(5);
    assert(first_forward == first_reverse);
    // Destination sampled Source's old Current (0), while HIGH observed 5.
    assert(first_forward.source == 5);
    assert(first_forward.destination == 7);
    assert(first_forward.high_combination == 22);
    assert((first_forward.bank == std::array<int, 2>({5, 6})));

    const auto held_forward = forward.run(99, false);
    const auto held_reverse = reverse.run(99, false);
    assert(held_forward == held_reverse);
    assert(held_forward.source == 5 && held_forward.bank[0] == 5);
    assert(held_forward.destination == 22);

    const auto reset_forward = forward.run(123, true, true);
    const auto reset_reverse = reverse.run(123, true, true);
    assert(reset_forward == reset_reverse);
    assert(reset_forward.source == 0 && reset_forward.destination == 0);
    assert((reset_forward.bank == std::array<int, 2>({0, 0})));
    return 0;
}

