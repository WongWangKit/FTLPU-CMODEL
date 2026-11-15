#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ftlpu::timing {

enum class ClockPhase : std::uint8_t {
    LOW = 0,
    POSEDGE = 1,
    HIGH = 2,
    NEGEDGE = 3,
};

constexpr std::string_view phase_name(const ClockPhase phase) noexcept
{
    switch (phase) {
    case ClockPhase::LOW: return "LOW";
    case ClockPhase::POSEDGE: return "POSEDGE";
    case ClockPhase::HIGH: return "HIGH";
    case ClockPhase::NEGEDGE: return "NEGEDGE";
    }
    return "INVALID";
}

class ClockParticipant {
public:
    virtual ~ClockParticipant() = default;

    virtual void low() {}
    virtual void validate_posedge() const {}
    virtual void posedge() {}
    virtual void high() {}
    virtual void negedge() {}
};

class ClockedModule : public ClockParticipant {
public:
    void low() override = 0;
    void posedge() override = 0;
};

// Clock owns no modules. Registered modules must outlive it. Clock deliberately
// does not invoke low()/high(): those callbacks belong to SystemPhase's explicit
// data-dependency order. Clock only owns the global edge boundary.
class Clock {
public:
    Clock() = default;
    Clock(const Clock&) = delete;
    Clock& operator=(const Clock&) = delete;
    Clock(Clock&&) = delete;
    Clock& operator=(Clock&&) = delete;

    void register_module(ClockedModule& module)
    {
        if (started_)
            throw std::logic_error("clock module registration is closed after simulation starts");
        if (faulted_)
            throw std::logic_error("cannot register a module on a faulted clock");
        if (std::find(modules_.begin(), modules_.end(), &module) != modules_.end())
            throw std::logic_error("clock module registered more than once");
        modules_.push_back(&module);
    }

    [[nodiscard]] std::size_t module_count() const noexcept { return modules_.size(); }
    [[nodiscard]] std::uint64_t phase_count() const noexcept { return phase_count_; }
    [[nodiscard]] std::uint64_t cycle() const noexcept { return phase_count_ >> 2U; }
    [[nodiscard]] ClockPhase phase() const noexcept
    {
        return static_cast<ClockPhase>(phase_count_ & 3U);
    }
    [[nodiscard]] bool clk_level() const noexcept
    {
        return phase() == ClockPhase::POSEDGE || phase() == ClockPhase::HIGH;
    }
    [[nodiscard]] bool started() const noexcept { return started_; }
    [[nodiscard]] bool faulted() const noexcept { return faulted_; }

    void require(const ClockPhase expected) const
    {
        if (faulted_)
            throw std::logic_error("faulted clock cannot continue");
        if (phase() != expected)
            throw std::logic_error("illegal clock phase transition");
    }

    // SystemPhase::low() must have completed exactly once before this call.
    void complete_low()
    {
        require(ClockPhase::LOW);
        started_ = true;
        posedge_validated_ = false;
        ++phase_count_;
    }

    void validate_all()
    {
        require(ClockPhase::POSEDGE);
        if (posedge_validated_)
            throw std::logic_error("posedge validation executed more than once");
        try {
            for (const auto* module : modules_) module->validate_posedge();
            posedge_validated_ = true;
        } catch (...) {
            faulted_ = true;
            throw;
        }
    }

    void commit_all()
    {
        require(ClockPhase::POSEDGE);
        if (!posedge_validated_)
            throw std::logic_error("posedge commit requires a completed validation barrier");
        try {
            for (auto* module : modules_) module->posedge();
        } catch (...) {
            faulted_ = true;
            throw;
        }
        posedge_validated_ = false;
        ++phase_count_;
    }

    // SystemPhase::high() runs while the clock remains in HIGH. It must finish
    // before this transition; Clock never calls module.high() itself.
    void complete_high()
    {
        require(ClockPhase::HIGH);
        ++phase_count_;
    }

    void negedge_all()
    {
        require(ClockPhase::NEGEDGE);
        try {
            for (auto* module : modules_) module->negedge();
        } catch (...) {
            faulted_ = true;
            throw;
        }
        ++phase_count_;
    }

    void mark_faulted() noexcept { faulted_ = true; }

    // Retain the registration/lifetime relationship while restarting physical
    // time. A reset is only legal at the complete-cycle LOW boundary.
    void reset_time()
    {
        if (phase() != ClockPhase::LOW && !faulted_)
            throw std::logic_error("clock time reset requires the LOW cycle boundary");
        phase_count_ = 0;
        posedge_validated_ = false;
        faulted_ = false;
    }

private:
    std::vector<ClockedModule*> modules_{};
    std::uint64_t phase_count_{};
    bool started_{};
    bool faulted_{};
    bool posedge_validated_{};
};

// Ordered LOW/HIGH combination scheduler. Its order expresses RTL topology and
// is intentionally independent of Clock's module registration/commit order.
class SystemPhase {
public:
    using Callback = std::function<void()>;

    struct Node {
        std::string name{};
        Callback low{};
        Callback high{};
    };

    void add_node(std::string name, Callback low, Callback high = {})
    {
        if (started_)
            throw std::logic_error("system-phase registration is closed after simulation starts");
        if (faulted_)
            throw std::logic_error("cannot register a node on a faulted system phase");
        if (name.empty()) throw std::invalid_argument("system-phase node name cannot be empty");
        if (std::find_if(nodes_.begin(), nodes_.end(), [&](const Node& node) {
                return node.name == name;
            }) != nodes_.end())
            throw std::logic_error("system-phase node registered more than once");
        nodes_.push_back(Node {std::move(name), std::move(low), std::move(high)});
    }

    void low()
    {
        if (faulted_) throw std::logic_error("faulted system phase cannot continue");
        if (state_ != State::ReadyForLow || running_)
            throw std::logic_error("SystemPhase LOW executed out of order or more than once");
        started_ = true;
        running_ = true;
        try {
            for (auto& node : nodes_) if (node.low) node.low();
        } catch (...) {
            running_ = false;
            faulted_ = true;
            throw;
        }
        running_ = false;
        state_ = State::ReadyForHigh;
        ++low_count_;
    }

    void high()
    {
        if (faulted_) throw std::logic_error("faulted system phase cannot continue");
        if (state_ != State::ReadyForHigh || running_)
            throw std::logic_error("SystemPhase HIGH executed without one completed LOW");
        running_ = true;
        try {
            for (auto& node : nodes_) if (node.high) node.high();
        } catch (...) {
            running_ = false;
            faulted_ = true;
            throw;
        }
        running_ = false;
        state_ = State::ReadyForLow;
        ++high_count_;
    }

    [[nodiscard]] std::size_t node_count() const noexcept { return nodes_.size(); }
    [[nodiscard]] std::uint64_t low_count() const noexcept { return low_count_; }
    [[nodiscard]] std::uint64_t high_count() const noexcept { return high_count_; }
    [[nodiscard]] bool ready_for_low() const noexcept
    {
        return !running_ && state_ == State::ReadyForLow;
    }
    [[nodiscard]] bool faulted() const noexcept { return faulted_; }

    void mark_faulted() noexcept { faulted_ = true; }

    void reset()
    {
        if (running_) throw std::logic_error("cannot reset a running SystemPhase");
        state_ = State::ReadyForLow;
        low_count_ = 0;
        high_count_ = 0;
        faulted_ = false;
    }

private:
    enum class State : std::uint8_t { ReadyForLow, ReadyForHigh };

    std::vector<Node> nodes_{};
    State state_{State::ReadyForLow};
    std::uint64_t low_count_{};
    std::uint64_t high_count_{};
    bool started_{};
    bool running_{};
    bool faulted_{};
};

template <typename T>
class ClockedState {
public:
    ClockedState() = default;
    explicit ClockedState(T initial)
        : current_(std::move(initial)), next_(current_)
    {}

    [[nodiscard]] const T& current() const noexcept { return current_; }
    [[nodiscard]] const T& next() const noexcept { return next_; }
    [[nodiscard]] bool next_prepared() const noexcept { return next_prepared_; }
    [[nodiscard]] std::uint64_t commit_count() const noexcept { return commit_count_; }

    void prepare()
    {
        if (next_prepared_)
            throw std::logic_error("ClockedState Next prepared more than once");
        next_ = current_;
        next_prepared_ = true;
    }

    T& next_value()
    {
        if (!next_prepared_)
            throw std::logic_error("ClockedState Next modified before prepare");
        return next_;
    }

    void validate_commit() const
    {
        if (!next_prepared_)
            throw std::logic_error("ClockedState committed without prepared Next");
    }

    void commit()
    {
        validate_commit();
        current_ = next_;
        next_prepared_ = false;
        ++commit_count_;
    }

    void reset(T value = {})
    {
        current_ = std::move(value);
        next_ = current_;
        next_prepared_ = false;
        commit_count_ = 0;
    }

private:
    T current_{};
    T next_{};
    bool next_prepared_{};
    std::uint64_t commit_count_{};
};

template <typename T>
class Register final : public ClockedModule {
public:
    explicit Register(T reset_value = {}, T initial_value = {})
        : state_(std::move(initial_value))
        , reset_value_(std::move(reset_value))
        , input_(state_.current())
    {}

    void drive(const T& input, const bool enable = true, const bool reset = false)
    {
        input_ = input;
        enable_ = enable;
        reset_ = reset;
    }

    void low() override
    {
        state_.prepare();
        if (reset_) state_.next_value() = reset_value_;
        else if (enable_) state_.next_value() = input_;
    }

    void validate_posedge() const override { state_.validate_commit(); }
    void posedge() override { state_.commit(); }

    [[nodiscard]] const T& q() const noexcept { return state_.current(); }
    [[nodiscard]] const T& next() const noexcept { return state_.next(); }

private:
    ClockedState<T> state_{};
    T reset_value_{};
    T input_{};
    bool enable_{true};
    bool reset_{};
};

template <typename T, std::size_t N>
class RegisterBank final : public ClockedModule {
public:
    using Value = std::array<T, N>;

    explicit RegisterBank(Value reset_value = {}, Value initial_value = {})
        : state_(std::move(initial_value))
        , reset_value_(std::move(reset_value))
        , input_(state_.current())
    {}

    void drive(const Value& input, const bool enable = true, const bool reset = false)
    {
        input_ = input;
        enable_ = enable;
        reset_ = reset;
    }

    void low() override
    {
        state_.prepare();
        if (reset_) state_.next_value() = reset_value_;
        else if (enable_) state_.next_value() = input_;
    }

    void validate_posedge() const override { state_.validate_commit(); }
    void posedge() override { state_.commit(); }

    [[nodiscard]] const Value& q() const noexcept { return state_.current(); }
    [[nodiscard]] const Value& next() const noexcept { return state_.next(); }

private:
    ClockedState<Value> state_{};
    Value reset_value_{};
    Value input_{};
    bool enable_{true};
    bool reset_{};
};

} // namespace ftlpu::timing
