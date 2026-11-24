#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace ftlpu::timing {

enum class ClockPhase : std::uint8_t {
    LowEvaluate = 0,
    PosedgeCommit = 1,
    PostEdgeSettle = 2,
    Negedge = 3,
};

constexpr std::string_view phase_name(const ClockPhase phase) {
    switch (phase) {
        case ClockPhase::LowEvaluate: return "LOW_EVALUATE";
        case ClockPhase::PosedgeCommit: return "POSEDGE_COMMIT";
        case ClockPhase::PostEdgeSettle: return "POST_EDGE_SETTLE";
        case ClockPhase::Negedge: return "NEGEDGE";
    }
    return "INVALID";
}

class FourPhaseClock {
public:
    [[nodiscard]] constexpr std::uint64_t phase_count() const { return phase_count_; }
    [[nodiscard]] constexpr std::uint64_t cycle() const { return phase_count_ >> 2U; }
    [[nodiscard]] constexpr ClockPhase phase() const {
        return static_cast<ClockPhase>(phase_count_ & 3U);
    }
    [[nodiscard]] constexpr bool clk_level() const {
        return phase() == ClockPhase::PosedgeCommit || phase() == ClockPhase::PostEdgeSettle;
    }
    [[nodiscard]] constexpr bool is_posedge_phase() const { return phase() == ClockPhase::PosedgeCommit; }
    [[nodiscard]] constexpr bool is_negedge_phase() const { return phase() == ClockPhase::Negedge; }

    void require(const ClockPhase expected) const {
        if (phase() != expected) throw std::logic_error("illegal four-phase clock transition");
    }

    void advance_from(const ClockPhase expected) {
        require(expected);
        ++phase_count_;
    }

private:
    std::uint64_t phase_count_{};
};

template <typename State>
class CurrentNext {
public:
    [[nodiscard]] const State& current() const { return current_; }
    [[nodiscard]] const State& next() const { return next_; }

    void prepare_next(State value, const ClockPhase phase) {
        if (phase != ClockPhase::LowEvaluate) throw std::logic_error("Next may only be prepared in Phase 0");
        next_ = std::move(value);
        next_prepared_ = true;
    }

    void commit(const ClockPhase phase) {
        if (phase != ClockPhase::PosedgeCommit || !next_prepared_)
            throw std::logic_error("Current may only commit a prepared Next in Phase 1");
        current_ = next_;
        next_prepared_ = false;
    }

private:
    State current_{};
    State next_{};
    bool next_prepared_{};
};

struct MacSignals {
    std::uint16_t mul0{};
    std::uint16_t add0{};
    std::uint16_t mul1{};
    std::uint16_t add1{};
    std::uint16_t mul2{};
    std::uint16_t result16{};
    std::uint8_t result8{};

    constexpr bool operator==(const MacSignals&) const = default;
};

constexpr std::uint16_t mul0(const std::uint8_t x) {
    return static_cast<std::uint16_t>(x) * std::uint16_t{3};
}
constexpr std::uint16_t add0(const std::uint16_t value) { return value + std::uint16_t{17}; }
constexpr std::uint16_t mul1(const std::uint16_t value) { return value * std::uint16_t{5}; }
constexpr std::uint16_t add1(const std::uint16_t value) { return value + std::uint16_t{29}; }
constexpr std::uint16_t mul2(const std::uint16_t value) { return value * std::uint16_t{7}; }
constexpr std::uint16_t add2(const std::uint16_t value) { return value + std::uint16_t{43}; }

constexpr MacSignals evaluate_mac(const std::uint8_t x) {
    MacSignals result;
    result.mul0 = mul0(x);
    result.add0 = add0(result.mul0);
    result.mul1 = mul1(result.add0);
    result.add1 = add1(result.mul1);
    result.mul2 = mul2(result.add1);
    result.result16 = add2(result.mul2);
    result.result8 = static_cast<std::uint8_t>(result.result16 & std::uint16_t{0x00FF});
    return result;
}

namespace framework {

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

struct PhaseStamp {
    std::uint64_t phase_count{};
    std::uint64_t cycle{};
    ClockPhase phase{ClockPhase::LowEvaluate};
    bool clk_level{};
};

class Clock {
public:
    Clock() = default;
    Clock(const Clock&) = delete;
    Clock& operator=(const Clock&) = delete;
    Clock(Clock&&) = delete;
    Clock& operator=(Clock&&) = delete;

    void register_participant(ClockParticipant& participant) {
        if (started_)
            throw std::logic_error("clock participant registration is closed after simulation starts");
        if (faulted_)
            throw std::logic_error("cannot register a participant on a faulted clock");
        if (std::find(participants_.begin(), participants_.end(), &participant) != participants_.end())
            throw std::logic_error("clock participant registered more than once");
        participants_.push_back(&participant);
    }

    [[nodiscard]] std::size_t participant_count() const noexcept { return participants_.size(); }
    [[nodiscard]] std::uint64_t phase_count() const noexcept { return phase_count_; }
    [[nodiscard]] std::uint64_t cycle() const noexcept { return phase_count_ >> 2U; }
    [[nodiscard]] ClockPhase phase() const noexcept {
        return static_cast<ClockPhase>(phase_count_ & 3U);
    }
    [[nodiscard]] bool clk_level() const noexcept {
        return phase() == ClockPhase::PosedgeCommit || phase() == ClockPhase::PostEdgeSettle;
    }
    [[nodiscard]] bool started() const noexcept { return started_; }
    [[nodiscard]] bool faulted() const noexcept { return faulted_; }

    PhaseStamp step(const ClockPhase expected) {
        if (faulted_) throw std::logic_error("faulted clock cannot continue");
        if (phase() != expected) throw std::logic_error("illegal clock phase request");

        started_ = true;
        const auto stamp = PhaseStamp {phase_count_, cycle(), expected, clk_level()};
        try {
            switch (expected) {
                case ClockPhase::LowEvaluate:
                    for (auto* participant : participants_) participant->low();
                    break;
                case ClockPhase::PosedgeCommit:
                    // This barrier prevents every predictable missing-Next error
                    // before the first module is allowed to commit.
                    for (const auto* participant : participants_) participant->validate_posedge();
                    for (auto* participant : participants_) participant->posedge();
                    break;
                case ClockPhase::PostEdgeSettle:
                    for (auto* participant : participants_) participant->high();
                    break;
                case ClockPhase::Negedge:
                    for (auto* participant : participants_) participant->negedge();
                    break;
            }
        } catch (...) {
            faulted_ = true;
            throw;
        }
        ++phase_count_;
        return stamp;
    }

    void tick() {
        if (phase() != ClockPhase::LowEvaluate)
            throw std::logic_error("a complete clock tick must start in LOW_EVALUATE");
        static_cast<void>(step(ClockPhase::LowEvaluate));
        static_cast<void>(step(ClockPhase::PosedgeCommit));
        static_cast<void>(step(ClockPhase::PostEdgeSettle));
        static_cast<void>(step(ClockPhase::Negedge));
    }

private:
    std::vector<ClockParticipant*> participants_{};
    std::uint64_t phase_count_{};
    bool started_{};
    bool faulted_{};
};

template <typename T>
class ClockedState {
public:
    ClockedState() = default;
    explicit ClockedState(T initial) : current_(std::move(initial)), next_(current_) {}

    [[nodiscard]] const T& current() const noexcept { return current_; }
    [[nodiscard]] const T& next() const noexcept { return next_; }
    [[nodiscard]] bool next_prepared() const noexcept { return next_prepared_; }
    [[nodiscard]] std::uint64_t commit_count() const noexcept { return commit_count_; }

    void prepare() {
        if (next_prepared_) throw std::logic_error("ClockedState Next prepared more than once");
        next_ = current_;
        next_prepared_ = true;
    }

    T& next_value() {
        if (!next_prepared_) throw std::logic_error("ClockedState Next modified before prepare");
        return next_;
    }

    void validate_commit() const {
        if (!next_prepared_) throw std::logic_error("ClockedState committed without prepared Next");
    }

    void commit() {
        validate_commit();
        // Assignment deliberately leaves next_ unchanged so Phase 2 can audit
        // the value prepared in Phase 0. If T has a throwing assignment, Clock
        // enters its permanent faulted state; framework state types should
        // therefore prefer noexcept copy assignment.
        current_ = next_;
        next_prepared_ = false;
        ++commit_count_;
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
        : state_(std::move(initial_value)), reset_value_(std::move(reset_value)), d_(state_.current()) {}

    void drive(const T& d, const bool enable = true, const bool reset = false) {
        d_ = d;
        enable_ = enable;
        reset_ = reset;
    }

    void low() override {
        state_.prepare();
        if (reset_) state_.next_value() = reset_value_;
        else if (enable_) state_.next_value() = d_;
    }

    void validate_posedge() const override { state_.validate_commit(); }
    void posedge() override { state_.commit(); }

    [[nodiscard]] const T& q() const noexcept { return state_.current(); }
    [[nodiscard]] const T& next() const noexcept { return state_.next(); }
    [[nodiscard]] bool next_prepared() const noexcept { return state_.next_prepared(); }

private:
    ClockedState<T> state_{};
    T reset_value_{};
    T d_{};
    bool enable_{true};
    bool reset_{};
};

template <typename T, std::size_t N>
class RegisterBank final : public ClockedModule {
public:
    using Value = std::array<T, N>;

    explicit RegisterBank(Value reset_value = {}, Value initial_value = {})
        : state_(std::move(initial_value)), reset_value_(std::move(reset_value)), d_(state_.current()) {}

    void drive(const Value& d, const bool enable = true, const bool reset = false) {
        d_ = d;
        enable_ = enable;
        reset_ = reset;
    }

    void low() override {
        state_.prepare();
        if (reset_) state_.next_value() = reset_value_;
        else if (enable_) state_.next_value() = d_;
    }

    void validate_posedge() const override { state_.validate_commit(); }
    void posedge() override { state_.commit(); }

    [[nodiscard]] const Value& q() const noexcept { return state_.current(); }
    [[nodiscard]] const Value& next() const noexcept { return state_.next(); }

private:
    ClockedState<Value> state_{};
    Value reset_value_{};
    Value d_{};
    bool enable_{true};
    bool reset_{};
};

template <typename T, std::size_t Depth>
class SyncFifo final : public ClockedModule {
public:
    static_assert(Depth > 0, "SyncFifo depth must be non-zero");

    struct State {
        std::array<T, Depth> memory{};
        std::array<bool, Depth> memory_valid{};
        std::size_t wr_ptr{};
        std::size_t rd_ptr{};
        std::size_t count{};
        T rd_data{};
        bool rd_data_valid{};
        constexpr bool operator==(const State&) const = default;
    };

    struct Signals {
        bool full{};
        bool empty{};
        bool wr_ready{};
        bool rd_available{};
        bool wr_fire{};
        bool rd_fire{};
        constexpr bool operator==(const Signals&) const = default;
    };

    void drive(const bool reset, const bool wr_en, const T& wr_data, const bool rd_en) {
        reset_ = reset;
        wr_en_ = wr_en;
        wr_data_ = wr_data;
        rd_en_ = rd_en;
    }

    void low() override {
        const auto& current = state_.current();
        signals_ = flags_for(current);
        signals_.wr_fire = wr_en_ && signals_.wr_ready && !reset_;
        signals_.rd_fire = rd_en_ && signals_.rd_available && !reset_;

        state_.prepare();
        state_valid_.prepare();
        state_valid_.next_value() = state_valid_.current() || reset_;
        auto& next = state_.next_value();
        next.rd_data_valid = false;

        if (reset_) {
            next.wr_ptr = 0;
            next.rd_ptr = 0;
            next.count = 0;
            next.rd_data = T{};
            next.rd_data_valid = false;
            next.memory_valid.fill(false);
            return;
        }

        if (signals_.wr_fire) {
            next.memory[current.wr_ptr] = wr_data_;
            next.memory_valid[current.wr_ptr] = true;
            next.wr_ptr = increment_pointer(current.wr_ptr);
        }
        if (signals_.rd_fire) {
            next.rd_data = current.memory[current.rd_ptr];
            next.rd_data_valid = true;
            next.rd_ptr = increment_pointer(current.rd_ptr);
        }
        if (signals_.wr_fire && !signals_.rd_fire) ++next.count;
        else if (!signals_.wr_fire && signals_.rd_fire) --next.count;
    }

    void validate_posedge() const override {
        state_.validate_commit();
        state_valid_.validate_commit();
    }

    void posedge() override {
        state_.commit();
        state_valid_.commit();
    }

    [[nodiscard]] const State& current() const noexcept { return state_.current(); }
    [[nodiscard]] const State& next() const noexcept { return state_.next(); }
    [[nodiscard]] const Signals& preedge_signals() const noexcept { return signals_; }
    [[nodiscard]] Signals current_flags() const noexcept { return flags_for(state_.current()); }
    [[nodiscard]] bool state_valid() const noexcept { return state_valid_.current(); }

private:
    static std::size_t increment_pointer(const std::size_t pointer) noexcept {
        return pointer == Depth - 1 ? 0 : pointer + 1;
    }
    static Signals flags_for(const State& state) noexcept {
        const bool full = state.count == Depth;
        const bool empty = state.count == 0;
        return {.full = full, .empty = empty, .wr_ready = !full, .rd_available = !empty};
    }

    ClockedState<State> state_{};
    ClockedState<bool> state_valid_{};
    Signals signals_{};
    T wr_data_{};
    bool reset_{};
    bool wr_en_{};
    bool rd_en_{};
};

template <typename T, std::size_t Depth>
class SyncSram final : public ClockedModule {
public:
    static_assert(Depth > 0, "SyncSram depth must be non-zero");

    enum class RequestKind : std::uint8_t { Idle = 0, Read = 1, Write = 2 };

    struct ReadOutput {
        T data{};
        bool valid{};
        bool initialized{};
        constexpr bool operator==(const ReadOutput&) const = default;
    };

    void request_idle(const bool reset = false) { set_request(RequestKind::Idle, 0, T{}, reset); }
    void request_read(const std::size_t address, const bool reset = false) {
        set_request(RequestKind::Read, address, T{}, reset);
    }
    void request_write(const std::size_t address, const T& data, const bool reset = false) {
        set_request(RequestKind::Write, address, data, reset);
    }

    void low() override {
        if (!request_set_) request_idle(false);
        if (address_ >= Depth && kind_ != RequestKind::Idle)
            throw std::out_of_range("SyncSram request address is outside the configured depth");

        read_output_.prepare();
        output_state_valid_.prepare();
        output_state_valid_.next_value() = output_state_valid_.current() || reset_;
        prepared_write_.reset();

        auto& next = read_output_.next_value();
        next.valid = false;
        next.initialized = false;
        if (reset_) {
            next = ReadOutput{};
            return;
        }

        if (kind_ == RequestKind::Read) {
            next.valid = true;
            next.initialized = initialized_[address_];
            if (next.initialized) next.data = memory_[address_];
        } else if (kind_ == RequestKind::Write) {
            prepared_write_ = WriteRequest {address_, write_data_};
        }
    }

    void validate_posedge() const override {
        read_output_.validate_commit();
        output_state_valid_.validate_commit();
    }

    void posedge() override {
        read_output_.commit();
        output_state_valid_.commit();
        if (prepared_write_) {
            memory_[prepared_write_->address] = prepared_write_->data;
            initialized_[prepared_write_->address] = true;
        }
        request_set_ = false;
        kind_ = RequestKind::Idle;
        reset_ = false;
        prepared_write_.reset();
    }

    [[nodiscard]] const ReadOutput& read_output() const noexcept { return read_output_.current(); }
    [[nodiscard]] const ReadOutput& next_read_output() const noexcept { return read_output_.next(); }
    [[nodiscard]] bool output_state_valid() const noexcept { return output_state_valid_.current(); }
    [[nodiscard]] const std::array<T, Depth>& memory() const noexcept { return memory_; }
    [[nodiscard]] const std::array<bool, Depth>& initialized() const noexcept { return initialized_; }

private:
    struct WriteRequest { std::size_t address{}; T data{}; };

    void set_request(const RequestKind kind, const std::size_t address, const T& data, const bool reset) {
        if (request_set_) throw std::logic_error("SyncSram received more than one request in a cycle");
        kind_ = kind;
        address_ = address;
        write_data_ = data;
        reset_ = reset;
        request_set_ = true;
    }

    std::array<T, Depth> memory_{};
    std::array<bool, Depth> initialized_{};
    ClockedState<ReadOutput> read_output_{};
    ClockedState<bool> output_state_valid_{};
    std::optional<WriteRequest> prepared_write_{};
    RequestKind kind_{RequestKind::Idle};
    std::size_t address_{};
    T write_data_{};
    bool reset_{};
    bool request_set_{};
};

}  // namespace framework
}  // namespace ftlpu::timing
