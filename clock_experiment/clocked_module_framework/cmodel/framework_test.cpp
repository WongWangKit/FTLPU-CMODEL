#include "test_modules.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ftlpu::timing::framework::test {
namespace {

void require(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("framework self-test failed: " + message);
}

template <typename Function>
void require_logic_error(Function&& function, const std::string& message) {
    bool caught = false;
    try { function(); } catch (const std::logic_error&) { caught = true; }
    require(caught, message);
}

class Probe final : public ClockParticipant {
public:
    explicit Probe(std::vector<int>& calls) : calls_(calls) {}
    void low() override { calls_.push_back(0); }
    void validate_posedge() const override { calls_.push_back(10); }
    void posedge() override { calls_.push_back(1); }
    void high() override { calls_.push_back(2); }
    void negedge() override { calls_.push_back(3); }
private:
    std::vector<int>& calls_;
};

class ThrowingParticipant final : public ClockParticipant {
public:
    void low() override { throw std::runtime_error("intentional fault"); }
};

void test_clock() {
    std::vector<int> calls;
    Probe probe(calls);
    Clock clock;
    clock.register_participant(probe);
    require_logic_error([&] { clock.register_participant(probe); }, "duplicate registration accepted");
    clock.tick();
    require(calls == std::vector<int>({0, 10, 1, 2, 3}), "virtual phase dispatch order");
    require(clock.phase_count() == 4 && clock.cycle() == 1 && clock.phase() == ClockPhase::LowEvaluate,
            "clock counters after tick");
    require_logic_error([&] { clock.step(ClockPhase::PosedgeCommit); }, "illegal phase accepted");
    Probe late(calls);
    require_logic_error([&] { clock.register_participant(late); }, "late registration accepted");

    ThrowingParticipant throwing;
    Clock fault_clock;
    fault_clock.register_participant(throwing);
    bool caught = false;
    try { fault_clock.step(ClockPhase::LowEvaluate); } catch (const std::runtime_error&) { caught = true; }
    require(caught && fault_clock.faulted(), "participant exception did not fault clock");
    require_logic_error([&] { fault_clock.step(ClockPhase::LowEvaluate); }, "faulted clock continued");
}

void test_clocked_state_and_registers() {
    ClockedState<int> state(7);
    require_logic_error([&] { state.commit(); }, "unprepared state committed");
    state.prepare();
    require(state.current() == 7 && state.next() == 7, "prepare did not hold by default");
    state.next_value() = 9;
    state.commit();
    require(state.current() == 9 && state.next() == 9 && state.commit_count() == 1,
            "ClockedState commit result");
    require_logic_error([&] { state.commit(); }, "duplicate commit accepted");

    Register<int> reg(0, 5);
    reg.drive(11, false, false); reg.low();
    require(reg.q() == 5 && reg.next() == 5, "register enable hold");
    reg.validate_posedge(); reg.posedge();
    reg.drive(11, true, false); reg.low(); reg.validate_posedge(); reg.posedge();
    require(reg.q() == 11, "register write");
    reg.drive(99, true, true); reg.low(); reg.validate_posedge(); reg.posedge();
    require(reg.q() == 0, "register synchronous reset");

    RegisterBank<int, 2> bank({0, 0}, {1, 2});
    bank.drive({3, 4}, true, false); bank.low(); bank.validate_posedge(); bank.posedge();
    require(bank.q() == std::array<int, 2>({3, 4}), "register bank update");
}

void advance_fifo(SyncFifo<int, 2>& fifo, bool reset, bool we, int wd, bool re) {
    fifo.drive(reset, we, wd, re); fifo.low(); fifo.validate_posedge(); fifo.posedge();
}

void test_fifo() {
    SyncFifo<int, 2> fifo;
    advance_fifo(fifo, true, false, 0, false);
    advance_fifo(fifo, false, true, 10, true); // empty: write only
    require(fifo.current().count == 1 && !fifo.current().rd_data_valid, "empty simultaneous contract");
    advance_fifo(fifo, false, true, 20, false);
    require(fifo.current().count == 2, "FIFO full setup");
    advance_fifo(fifo, false, true, 30, true); // full: read only
    require(fifo.current().count == 1 && fifo.current().rd_data_valid && fifo.current().rd_data == 10,
            "full simultaneous contract");
    const auto memory_before_reset = fifo.current().memory;
    advance_fifo(fifo, true, false, 0, false);
    require(fifo.current().count == 0 && fifo.current().memory == memory_before_reset,
            "FIFO reset changed memory data");
    require(fifo.current().memory_valid == std::array<bool, 2>({false, false}),
            "FIFO reset did not clear memory_valid");
}

void advance_sram(SyncSram<int, 4>& sram, SyncSram<int, 4>::RequestKind kind,
                  std::size_t address = 0, int data = 0, bool reset = false) {
    if (kind == SyncSram<int, 4>::RequestKind::Idle) sram.request_idle(reset);
    else if (kind == SyncSram<int, 4>::RequestKind::Read) sram.request_read(address, reset);
    else sram.request_write(address, data, reset);
    sram.low(); sram.validate_posedge(); sram.posedge();
}

void test_sram() {
    using Sram = SyncSram<int, 4>;
    Sram sram;
    advance_sram(sram, Sram::RequestKind::Read, 2);
    require(sram.read_output().valid && !sram.read_output().initialized, "uninitialized SRAM read");
    advance_sram(sram, Sram::RequestKind::Write, 2, 77);
    require(!sram.read_output().valid && sram.initialized()[2] && sram.memory()[2] == 77,
            "SRAM write");
    advance_sram(sram, Sram::RequestKind::Read, 2);
    require(sram.read_output().valid && sram.read_output().initialized && sram.read_output().data == 77,
            "synchronous SRAM read");
    advance_sram(sram, Sram::RequestKind::Idle, 0, 0, true);
    require(!sram.read_output().valid && sram.initialized()[2] && sram.memory()[2] == 77,
            "SRAM reset retention");
}

}  // namespace

void run_framework_unit_tests() {
    test_clock();
    test_clocked_state_and_registers();
    test_fifo();
    test_sram();
}

}  // namespace ftlpu::timing::framework::test

