#include "cpp_timing_modeling.hpp"

#include <array>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using ftlpu::timing::ClockPhase;
using ftlpu::timing::FourPhaseClock;
using ftlpu::timing::MacSignals;
using ftlpu::timing::evaluate_mac;
using ftlpu::timing::phase_name;

constexpr std::size_t kDepth = 4;

struct CycleControl { bool rst{}; bool consumer_ready{}; };
struct Stimulus { bool rst{}; bool producer_valid{}; std::uint8_t producer_data{}; bool consumer_ready{}; };
struct ProducerView { std::size_t transaction_index{}; bool valid{}; std::uint8_t data{}; };

class ProducerDriver {
public:
    explicit ProducerDriver(const std::vector<std::uint8_t>& transactions) : transactions_(transactions) {}
    [[nodiscard]] ProducerView drive(const bool rst) const {
        const bool has_transaction = transaction_index_ < transactions_.size();
        return {.transaction_index = transaction_index_, .valid = !rst && has_transaction,
                .data = !rst && has_transaction ? transactions_[transaction_index_] : std::uint8_t{0}};
    }
    void update_on_negedge(const bool producer_fire) {
        if (producer_fire) {
            if (transaction_index_ >= transactions_.size()) throw std::runtime_error("producer advanced beyond list");
            ++transaction_index_;
        }
    }
private:
    const std::vector<std::uint8_t>& transactions_;
    std::size_t transaction_index_{};
};

struct FifoInputs { bool rst{}; bool wr_en{}; std::uint8_t wr_data{}; bool rd_en{}; };
struct FifoState {
    std::array<std::uint8_t, kDepth> memory{};
    std::array<bool, kDepth> memory_valid{};
    std::size_t wr_ptr{};
    std::size_t rd_ptr{};
    std::size_t count{};
    std::uint8_t rd_data{};
    bool rd_data_valid{};
    constexpr bool operator==(const FifoState&) const = default;
};
struct FifoSignals { bool full{}; bool empty{}; bool wr_ready{}; bool rd_available{}; bool wr_fire{}; bool rd_fire{}; };

class SyncFifo {
public:
    void evaluate(const FifoInputs& input, const ClockPhase phase) {
        if (phase != ClockPhase::LowEvaluate) throw std::logic_error("FIFO evaluate outside Phase 0");
        signals_.full = current_.count == kDepth;
        signals_.empty = current_.count == 0;
        signals_.wr_ready = !signals_.full;
        signals_.rd_available = !signals_.empty;
        signals_.wr_fire = input.wr_en && signals_.wr_ready && !input.rst;
        signals_.rd_fire = input.rd_en && signals_.rd_available && !input.rst;
        next_ = current_;
        next_.rd_data_valid = false;
        next_state_valid_ = current_state_valid_ || input.rst;
        next_prepared_ = true;
        if (input.rst) {
            next_.wr_ptr = 0; next_.rd_ptr = 0; next_.count = 0; next_.rd_data = 0;
            next_.rd_data_valid = false; next_.memory_valid.fill(false); return;
        }
        if (signals_.wr_fire) {
            next_.memory[current_.wr_ptr] = input.wr_data;
            next_.memory_valid[current_.wr_ptr] = true;
            next_.wr_ptr = increment_pointer(current_.wr_ptr);
        }
        if (signals_.rd_fire) {
            next_.rd_data = current_.memory[current_.rd_ptr];
            next_.rd_data_valid = true;
            next_.rd_ptr = increment_pointer(current_.rd_ptr);
        }
        if (signals_.wr_fire && !signals_.rd_fire) ++next_.count;
        else if (!signals_.wr_fire && signals_.rd_fire) --next_.count;
    }

    void commit(const ClockPhase phase) {
        if (phase != ClockPhase::PosedgeCommit || !next_prepared_)
            throw std::logic_error("FIFO commit outside Phase 1 or without prepared Next");
        current_ = next_;
        current_state_valid_ = next_state_valid_;
        next_prepared_ = false;
    }

    [[nodiscard]] const FifoState& current() const { return current_; }
    [[nodiscard]] const FifoState& next() const { return next_; }
    [[nodiscard]] const FifoSignals& signals() const { return signals_; }
    [[nodiscard]] bool state_valid() const { return current_state_valid_; }
    [[nodiscard]] FifoSignals current_flags() const {
        FifoSignals result = signals_;
        result.full = current_.count == kDepth; result.empty = current_.count == 0;
        result.wr_ready = !result.full; result.rd_available = !result.empty;
        return result;
    }
private:
    static std::size_t increment_pointer(const std::size_t pointer) { return pointer == kDepth - 1 ? 0 : pointer + 1; }
    FifoState current_{}; FifoState next_{}; FifoSignals signals_{};
    bool current_state_valid_{}; bool next_state_valid_{}; bool next_prepared_{};
};

struct MacObservation { bool response_valid{}; std::uint8_t input{}; MacSignals nodes{}; };
MacObservation observe_mac(const FifoState& a) {
    return {.response_valid = a.rd_data_valid, .input = a.rd_data, .nodes = evaluate_mac(a.rd_data)};
}

struct TransferSignals {
    bool a_wr_en{}; std::uint8_t a_wr_data{}; bool a_rd_en{}; bool b_wr_en{};
    std::uint8_t b_wr_data{}; bool b_rd_en{}; bool producer_ready{}; bool transfer_invariant_ok{};
};

class TransferController {
public:
    [[nodiscard]] TransferSignals evaluate(const FifoState& a, const FifoState& b,
                                           const MacObservation& mac, const Stimulus& input) const {
        const bool a_full = a.count == kDepth, a_empty = a.count == 0, b_full = b.count == kDepth;
        TransferSignals result;
        result.a_wr_en = input.producer_valid; result.a_wr_data = input.producer_data;
        result.a_rd_en = !input.rst && !a_empty && !a.rd_data_valid && !b_full;
        result.b_wr_en = !input.rst && a.rd_data_valid;
        result.b_wr_data = mac.nodes.result8;
        result.b_rd_en = !input.rst && input.consumer_ready;
        result.producer_ready = !input.rst && !a_full;
        result.transfer_invariant_ok = !(a.rd_data_valid && b_full);
        return result;
    }
};

struct Schedule { bool evaluate_a_first{}; bool commit_a_first{}; };
Schedule parse_schedule(const std::string_view name) {
    if (name == "eval_ab_commit_ab") return {true, true};
    if (name == "eval_ab_commit_ba") return {true, false};
    if (name == "eval_ba_commit_ab") return {false, true};
    if (name == "eval_ba_commit_ba") return {false, false};
    throw std::runtime_error("unknown schedule: " + std::string(name));
}

std::vector<unsigned> load_hex_values(const std::string& path, const unsigned maximum, const char* description) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open " + std::string(description) + ": " + path);
    std::vector<unsigned> result;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(stream, line)) {
        ++line_number;
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        std::istringstream words(line);
        std::string token;
        if (!(words >> token)) continue;
        std::size_t consumed = 0;
        const auto raw = std::stoul(token, &consumed, 16);
        if (consumed != token.size() || raw > maximum)
            throw std::runtime_error("invalid " + std::string(description) + " at line " + std::to_string(line_number));
        result.push_back(raw);
    }
    if (result.empty()) throw std::runtime_error(std::string(description) + " contains no entries");
    return result;
}

std::vector<std::uint8_t> load_transactions(const std::string& path) {
    const auto raw = load_hex_values(path, 0xFFU, "producer transaction list");
    std::vector<std::uint8_t> result;
    for (const auto item : raw) result.push_back(static_cast<std::uint8_t>(item));
    return result;
}
std::vector<CycleControl> load_controls(const std::string& path) {
    const auto raw = load_hex_values(path, 0x3U, "consumer control list");
    std::vector<CycleControl> result;
    for (const auto item : raw)
        result.push_back({.rst = ((item >> 1U) & 1U) != 0, .consumer_ready = (item & 1U) != 0});
    return result;
}

std::string number_or_x(const bool valid, const std::size_t value) { return valid ? std::to_string(value) : "x"; }
std::string byte_or_x(const bool valid, const std::uint8_t value) { return valid ? std::to_string(static_cast<unsigned>(value)) : "x"; }
std::string word_or_x(const bool valid, const std::uint16_t value) { return valid ? std::to_string(value) : "x"; }
std::string bool_or_x(const bool valid, const bool value) { return valid ? (value ? "1" : "0") : "x"; }

void write_memory(std::ofstream& trace, const FifoState& state, const bool valid) {
    for (std::size_t slot = 0; slot < kDepth; ++slot) {
        trace << ',' << bool_or_x(valid, state.memory_valid[slot])
              << ',' << byte_or_x(valid && state.memory_valid[slot], state.memory[slot]);
    }
}

void write_fifo(std::ofstream& trace, const SyncFifo& fifo, const bool phase0) {
    const auto& state = fifo.current();
    const auto& next = fifo.next();
    const auto sig = phase0 ? fifo.signals() : fifo.current_flags();
    const bool valid = fifo.state_valid();
    trace << ',' << bool_or_x(valid, sig.full) << ',' << bool_or_x(valid, sig.empty)
          << ',' << bool_or_x(valid, sig.wr_ready) << ',' << bool_or_x(valid, sig.rd_available)
          << ',' << static_cast<unsigned>(fifo.signals().wr_fire)
          << ',' << static_cast<unsigned>(fifo.signals().rd_fire)
          << ',' << static_cast<unsigned>(valid)
          << ',' << number_or_x(valid, state.wr_ptr) << ',' << number_or_x(valid, state.rd_ptr)
          << ',' << number_or_x(valid, state.count) << ',' << byte_or_x(valid, state.rd_data)
          << ',' << bool_or_x(valid, state.rd_data_valid);
    write_memory(trace, state, valid);
    trace << ',' << next.wr_ptr << ',' << next.rd_ptr << ',' << next.count
          << ',' << static_cast<unsigned>(next.rd_data) << ',' << static_cast<unsigned>(next.rd_data_valid);
}

void write_common(std::ofstream& trace, const FourPhaseClock& clock, const Stimulus& input,
                  const ProducerView& producer, const TransferSignals& transfer,
                  const bool producer_fire, const bool producer_stall,
                  const std::size_t accepted_count, const std::size_t transaction_count,
                  const MacObservation& mac, const bool mac_known,
                  const std::string& pre_result, const std::string& post_result) {
    trace << clock.phase_count() << ',' << clock.phase_count() << ',' << clock.cycle() << ','
          << static_cast<unsigned>(clock.phase()) << ',' << phase_name(clock.phase()) << ','
          << static_cast<unsigned>(clock.clk_level()) << ',' << static_cast<unsigned>(input.rst)
          << ',' << producer.transaction_index << ',' << static_cast<unsigned>(producer.valid)
          << ',' << static_cast<unsigned>(producer.data) << ',' << static_cast<unsigned>(transfer.producer_ready)
          << ',' << static_cast<unsigned>(producer_fire) << ',' << static_cast<unsigned>(producer_stall)
          << ',' << transaction_count - accepted_count << ',' << accepted_count
          << ',' << static_cast<unsigned>(input.consumer_ready) << ',' << static_cast<unsigned>(transfer.a_rd_en)
          << ',' << static_cast<unsigned>(transfer.b_wr_en) << ',' << static_cast<unsigned>(transfer.b_wr_data)
          << ',' << static_cast<unsigned>(transfer.b_rd_en) << ',' << static_cast<unsigned>(transfer.transfer_invariant_ok)
          << ',' << bool_or_x(mac_known, mac.response_valid)
          << ',' << byte_or_x(mac_known, mac.input)
          << ',' << word_or_x(mac_known, mac.nodes.mul0) << ',' << word_or_x(mac_known, mac.nodes.add0)
          << ',' << word_or_x(mac_known, mac.nodes.mul1) << ',' << word_or_x(mac_known, mac.nodes.add1)
          << ',' << word_or_x(mac_known, mac.nodes.mul2) << ',' << word_or_x(mac_known, mac.nodes.result16)
          << ',' << byte_or_x(mac_known, mac.nodes.result8) << ',' << pre_result << ',' << post_result;
}

void write_header(std::ofstream& trace) {
    trace << "phase_count,half_clk,cycle,phase,phase_name,clk_level,rst,producer_transaction_index,"
             "producer_valid,producer_data,producer_ready,producer_fire,producer_stall,producer_pending_count,"
             "producer_accepted_count,consumer_ready,tc_a_rd_en,tc_b_wr_en,tc_b_wr_data,tc_b_rd_en,"
             "transfer_invariant_ok,mac_valid,mac_input,mac_mul0,mac_add0,mac_mul1,mac_add1,mac_mul2,"
             "mac_result16,mac_result8,mac_pre_edge_result,mac_post_edge_result";
    for (const char* prefix : {"a_", "b_"}) {
        trace << ',' << prefix << "full" << ',' << prefix << "empty" << ',' << prefix << "wr_ready"
              << ',' << prefix << "rd_available" << ',' << prefix << "wr_fire" << ',' << prefix << "rd_fire"
              << ',' << prefix << "state_valid" << ',' << prefix << "wr_ptr_q" << ',' << prefix << "rd_ptr_q"
              << ',' << prefix << "count_q" << ',' << prefix << "rd_data_q" << ',' << prefix << "rd_data_valid_q";
        for (std::size_t slot = 0; slot < kDepth; ++slot)
            trace << ',' << prefix << "mem" << slot << "_valid" << ',' << prefix << "mem" << slot;
        trace << ',' << prefix << "wr_ptr_d" << ',' << prefix << "rd_ptr_d" << ',' << prefix << "count_d"
              << ',' << prefix << "rd_data_d" << ',' << prefix << "rd_data_valid_d";
    }
    trace << '\n';
}

void write_mac_exhaustive(const std::string& path) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open MAC exhaustive trace: " + path);
    out << "input,mul0,add0,mul1,add1,mul2,result16,result8\n";
    for (unsigned input = 0; input < 256; ++input) {
        const auto m = evaluate_mac(static_cast<std::uint8_t>(input));
        out << input << ',' << m.mul0 << ',' << m.add0 << ',' << m.mul1 << ',' << m.add1 << ','
            << m.mul2 << ',' << m.result16 << ',' << static_cast<unsigned>(m.result8) << '\n';
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 6) {
            std::cerr << "usage: stage4_cmodel <transactions.hex> <control.hex> <trace.csv> <mac.csv> <schedule>\n";
            return 2;
        }
        const auto transactions = load_transactions(argv[1]);
        const auto controls = load_controls(argv[2]);
        const auto schedule = parse_schedule(argv[5]);
        write_mac_exhaustive(argv[4]);
        std::ofstream trace(argv[3], std::ios::trunc);
        if (!trace) throw std::runtime_error("cannot open trace file");
        write_header(trace);

        std::vector<std::uint8_t> expected_outputs;
        for (const auto item : transactions) expected_outputs.push_back(evaluate_mac(item).result8);
        SyncFifo fifo_a, fifo_b;
        TransferController controller;
        ProducerDriver producer_driver(transactions);
        FourPhaseClock clock;
        std::size_t accepted_count = 0, consumer_count = 0;
        std::size_t protocol_failures = 0, invariant_failures = 0, transfer_failures = 0;
        std::size_t latency_failures = 0, consumer_failures = 0, next_mutation_failures = 0;
        std::size_t phase_failures = 0, mac_postedge_failures = 0;
        std::size_t valid_cycles = 0, ready_cycles = 0, fire_cycles = 0, stall_cycles = 0;
        std::size_t consecutive_stalls = 0, max_stalls = 0;
        std::deque<std::pair<std::size_t, std::uint8_t>> pending_transfers;
        bool have_previous = false, previous_stall = false, previous_fire = false;
        ProducerView previous_producer{};

        for (std::size_t cycle = 0; cycle < controls.size(); ++cycle) {
            try { clock.require(ClockPhase::LowEvaluate); } catch (...) { ++phase_failures; }
            const auto& control = controls[cycle];
            const auto producer = producer_driver.drive(control.rst);
            const Stimulus input{.rst = control.rst, .producer_valid = producer.valid,
                                 .producer_data = producer.data, .consumer_ready = control.consumer_ready};
            if (have_previous) {
                const auto expected_index = previous_producer.transaction_index + static_cast<std::size_t>(previous_fire);
                if (producer.transaction_index != expected_index) ++protocol_failures;
                if (previous_stall && !control.rst &&
                    (!producer.valid || producer.data != previous_producer.data ||
                     producer.transaction_index != previous_producer.transaction_index)) ++protocol_failures;
            }
            if (producer.transaction_index != accepted_count) ++protocol_failures;

            const auto mac_pre = observe_mac(fifo_a.current());
            const bool mac_pre_known = fifo_a.state_valid();
            const auto transfer = controller.evaluate(fifo_a.current(), fifo_b.current(), mac_pre, input);
            const bool producer_fire = producer.valid && transfer.producer_ready && !input.rst;
            const bool producer_stall = producer.valid && !transfer.producer_ready && !input.rst;
            const FifoInputs a_input{.rst = input.rst, .wr_en = transfer.a_wr_en,
                                     .wr_data = transfer.a_wr_data, .rd_en = transfer.a_rd_en};
            const FifoInputs b_input{.rst = input.rst, .wr_en = transfer.b_wr_en,
                                     .wr_data = transfer.b_wr_data, .rd_en = transfer.b_rd_en};
            if (schedule.evaluate_a_first) {
                fifo_a.evaluate(a_input, clock.phase()); fifo_b.evaluate(b_input, clock.phase());
            } else {
                fifo_b.evaluate(b_input, clock.phase()); fifo_a.evaluate(a_input, clock.phase());
            }
            const auto prepared_a_next = fifo_a.next();
            const auto prepared_b_next = fifo_b.next();
            if (producer.valid) ++valid_cycles;
            if (!input.rst && transfer.producer_ready) ++ready_cycles;
            if (producer_fire) ++fire_cycles;
            if (producer_stall) { ++stall_cycles; ++consecutive_stalls; if (consecutive_stalls > max_stalls) max_stalls = consecutive_stalls; }
            else consecutive_stalls = 0;
            if (producer_fire != fifo_a.signals().wr_fire) ++protocol_failures;
            if (!transfer.transfer_invariant_ok) ++invariant_failures;
            if (producer_fire && (accepted_count >= transactions.size() || producer.data != transactions[accepted_count])) ++protocol_failures;

            if (fifo_b.signals().wr_fire) {
                if (pending_transfers.empty()) ++transfer_failures;
                else {
                    const auto [read_cycle, expected] = pending_transfers.front(); pending_transfers.pop_front();
                    if (transfer.b_wr_data != expected) ++transfer_failures;
                    if (cycle != read_cycle + 1U) ++latency_failures;
                }
            }
            if (fifo_a.signals().rd_fire) {
                const auto& a = fifo_a.current();
                if (!a.memory_valid[a.rd_ptr]) ++transfer_failures;
                else pending_transfers.emplace_back(cycle, evaluate_mac(a.memory[a.rd_ptr]).result8);
            }

            const auto pre_result_text = byte_or_x(mac_pre_known, mac_pre.nodes.result8);
            write_common(trace, clock, input, producer, transfer, producer_fire, producer_stall,
                         accepted_count, transactions.size(), mac_pre, mac_pre_known,
                         pre_result_text, pre_result_text);
            write_fifo(trace, fifo_a, true); write_fifo(trace, fifo_b, true); trace << '\n';

            clock.advance_from(ClockPhase::LowEvaluate);
            if (schedule.commit_a_first) { fifo_a.commit(clock.phase()); fifo_b.commit(clock.phase()); }
            else { fifo_b.commit(clock.phase()); fifo_a.commit(clock.phase()); }
            const auto accepted_after = accepted_count + static_cast<std::size_t>(producer_fire);
            if (fifo_b.signals().rd_fire) {
                if (!fifo_b.current().rd_data_valid || consumer_count >= expected_outputs.size() ||
                    fifo_b.current().rd_data != expected_outputs[consumer_count]) ++consumer_failures;
                ++consumer_count;
            }
            write_common(trace, clock, input, producer, transfer, producer_fire, producer_stall,
                         accepted_after, transactions.size(), mac_pre, mac_pre_known,
                         pre_result_text, pre_result_text);
            write_fifo(trace, fifo_a, false); write_fifo(trace, fifo_b, false); trace << '\n';

            clock.advance_from(ClockPhase::PosedgeCommit);
            const auto mac_post = observe_mac(fifo_a.current());
            const bool mac_post_known = fifo_a.state_valid();
            if (fifo_a.next() != prepared_a_next || fifo_b.next() != prepared_b_next) ++next_mutation_failures;
            if (mac_post_known && mac_post.response_valid && mac_post.nodes != evaluate_mac(fifo_a.current().rd_data))
                ++mac_postedge_failures;
            const auto post_result_text = byte_or_x(mac_post_known, mac_post.nodes.result8);
            write_common(trace, clock, input, producer, transfer, producer_fire, producer_stall,
                         accepted_after, transactions.size(), mac_post, mac_post_known,
                         pre_result_text, post_result_text);
            write_fifo(trace, fifo_a, false); write_fifo(trace, fifo_b, false); trace << '\n';

            const auto a_after_commit = fifo_a.current();
            const auto b_after_commit = fifo_b.current();
            clock.advance_from(ClockPhase::PostEdgeSettle);
            if (fifo_a.current() != a_after_commit || fifo_b.current() != b_after_commit) ++phase_failures;
            if (fifo_a.next() != prepared_a_next || fifo_b.next() != prepared_b_next) ++next_mutation_failures;
            write_common(trace, clock, input, producer, transfer, producer_fire, producer_stall,
                         accepted_after, transactions.size(), mac_post, mac_post_known,
                         pre_result_text, post_result_text);
            write_fifo(trace, fifo_a, false); write_fifo(trace, fifo_b, false); trace << '\n';

            producer_driver.update_on_negedge(producer_fire);
            accepted_count = accepted_after;
            have_previous = true; previous_stall = producer_stall; previous_fire = producer_fire;
            previous_producer = producer;
            clock.advance_from(ClockPhase::Negedge);
        }

        const auto pending_count = transactions.size() - accepted_count;
        const bool drained = pending_count == 0 && fifo_a.current().count == 0 && fifo_b.current().count == 0 &&
                             !fifo_a.current().rd_data_valid && !fifo_b.current().rd_data_valid && pending_transfers.empty();
        const bool scoreboard = accepted_count == transactions.size() && consumer_count == transactions.size() &&
                                transfer_failures == 0 && latency_failures == 0 && consumer_failures == 0;
        std::cout << "schedule=" << argv[5] << '\n' << "cycles=" << controls.size() << '\n'
                  << "phase_samples=" << clock.phase_count() << '\n' << "producer_transaction_count=" << transactions.size() << '\n'
                  << "producer_valid_cycles=" << valid_cycles << '\n' << "producer_ready_cycles=" << ready_cycles << '\n'
                  << "producer_fire_cycles=" << fire_cycles << '\n' << "producer_stall_cycles=" << stall_cycles << '\n'
                  << "max_consecutive_producer_stall_cycles=" << max_stalls << '\n'
                  << "producer_accepted_count=" << accepted_count << '\n' << "producer_pending_count=" << pending_count << '\n'
                  << "producer_protocol_failures=" << protocol_failures << '\n'
                  << "transfer_invariant_failures=" << invariant_failures << '\n'
                  << "transfer_response_failures=" << transfer_failures << '\n'
                  << "transfer_latency_failures=" << latency_failures << '\n'
                  << "consumer_received_count=" << consumer_count << '\n'
                  << "consumer_order_failures=" << consumer_failures << '\n'
                  << "next_mutation_failures=" << next_mutation_failures << '\n'
                  << "phase_failures=" << phase_failures << '\n'
                  << "mac_postedge_failures=" << mac_postedge_failures << '\n'
                  << "pending_transfer_responses=" << pending_transfers.size() << '\n'
                  << "final_a_count=" << fifo_a.current().count << '\n' << "final_b_count=" << fifo_b.current().count << '\n'
                  << "drained=" << (drained ? "PASS" : "FAIL") << '\n'
                  << "scoreboard=" << (scoreboard ? "PASS" : "FAIL") << '\n';
        return protocol_failures == 0 && invariant_failures == 0 && next_mutation_failures == 0 &&
                       phase_failures == 0 && mac_postedge_failures == 0 && drained && scoreboard ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "stage4_cmodel: " << error.what() << '\n';
        return 1;
    }
}



template<T> state{
    T value;
};
class clock{
    private:
        size_t clk = 0;
    public:
        void tick();
        void low();
        void posedge();
        void high();
        void negedge();
};

void clock::tick(){
    low();
    posedge();
    high();
    negedge();
    clk++;
}

//interface might be reloaded
void clock::low(state current_state, state next_state){
    next_state = ...current_state..;
}
void clock::posedge(){
    current_state = next_state;
}
void clock::high(){clk++;}
void clock::negedge(){}


class clock_specific extends clock{
    //reload tick, low, posedge, high, negedge from clock
};


