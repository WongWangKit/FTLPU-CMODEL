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
    void update_on_falling_edge(const bool producer_fire) {
        if (producer_fire) {
            if (transaction_index_ >= transactions_.size()) throw std::runtime_error("producer advanced beyond transaction list");
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
};
struct FifoSignals { bool full{}; bool empty{}; bool wr_ready{}; bool rd_available{}; bool wr_fire{}; bool rd_fire{}; };

class SyncFifo {
public:
    void evaluate(const FifoInputs& input) {
        signals_.full = current_.count == kDepth;
        signals_.empty = current_.count == 0;
        signals_.wr_ready = !signals_.full;
        signals_.rd_available = !signals_.empty;
        signals_.wr_fire = input.wr_en && signals_.wr_ready && !input.rst;
        signals_.rd_fire = input.rd_en && signals_.rd_available && !input.rst;
        next_ = current_;
        next_.rd_data_valid = false;
        next_state_valid_ = current_state_valid_ || input.rst;
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
    void commit() { current_ = next_; current_state_valid_ = next_state_valid_; }
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
    bool current_state_valid_{false}; bool next_state_valid_{false};
};

struct TransferSignals {
    bool a_wr_en{}; std::uint8_t a_wr_data{}; bool a_rd_en{}; bool b_wr_en{};
    std::uint8_t b_wr_data{}; bool b_rd_en{}; bool producer_ready{}; bool transfer_invariant_ok{};
};

class TransferController {
public:
    [[nodiscard]] TransferSignals evaluate(const FifoState& a, const FifoState& b, const Stimulus& input) const {
        const bool a_full = a.count == kDepth;
        const bool a_empty = a.count == 0;
        const bool b_full = b.count == kDepth;
        TransferSignals result;
        result.a_wr_en = input.producer_valid; result.a_wr_data = input.producer_data;
        result.a_rd_en = !input.rst && !a_empty && !a.rd_data_valid && !b_full;
        result.b_wr_en = !input.rst && a.rd_data_valid; result.b_wr_data = a.rd_data;
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
    result.reserve(raw.size());
    for (const auto value : raw) result.push_back(static_cast<std::uint8_t>(value));
    return result;
}

std::vector<CycleControl> load_controls(const std::string& path) {
    const auto raw = load_hex_values(path, 0x3U, "consumer control list");
    std::vector<CycleControl> result;
    result.reserve(raw.size());
    for (const auto value : raw)
        result.push_back({.rst = ((value >> 1U) & 1U) != 0, .consumer_ready = (value & 1U) != 0});
    return result;
}

std::string number_or_x(const bool valid, const std::size_t value) { return valid ? std::to_string(value) : "x"; }
std::string byte_or_x(const bool valid, const std::uint8_t value) { return valid ? std::to_string(static_cast<unsigned>(value)) : "x"; }
std::string bool_or_x(const bool valid, const bool value) { return valid ? (value ? "1" : "0") : "x"; }

void write_memory(std::ofstream& trace, const FifoState& state, const bool state_valid) {
    for (std::size_t index = 0; index < kDepth; ++index) {
        const bool data_valid = state_valid && state.memory_valid[index];
        trace << ',' << bool_or_x(state_valid, state.memory_valid[index]) << ',' << byte_or_x(data_valid, state.memory[index]);
    }
}

void write_fifo_low(std::ofstream& trace, const SyncFifo& fifo) {
    const auto& state = fifo.current(); const auto& next = fifo.next(); const auto& sig = fifo.signals();
    const bool valid = fifo.state_valid();
    trace << ',' << bool_or_x(valid, sig.full) << ',' << bool_or_x(valid, sig.empty)
          << ',' << bool_or_x(valid, sig.wr_ready) << ',' << bool_or_x(valid, sig.rd_available)
          << ',' << static_cast<unsigned>(sig.wr_fire) << ',' << static_cast<unsigned>(sig.rd_fire)
          << ',' << static_cast<unsigned>(valid) << ',' << number_or_x(valid, state.wr_ptr)
          << ',' << number_or_x(valid, state.rd_ptr) << ',' << number_or_x(valid, state.count)
          << ',' << byte_or_x(valid, state.rd_data) << ',' << bool_or_x(valid, state.rd_data_valid);
    write_memory(trace, state, valid);
    trace << ',' << next.wr_ptr << ',' << next.rd_ptr << ',' << next.count
          << ',' << static_cast<unsigned>(next.rd_data) << ',' << static_cast<unsigned>(next.rd_data_valid);
}

void write_fifo_high(std::ofstream& trace, const SyncFifo& fifo) {
    const auto& state = fifo.current(); const auto sig = fifo.current_flags(); const bool valid = fifo.state_valid();
    trace << ',' << bool_or_x(valid, sig.full) << ',' << bool_or_x(valid, sig.empty)
          << ',' << bool_or_x(valid, sig.wr_ready) << ',' << bool_or_x(valid, sig.rd_available)
          << ',' << static_cast<unsigned>(sig.wr_fire) << ',' << static_cast<unsigned>(sig.rd_fire)
          << ',' << static_cast<unsigned>(valid) << ',' << number_or_x(valid, state.wr_ptr)
          << ',' << number_or_x(valid, state.rd_ptr) << ',' << number_or_x(valid, state.count)
          << ',' << byte_or_x(valid, state.rd_data) << ',' << bool_or_x(valid, state.rd_data_valid);
    write_memory(trace, state, valid);
    trace << ",,,,,";
}

void write_common(std::ofstream& trace, const std::uint64_t half_clk, const char* phase,
                  const Stimulus& input, const ProducerView& producer, const TransferSignals& transfer,
                  const bool producer_fire, const bool producer_stall,
                  const std::size_t accepted_count, const std::size_t transaction_count) {
    trace << half_clk << ',' << (half_clk >> 1U) << ',' << phase << ',' << static_cast<unsigned>(half_clk & 1U)
          << ',' << static_cast<unsigned>(input.rst) << ',' << producer.transaction_index
          << ',' << static_cast<unsigned>(producer.valid) << ',' << static_cast<unsigned>(producer.data)
          << ',' << static_cast<unsigned>(transfer.producer_ready) << ',' << static_cast<unsigned>(producer_fire)
          << ',' << static_cast<unsigned>(producer_stall) << ',' << transaction_count - accepted_count
          << ',' << accepted_count << ',' << static_cast<unsigned>(input.consumer_ready)
          << ',' << static_cast<unsigned>(transfer.a_rd_en) << ',' << static_cast<unsigned>(transfer.b_wr_en)
          << ',' << static_cast<unsigned>(transfer.b_wr_data) << ',' << static_cast<unsigned>(transfer.b_rd_en)
          << ',' << static_cast<unsigned>(transfer.transfer_invariant_ok);
}

void write_header(std::ofstream& trace) {
    trace << "half_clk,cycle,phase,clk_level,rst,producer_transaction_index,producer_valid,producer_data,"
             "producer_ready,producer_fire,producer_stall,producer_pending_count,producer_accepted_count,"
             "consumer_ready,tc_a_rd_en,tc_b_wr_en,tc_b_wr_data,tc_b_rd_en,transfer_invariant_ok";
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
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            std::cerr << "usage: two_fifo_cmodel <producer_transactions.hex> <consumer_control.hex> <trace.csv> <schedule>\n";
            return 2;
        }
        const auto transactions = load_transactions(argv[1]);
        const auto controls = load_controls(argv[2]);
        const auto schedule = parse_schedule(argv[4]);
        std::ofstream trace(argv[3], std::ios::trunc);
        if (!trace) throw std::runtime_error("cannot open trace file: " + std::string(argv[3]));
        write_header(trace);

        SyncFifo fifo_a; SyncFifo fifo_b; TransferController controller; ProducerDriver producer_driver(transactions);
        std::uint64_t half_clk = 0;
        std::size_t accepted_count = 0, consumer_received_count = 0;
        std::size_t producer_protocol_violations = 0, transfer_invariant_failures = 0;
        std::size_t producer_accept_order_failures = 0, transfer_response_failures = 0;
        std::size_t transfer_latency_failures = 0, consumer_order_failures = 0;
        std::size_t producer_valid_cycles = 0, producer_ready_cycles = 0, producer_fire_cycles = 0;
        std::size_t producer_stall_cycles = 0, consecutive_stalls = 0, max_consecutive_stalls = 0;
        std::deque<std::pair<std::size_t, std::uint8_t>> pending_transfers;
        bool have_previous_cycle = false, previous_stall = false, previous_fire = false;
        ProducerView previous_producer{};

        for (std::size_t cycle = 0; cycle < controls.size(); ++cycle) {
            const auto& control = controls[cycle];
            const auto producer = producer_driver.drive(control.rst);
            const Stimulus stimulus{.rst = control.rst, .producer_valid = producer.valid,
                                    .producer_data = producer.data, .consumer_ready = control.consumer_ready};
            if (have_previous_cycle) {
                const auto expected_index = previous_producer.transaction_index + static_cast<std::size_t>(previous_fire);
                if (producer.transaction_index != expected_index) ++producer_protocol_violations;
                if (previous_stall && !control.rst &&
                    (!producer.valid || producer.data != previous_producer.data ||
                     producer.transaction_index != previous_producer.transaction_index)) ++producer_protocol_violations;
            }
            if (producer.transaction_index != accepted_count) ++producer_protocol_violations;

            const auto transfer = controller.evaluate(fifo_a.current(), fifo_b.current(), stimulus);
            const bool producer_fire = producer.valid && transfer.producer_ready && !stimulus.rst;
            const bool producer_stall = producer.valid && !transfer.producer_ready && !stimulus.rst;
            const FifoInputs a_input{.rst = stimulus.rst, .wr_en = transfer.a_wr_en,
                                     .wr_data = transfer.a_wr_data, .rd_en = transfer.a_rd_en};
            const FifoInputs b_input{.rst = stimulus.rst, .wr_en = transfer.b_wr_en,
                                     .wr_data = transfer.b_wr_data, .rd_en = transfer.b_rd_en};
            if (schedule.evaluate_a_first) { fifo_a.evaluate(a_input); fifo_b.evaluate(b_input); }
            else { fifo_b.evaluate(b_input); fifo_a.evaluate(a_input); }

            if (producer.valid) ++producer_valid_cycles;
            if (!stimulus.rst && transfer.producer_ready) ++producer_ready_cycles;
            if (producer_fire) ++producer_fire_cycles;
            if (producer_stall) {
                ++producer_stall_cycles; ++consecutive_stalls;
                if (consecutive_stalls > max_consecutive_stalls) max_consecutive_stalls = consecutive_stalls;
            } else consecutive_stalls = 0;

            if (!transfer.transfer_invariant_ok) ++transfer_invariant_failures;
            if (producer_fire != fifo_a.signals().wr_fire || (stimulus.rst && producer_fire)) ++producer_protocol_violations;
            if (producer_fire && (accepted_count >= transactions.size() || producer.data != transactions[accepted_count]))
                ++producer_accept_order_failures;

            if (fifo_b.signals().wr_fire) {
                if (pending_transfers.empty()) ++transfer_response_failures;
                else {
                    const auto [read_cycle, expected_data] = pending_transfers.front(); pending_transfers.pop_front();
                    if (transfer.b_wr_data != expected_data) ++transfer_response_failures;
                    if (cycle != read_cycle + 1U) ++transfer_latency_failures;
                }
            }
            if (fifo_a.signals().rd_fire) {
                const auto& a = fifo_a.current();
                if (!a.memory_valid[a.rd_ptr]) ++transfer_response_failures;
                else pending_transfers.emplace_back(cycle, a.memory[a.rd_ptr]);
            }

            const auto accepted_after_edge = accepted_count + static_cast<std::size_t>(producer_fire);
            write_common(trace, half_clk, "LOW", stimulus, producer, transfer, producer_fire, producer_stall,
                         accepted_count, transactions.size());
            write_fifo_low(trace, fifo_a); write_fifo_low(trace, fifo_b); trace << '\n';

            ++half_clk;
            if (schedule.commit_a_first) { fifo_a.commit(); fifo_b.commit(); }
            else { fifo_b.commit(); fifo_a.commit(); }
            if (fifo_b.signals().rd_fire) {
                if (!fifo_b.current().rd_data_valid || consumer_received_count >= transactions.size() ||
                    fifo_b.current().rd_data != transactions[consumer_received_count]) ++consumer_order_failures;
                ++consumer_received_count;
            }
            write_common(trace, half_clk, "HIGH", stimulus, producer, transfer, producer_fire, producer_stall,
                         accepted_after_edge, transactions.size());
            write_fifo_high(trace, fifo_a); write_fifo_high(trace, fifo_b); trace << '\n';
            ++half_clk;

            producer_driver.update_on_falling_edge(producer_fire);
            accepted_count = accepted_after_edge;
            have_previous_cycle = true; previous_stall = producer_stall; previous_fire = producer_fire;
            previous_producer = producer;
        }

        const std::size_t producer_pending_count = transactions.size() - accepted_count;
        const bool drained = producer_pending_count == 0 && fifo_a.current().count == 0 && fifo_b.current().count == 0 &&
                             !fifo_a.current().rd_data_valid && pending_transfers.empty();
        const bool scoreboard_ok = accepted_count == transactions.size() && consumer_received_count == transactions.size() &&
                                   producer_accept_order_failures == 0 && transfer_response_failures == 0 &&
                                   transfer_latency_failures == 0 && consumer_order_failures == 0;
        std::cout << "schedule=" << argv[4] << '\n' << "cycles=" << controls.size() << '\n'
                  << "samples=" << controls.size() * 2U << '\n'
                  << "producer_transaction_count=" << transactions.size() << '\n'
                  << "producer_valid_cycles=" << producer_valid_cycles << '\n'
                  << "producer_ready_cycles=" << producer_ready_cycles << '\n'
                  << "producer_fire_cycles=" << producer_fire_cycles << '\n'
                  << "producer_stall_cycles=" << producer_stall_cycles << '\n'
                  << "max_consecutive_producer_stall_cycles=" << max_consecutive_stalls << '\n'
                  << "producer_accepted_count=" << accepted_count << '\n'
                  << "producer_pending_count=" << producer_pending_count << '\n'
                  << "producer_protocol_violations=" << producer_protocol_violations << '\n'
                  << "producer_accept_order_failures=" << producer_accept_order_failures << '\n'
                  << "transfer_invariant_failures=" << transfer_invariant_failures << '\n'
                  << "transfer_response_failures=" << transfer_response_failures << '\n'
                  << "transfer_latency_failures=" << transfer_latency_failures << '\n'
                  << "consumer_received_count=" << consumer_received_count << '\n'
                  << "consumer_order_failures=" << consumer_order_failures << '\n'
                  << "pending_transfer_responses=" << pending_transfers.size() << '\n'
                  << "final_a_count=" << fifo_a.current().count << '\n'
                  << "final_b_count=" << fifo_b.current().count << '\n'
                  << "drained=" << (drained ? "PASS" : "FAIL") << '\n'
                  << "scoreboard=" << (scoreboard_ok ? "PASS" : "FAIL") << '\n';
        return producer_protocol_violations == 0 && transfer_invariant_failures == 0 && drained && scoreboard_ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "two_fifo_cmodel: " << error.what() << '\n';
        return 1;
    }
}
