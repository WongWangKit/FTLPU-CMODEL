#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kDepth = 4;

struct Stimulus {
    bool rst{};
    bool producer_valid{};
    std::uint8_t producer_data{};
    bool consumer_ready{};
};

struct FifoInputs {
    bool rst{};
    bool wr_en{};
    std::uint8_t wr_data{};
    bool rd_en{};
};

struct FifoState {
    std::array<std::uint8_t, kDepth> memory{};
    std::array<bool, kDepth> memory_valid{};
    std::size_t wr_ptr{};
    std::size_t rd_ptr{};
    std::size_t count{};
    std::uint8_t rd_data{};
    bool rd_data_valid{};
};

struct FifoSignals {
    bool full{};
    bool empty{};
    bool wr_ready{};
    bool rd_available{};
    bool wr_fire{};
    bool rd_fire{};
};

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
            next_.wr_ptr = 0;
            next_.rd_ptr = 0;
            next_.count = 0;
            next_.rd_data = 0;
            next_.rd_data_valid = false;
            next_.memory_valid.fill(false);
            return;
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

        if (signals_.wr_fire && !signals_.rd_fire) {
            ++next_.count;
        } else if (!signals_.wr_fire && signals_.rd_fire) {
            --next_.count;
        }
    }

    void commit() {
        current_ = next_;
        current_state_valid_ = next_state_valid_;
    }

    [[nodiscard]] const FifoState& current() const { return current_; }
    [[nodiscard]] const FifoState& next() const { return next_; }
    [[nodiscard]] const FifoSignals& signals() const { return signals_; }
    [[nodiscard]] bool state_valid() const { return current_state_valid_; }

    [[nodiscard]] FifoSignals current_flags() const {
        FifoSignals result = signals_;
        result.full = current_.count == kDepth;
        result.empty = current_.count == 0;
        result.wr_ready = !result.full;
        result.rd_available = !result.empty;
        return result;
    }

private:
    static std::size_t increment_pointer(const std::size_t pointer) {
        return pointer == (kDepth - 1) ? 0 : pointer + 1;
    }

    FifoState current_{};
    FifoState next_{};
    FifoSignals signals_{};
    bool current_state_valid_{false};
    bool next_state_valid_{false};
};

struct TransferSignals {
    bool a_wr_en{};
    std::uint8_t a_wr_data{};
    bool a_rd_en{};
    bool b_wr_en{};
    std::uint8_t b_wr_data{};
    bool b_rd_en{};
    bool producer_ready{};
    bool transfer_invariant_ok{};
};

class TransferController {
public:
    [[nodiscard]] TransferSignals evaluate(const FifoState& a,
                                           const FifoState& b,
                                           const Stimulus& input) const {
        const bool a_full = a.count == kDepth;
        const bool a_empty = a.count == 0;
        const bool b_full = b.count == kDepth;

        TransferSignals result;
        result.a_wr_en = input.producer_valid;
        result.a_wr_data = input.producer_data;
        result.a_rd_en = !input.rst && !a_empty && !a.rd_data_valid && !b_full;
        result.b_wr_en = !input.rst && a.rd_data_valid;
        result.b_wr_data = a.rd_data;
        result.b_rd_en = !input.rst && input.consumer_ready;
        result.producer_ready = !input.rst && !a_full;
        result.transfer_invariant_ok = !(a.rd_data_valid && b_full);
        return result;
    }
};

struct Schedule {
    bool evaluate_a_first{};
    bool commit_a_first{};
};

Schedule parse_schedule(const std::string_view name) {
    if (name == "eval_ab_commit_ab") return {true, true};
    if (name == "eval_ab_commit_ba") return {true, false};
    if (name == "eval_ba_commit_ab") return {false, true};
    if (name == "eval_ba_commit_ba") return {false, false};
    throw std::runtime_error("unknown schedule: " + std::string(name));
}

std::vector<Stimulus> load_stimulus(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open stimulus file: " + path);

    std::vector<Stimulus> result;
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
        if (consumed != token.size() || raw > 0x7FFU) {
            throw std::runtime_error("invalid 11-bit stimulus at line " +
                                     std::to_string(line_number));
        }
        result.push_back(Stimulus{
            .rst = ((raw >> 10U) & 1U) != 0,
            .producer_valid = ((raw >> 9U) & 1U) != 0,
            .producer_data = static_cast<std::uint8_t>((raw >> 1U) & 0xFFU),
            .consumer_ready = (raw & 1U) != 0,
        });
    }
    if (result.empty()) throw std::runtime_error("stimulus file contains no vectors");
    return result;
}

std::string number_or_x(const bool valid, const std::size_t value) {
    return valid ? std::to_string(value) : "x";
}

std::string byte_or_x(const bool valid, const std::uint8_t value) {
    return valid ? std::to_string(static_cast<unsigned>(value)) : "x";
}

std::string bool_or_x(const bool valid, const bool value) {
    return valid ? (value ? "1" : "0") : "x";
}

void write_memory(std::ofstream& trace, const FifoState& state, const bool state_valid) {
    for (std::size_t index = 0; index < kDepth; ++index) {
        const bool data_valid = state_valid && state.memory_valid[index];
        trace << ',' << bool_or_x(state_valid, state.memory_valid[index])
              << ',' << byte_or_x(data_valid, state.memory[index]);
    }
}

void write_fifo_low(std::ofstream& trace, const SyncFifo& fifo) {
    const auto& state = fifo.current();
    const auto& next = fifo.next();
    const auto& sig = fifo.signals();
    const bool valid = fifo.state_valid();
    trace << ',' << bool_or_x(valid, sig.full)
          << ',' << bool_or_x(valid, sig.empty)
          << ',' << bool_or_x(valid, sig.wr_ready)
          << ',' << bool_or_x(valid, sig.rd_available)
          << ',' << static_cast<unsigned>(sig.wr_fire)
          << ',' << static_cast<unsigned>(sig.rd_fire)
          << ',' << static_cast<unsigned>(valid)
          << ',' << number_or_x(valid, state.wr_ptr)
          << ',' << number_or_x(valid, state.rd_ptr)
          << ',' << number_or_x(valid, state.count)
          << ',' << byte_or_x(valid, state.rd_data)
          << ',' << bool_or_x(valid, state.rd_data_valid);
    write_memory(trace, state, valid);
    trace << ',' << next.wr_ptr
          << ',' << next.rd_ptr
          << ',' << next.count
          << ',' << static_cast<unsigned>(next.rd_data)
          << ',' << static_cast<unsigned>(next.rd_data_valid);
}

void write_fifo_high(std::ofstream& trace, const SyncFifo& fifo) {
    const auto& state = fifo.current();
    const auto sig = fifo.current_flags();
    const bool valid = fifo.state_valid();
    trace << ',' << bool_or_x(valid, sig.full)
          << ',' << bool_or_x(valid, sig.empty)
          << ',' << bool_or_x(valid, sig.wr_ready)
          << ',' << bool_or_x(valid, sig.rd_available)
          << ',' << static_cast<unsigned>(sig.wr_fire)
          << ',' << static_cast<unsigned>(sig.rd_fire)
          << ',' << static_cast<unsigned>(valid)
          << ',' << number_or_x(valid, state.wr_ptr)
          << ',' << number_or_x(valid, state.rd_ptr)
          << ',' << number_or_x(valid, state.count)
          << ',' << byte_or_x(valid, state.rd_data)
          << ',' << bool_or_x(valid, state.rd_data_valid);
    write_memory(trace, state, valid);
    trace << ",,,,,";
}

void write_common(std::ofstream& trace,
                  const std::uint64_t half_clk,
                  const char* phase,
                  const Stimulus& input,
                  const TransferSignals& transfer) {
    trace << half_clk << ',' << (half_clk >> 1U) << ',' << phase << ','
          << static_cast<unsigned>(half_clk & 1U) << ','
          << static_cast<unsigned>(input.rst) << ','
          << static_cast<unsigned>(input.producer_valid) << ','
          << static_cast<unsigned>(input.producer_data) << ','
          << static_cast<unsigned>(input.consumer_ready) << ','
          << static_cast<unsigned>(transfer.a_rd_en) << ','
          << static_cast<unsigned>(transfer.b_wr_en) << ','
          << static_cast<unsigned>(transfer.b_wr_data) << ','
          << static_cast<unsigned>(transfer.b_rd_en) << ','
          << static_cast<unsigned>(transfer.producer_ready) << ','
          << static_cast<unsigned>(transfer.transfer_invariant_ok);
}

void write_header(std::ofstream& trace) {
    trace << "half_clk,cycle,phase,clk_level,rst,producer_valid,producer_data,consumer_ready,"
             "tc_a_rd_en,tc_b_wr_en,tc_b_wr_data,tc_b_rd_en,producer_ready,transfer_invariant_ok";
    for (const char* prefix : {"a_", "b_"}) {
        trace << ',' << prefix << "full"
              << ',' << prefix << "empty"
              << ',' << prefix << "wr_ready"
              << ',' << prefix << "rd_available"
              << ',' << prefix << "wr_fire"
              << ',' << prefix << "rd_fire"
              << ',' << prefix << "state_valid"
              << ',' << prefix << "wr_ptr_q"
              << ',' << prefix << "rd_ptr_q"
              << ',' << prefix << "count_q"
              << ',' << prefix << "rd_data_q"
              << ',' << prefix << "rd_data_valid_q";
        for (std::size_t slot = 0; slot < kDepth; ++slot) {
            trace << ',' << prefix << "mem" << slot << "_valid"
                  << ',' << prefix << "mem" << slot;
        }
        trace << ',' << prefix << "wr_ptr_d"
              << ',' << prefix << "rd_ptr_d"
              << ',' << prefix << "count_d"
              << ',' << prefix << "rd_data_d"
              << ',' << prefix << "rd_data_valid_d";
    }
    trace << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr << "usage: two_fifo_cmodel <stimulus.hex> <trace.csv> <schedule>\n";
            return 2;
        }

        const auto stimuli = load_stimulus(argv[1]);
        const auto schedule = parse_schedule(argv[3]);
        std::ofstream trace(argv[2], std::ios::trunc);
        if (!trace) throw std::runtime_error("cannot open trace file: " + std::string(argv[2]));
        write_header(trace);

        SyncFifo fifo_a;
        SyncFifo fifo_b;
        TransferController controller;
        std::uint64_t half_clk = 0;
        std::vector<std::uint8_t> accepted_data;
        std::size_t consumed_count = 0;
        std::size_t invariant_failures = 0;
        std::size_t data_failures = 0;

        for (const auto& stimulus : stimuli) {
            if (stimulus.rst) {
                accepted_data.clear();
                consumed_count = 0;
            }

            const auto transfer = controller.evaluate(fifo_a.current(), fifo_b.current(), stimulus);
            const FifoInputs a_input{
                .rst = stimulus.rst,
                .wr_en = transfer.a_wr_en,
                .wr_data = transfer.a_wr_data,
                .rd_en = transfer.a_rd_en,
            };
            const FifoInputs b_input{
                .rst = stimulus.rst,
                .wr_en = transfer.b_wr_en,
                .wr_data = transfer.b_wr_data,
                .rd_en = transfer.b_rd_en,
            };

            if (schedule.evaluate_a_first) {
                fifo_a.evaluate(a_input);
                fifo_b.evaluate(b_input);
            } else {
                fifo_b.evaluate(b_input);
                fifo_a.evaluate(a_input);
            }

            if (!transfer.transfer_invariant_ok) ++invariant_failures;
            if (fifo_a.signals().wr_fire) accepted_data.push_back(stimulus.producer_data);

            write_common(trace, half_clk, "LOW", stimulus, transfer);
            write_fifo_low(trace, fifo_a);
            write_fifo_low(trace, fifo_b);
            trace << '\n';

            ++half_clk;
            if (schedule.commit_a_first) {
                fifo_a.commit();
                fifo_b.commit();
            } else {
                fifo_b.commit();
                fifo_a.commit();
            }

            if (fifo_b.signals().rd_fire) {
                if (!fifo_b.current().rd_data_valid || consumed_count >= accepted_data.size() ||
                    fifo_b.current().rd_data != accepted_data[consumed_count]) {
                    ++data_failures;
                }
                ++consumed_count;
            }

            write_common(trace, half_clk, "HIGH", stimulus, transfer);
            write_fifo_high(trace, fifo_a);
            write_fifo_high(trace, fifo_b);
            trace << '\n';
            ++half_clk;
        }

        const bool drained = fifo_a.current().count == 0 && fifo_b.current().count == 0 &&
                             !fifo_a.current().rd_data_valid;
        const bool scoreboard_ok = consumed_count == accepted_data.size() && data_failures == 0;
        std::cout << "schedule=" << argv[3] << '\n'
                  << "cycles=" << stimuli.size() << '\n'
                  << "samples=" << stimuli.size() * 2U << '\n'
                  << "producer_accepted=" << accepted_data.size() << '\n'
                  << "consumer_received=" << consumed_count << '\n'
                  << "transfer_invariant_failures=" << invariant_failures << '\n'
                  << "data_order_failures=" << data_failures << '\n'
                  << "drained=" << (drained ? "PASS" : "FAIL") << '\n'
                  << "scoreboard=" << (scoreboard_ok ? "PASS" : "FAIL") << '\n';

        return invariant_failures == 0 && drained && scoreboard_ok ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "two_fifo_cmodel: " << error.what() << '\n';
        return 1;
    }
}
