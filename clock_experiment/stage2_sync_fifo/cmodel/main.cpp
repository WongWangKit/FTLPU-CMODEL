#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kDepth = 4;

struct Inputs {
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

struct Combinational {
    bool full{};
    bool empty{};
    bool wr_ready{};
    bool rd_available{};
    bool wr_fire{};
    bool rd_fire{};
};

class SyncFifoModel {
public:
    void evaluate(const Inputs& input) {
        combinational_.full = current_.count == kDepth;
        combinational_.empty = current_.count == 0;
        combinational_.wr_ready = !combinational_.full;
        combinational_.rd_available = !combinational_.empty;
        combinational_.wr_fire = input.wr_en && combinational_.wr_ready && !input.rst;
        combinational_.rd_fire = input.rd_en && combinational_.rd_available && !input.rst;

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

        if (combinational_.wr_fire) {
            next_.memory[current_.wr_ptr] = input.wr_data;
            next_.memory_valid[current_.wr_ptr] = true;
            next_.wr_ptr = increment_pointer(current_.wr_ptr);
        }

        if (combinational_.rd_fire) {
            next_.rd_data = current_.memory[current_.rd_ptr];
            next_.rd_data_valid = true;
            next_.rd_ptr = increment_pointer(current_.rd_ptr);
        }

        if (combinational_.wr_fire && !combinational_.rd_fire) {
            ++next_.count;
        } else if (!combinational_.wr_fire && combinational_.rd_fire) {
            --next_.count;
        }
    }

    void advance_half_cycle() {
        const bool old_level = (clk_ & 1U) != 0;
        ++clk_;
        const bool new_level = (clk_ & 1U) != 0;
        if (!old_level && new_level) {
            commit_all();
        }
    }

    [[nodiscard]] std::uint64_t half_clk() const { return clk_; }
    [[nodiscard]] std::uint64_t cycle_index() const { return clk_ >> 1U; }
    [[nodiscard]] bool clk_level() const { return (clk_ & 1U) != 0; }
    [[nodiscard]] bool state_valid() const { return current_state_valid_; }
    [[nodiscard]] const FifoState& current() const { return current_; }
    [[nodiscard]] const FifoState& next() const { return next_; }
    [[nodiscard]] const Combinational& combinational() const { return combinational_; }

    [[nodiscard]] Combinational current_flags() const {
        Combinational result;
        result.full = current_.count == kDepth;
        result.empty = current_.count == 0;
        result.wr_ready = !result.full;
        result.rd_available = !result.empty;
        result.wr_fire = combinational_.wr_fire;
        result.rd_fire = combinational_.rd_fire;
        return result;
    }

private:
    static std::size_t increment_pointer(const std::size_t pointer) {
        return pointer == (kDepth - 1) ? 0 : pointer + 1;
    }

    void commit_all() {
        current_ = next_;
        current_state_valid_ = next_state_valid_;
    }

    std::uint64_t clk_{0};
    FifoState current_{};
    FifoState next_{};
    Combinational combinational_{};
    bool current_state_valid_{false};
    bool next_state_valid_{false};
};

std::vector<Inputs> load_stimulus(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open stimulus file: " + path);
    }

    std::vector<Inputs> result;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto comment = line.find('#');
        if (comment != std::string::npos) {
            line.erase(comment);
        }
        std::istringstream words(line);
        std::string token;
        if (!(words >> token)) {
            continue;
        }

        std::size_t consumed = 0;
        const auto raw = std::stoul(token, &consumed, 16);
        if (consumed != token.size() || raw > 0x7FFU) {
            throw std::runtime_error("invalid 11-bit stimulus at line " +
                                     std::to_string(line_number));
        }
        result.push_back(Inputs{
            .rst = ((raw >> 10U) & 1U) != 0,
            .wr_en = ((raw >> 9U) & 1U) != 0,
            .wr_data = static_cast<std::uint8_t>((raw >> 1U) & 0xFFU),
            .rd_en = (raw & 1U) != 0,
        });
    }

    if (result.empty()) {
        throw std::runtime_error("stimulus file contains no vectors");
    }
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
        const bool memory_valid = state_valid && state.memory_valid[index];
        trace << ',' << bool_or_x(state_valid, state.memory_valid[index])
              << ',' << byte_or_x(memory_valid, state.memory[index]);
    }
}

void write_prefix(std::ofstream& trace,
                  const SyncFifoModel& fifo,
                  const Inputs& input,
                  const char* phase) {
    trace << fifo.half_clk() << ',' << fifo.cycle_index() << ',' << phase << ','
          << static_cast<unsigned>(fifo.clk_level()) << ','
          << static_cast<unsigned>(fifo.state_valid()) << ','
          << static_cast<unsigned>(input.rst) << ','
          << static_cast<unsigned>(input.wr_en) << ','
          << static_cast<unsigned>(input.wr_data) << ','
          << static_cast<unsigned>(input.rd_en);
}

void write_low(std::ofstream& trace,
               const SyncFifoModel& fifo,
               const Inputs& input) {
    const auto& current = fifo.current();
    const auto& next = fifo.next();
    const auto& signals = fifo.combinational();
    const bool valid = fifo.state_valid();

    write_prefix(trace, fifo, input, "LOW");
    trace << ',' << bool_or_x(valid, signals.full)
          << ',' << bool_or_x(valid, signals.empty)
          << ',' << bool_or_x(valid, signals.wr_ready)
          << ',' << bool_or_x(valid, signals.rd_available)
          << ',' << static_cast<unsigned>(signals.wr_fire)
          << ',' << static_cast<unsigned>(signals.rd_fire)
          << ',' << number_or_x(valid, current.wr_ptr)
          << ',' << number_or_x(valid, current.rd_ptr)
          << ',' << number_or_x(valid, current.count)
          << ',' << byte_or_x(valid, current.rd_data)
          << ',' << bool_or_x(valid, current.rd_data_valid);
    write_memory(trace, current, valid);
    trace << ',' << next.wr_ptr
          << ',' << next.rd_ptr
          << ',' << next.count
          << ',' << static_cast<unsigned>(next.rd_data)
          << ',' << static_cast<unsigned>(next.rd_data_valid)
          << '\n';
}

void write_high(std::ofstream& trace,
                const SyncFifoModel& fifo,
                const Inputs& input) {
    const auto& current = fifo.current();
    const auto signals = fifo.current_flags();
    const bool valid = fifo.state_valid();

    write_prefix(trace, fifo, input, "HIGH");
    trace << ',' << bool_or_x(valid, signals.full)
          << ',' << bool_or_x(valid, signals.empty)
          << ',' << bool_or_x(valid, signals.wr_ready)
          << ',' << bool_or_x(valid, signals.rd_available)
          << ',' << static_cast<unsigned>(signals.wr_fire)
          << ',' << static_cast<unsigned>(signals.rd_fire)
          << ',' << number_or_x(valid, current.wr_ptr)
          << ',' << number_or_x(valid, current.rd_ptr)
          << ',' << number_or_x(valid, current.count)
          << ',' << byte_or_x(valid, current.rd_data)
          << ',' << bool_or_x(valid, current.rd_data_valid);
    write_memory(trace, current, valid);
    trace << ",,,,,\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "usage: sync_fifo_cmodel <stimulus.hex> <trace.csv>\n";
            return 2;
        }

        const auto stimuli = load_stimulus(argv[1]);
        std::ofstream trace(argv[2], std::ios::trunc);
        if (!trace) {
            throw std::runtime_error("cannot open trace file: " + std::string(argv[2]));
        }
        trace << "half_clk,cycle,phase,clk_level,state_valid,rst,wr_en,wr_data,rd_en,"
                 "full,empty,wr_ready,rd_available,wr_fire,rd_fire,wr_ptr_q,rd_ptr_q,"
                 "count_q,rd_data_q,rd_data_valid_q,mem0_valid,mem0,mem1_valid,mem1,"
                 "mem2_valid,mem2,mem3_valid,mem3,wr_ptr_d,rd_ptr_d,count_d,rd_data_d,"
                 "rd_data_valid_d\n";

        SyncFifoModel fifo;
        for (const auto& input : stimuli) {
            fifo.evaluate(input);
            write_low(trace, fifo, input);
            fifo.advance_half_cycle();
            write_high(trace, fifo, input);
            fifo.advance_half_cycle();
        }

        std::cout << "C++ FIFO trace complete: " << stimuli.size()
                  << " cycles, " << (stimuli.size() * 2U) << " samples\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sync_fifo_cmodel: " << error.what() << '\n';
        return 1;
    }
}
