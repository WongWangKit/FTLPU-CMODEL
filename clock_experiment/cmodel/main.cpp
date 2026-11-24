#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Stimulus {
    bool rst{};
    bool en{};
    std::uint8_t din{};
};

struct PipelineRegisters {
    std::uint8_t q1{};
    std::uint8_t q2{};
};

class PipelineModel {
public:
    void evaluate(const Stimulus& stimulus) {
        next_ = current_;
        next_valid_ = current_valid_;

        if (stimulus.rst) {
            next_.q1 = 0;
            next_.q2 = 0;
            next_valid_ = true;
        } else if (stimulus.en) {
            next_.q1 = stimulus.din;
            next_.q2 = current_.q1;
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
    [[nodiscard]] bool current_valid() const { return current_valid_; }
    [[nodiscard]] const PipelineRegisters& current() const { return current_; }
    [[nodiscard]] const PipelineRegisters& next() const { return next_; }

private:
    void commit_all() {
        current_ = next_;
        current_valid_ = next_valid_;
    }

    std::uint64_t clk_{0};
    PipelineRegisters current_{};
    PipelineRegisters next_{};
    bool current_valid_{false};
    bool next_valid_{false};
};

static std::vector<Stimulus> load_stimulus(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open stimulus file: " + path);
    }

    std::vector<Stimulus> result;
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
        if (consumed != token.size() || raw > 0x3FFU) {
            throw std::runtime_error("invalid 10-bit stimulus at line " +
                                     std::to_string(line_number));
        }

        result.push_back(Stimulus{
            .rst = ((raw >> 9U) & 1U) != 0,
            .en = ((raw >> 8U) & 1U) != 0,
            .din = static_cast<std::uint8_t>(raw & 0xFFU),
        });
    }

    if (result.empty()) {
        throw std::runtime_error("stimulus file contains no vectors");
    }
    return result;
}

static std::string value_or_x(const bool valid, const std::uint8_t value) {
    return valid ? std::to_string(static_cast<unsigned>(value)) : "x";
}

static void write_low(std::ofstream& trace,
                      const PipelineModel& model,
                      const Stimulus& stimulus) {
    const auto& current = model.current();
    const auto& next = model.next();
    const bool valid = model.current_valid();
    trace << model.half_clk() << ',' << model.cycle_index()
          << ",LOW," << static_cast<unsigned>(model.clk_level()) << ','
          << static_cast<unsigned>(valid) << ','
          << static_cast<unsigned>(stimulus.rst) << ','
          << static_cast<unsigned>(stimulus.en) << ','
          << static_cast<unsigned>(stimulus.din) << ','
          << value_or_x(valid, current.q1) << ','
          << value_or_x(valid, current.q2) << ','
          << value_or_x(valid, current.q2) << ','
          << static_cast<unsigned>(next.q1) << ','
          << static_cast<unsigned>(next.q2) << '\n';
}

static void write_high(std::ofstream& trace,
                       const PipelineModel& model,
                       const Stimulus& stimulus) {
    const auto& current = model.current();
    const bool valid = model.current_valid();
    trace << model.half_clk() << ',' << model.cycle_index()
          << ",HIGH," << static_cast<unsigned>(model.clk_level()) << ','
          << static_cast<unsigned>(valid) << ','
          << static_cast<unsigned>(stimulus.rst) << ','
          << static_cast<unsigned>(stimulus.en) << ','
          << static_cast<unsigned>(stimulus.din) << ','
          << value_or_x(valid, current.q1) << ','
          << value_or_x(valid, current.q2) << ','
          << value_or_x(valid, current.q2) << ",,\n";
}

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "usage: clock_cmodel <stimulus.hex> <trace.csv>\n";
            return 2;
        }

        const auto stimuli = load_stimulus(argv[1]);
        std::ofstream trace(argv[2], std::ios::trunc);
        if (!trace) {
            throw std::runtime_error("cannot open trace file: " + std::string(argv[2]));
        }
        trace << "half_clk,cycle,phase,clk_level,valid,rst,en,din,q1,q2,dout,d1,d2\n";

        PipelineModel model;
        for (const auto& stimulus : stimuli) {
            model.evaluate(stimulus);
            write_low(trace, model, stimulus);

            model.advance_half_cycle();
            write_high(trace, model, stimulus);

            model.advance_half_cycle();
        }

        std::cout << "C++ trace complete: " << stimuli.size()
                  << " cycles, " << (stimuli.size() * 2U) << " samples\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "clock_cmodel: " << error.what() << '\n';
        return 1;
    }
}
