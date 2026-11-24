#include "test_modules.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fw = ftlpu::timing::framework;
namespace test = ftlpu::timing::framework::test;

namespace {

std::vector<test::Stimulus> load_stimulus(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open stimulus: " + path.string());
    std::vector<test::Stimulus> result;
    std::string line;
    while (std::getline(input, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        std::istringstream parser(line);
        std::uint32_t word{};
        if (parser >> std::hex >> word) result.push_back(test::Stimulus::decode(word));
    }
    if (result.empty()) throw std::runtime_error("stimulus contains no vectors");
    return result;
}

std::string value_or_x(const bool valid, const std::uint64_t value) {
    return valid ? std::to_string(value) : "x";
}

void write_header(std::ostream& out) {
    out << "cycle,phase,clk,rst,reg_en,reg_data,fifo_wr_en,fifo_wr_data,fifo_rd_en,"
           "sram_op,sram_addr,sram_wdata,state_valid,"
           "src_q,src_next,dst_q,dst_next,bank0_q,bank1_q,bank0_next,bank1_next,"
           "mac_mul0,mac_add0,mac_mul1,mac_add1,mac_mul2,mac_result16,mac_result8,"
           "fifo_full,fifo_empty,fifo_wr_ready,fifo_rd_available,fifo_wr_fire,fifo_rd_fire,"
           "fifo_wr_ptr,fifo_rd_ptr,fifo_count,fifo_rd_data,fifo_rd_data_valid,"
           "fifo_m0,fifo_m1,fifo_m2,fifo_m3,fifo_v0,fifo_v1,fifo_v2,fifo_v3,"
           "fifo_n_wr_ptr,fifo_n_rd_ptr,fifo_n_count,fifo_n_rd_data,fifo_n_rd_data_valid,"
           "fifo_n_m0,fifo_n_m1,fifo_n_m2,fifo_n_m3,fifo_n_v0,fifo_n_v1,fifo_n_v2,fifo_n_v3,"
           "sram_rd_data,sram_rd_valid,sram_rd_initialized,"
           "sram_n_rd_data,sram_n_rd_valid,sram_n_rd_initialized,"
           "sram_m0,sram_m1,sram_m2,sram_m3,sram_i0,sram_i1,sram_i2,sram_i3\n";
}

void write_row(std::ostream& out, const std::size_t cycle, const int phase,
               const test::FrameworkTop& top) {
    const auto image = top.state_image();
    const auto& stimulus = top.stimulus();
    const auto& mac = top.combination();
    const auto fifo_flags = top.fifo().current_flags();
    const auto fifo_events = top.fifo().preedge_signals();
    const bool state_valid = top.trace_state_valid();
    const auto& fifo = image.fifo_current;
    const auto& fifo_next = image.fifo_next;
    const auto& sram = image.sram_output;
    const auto& sram_next = image.sram_next_output;

    out << cycle << ',' << phase << ',' << ((phase == 1 || phase == 2) ? 1 : 0) << ','
        << stimulus.reset << ',' << stimulus.reg_enable << ',' << +stimulus.reg_data << ','
        << stimulus.fifo_wr_enable << ',' << +stimulus.fifo_wr_data << ',' << stimulus.fifo_rd_enable << ','
        << static_cast<int>(stimulus.sram_kind) << ',' << stimulus.sram_address << ','
        << +stimulus.sram_write_data << ',' << state_valid << ','
        << value_or_x(state_valid, image.source_q) << ',' << +image.source_next << ','
        << value_or_x(state_valid, image.destination_q) << ',' << +image.destination_next << ','
        << value_or_x(state_valid, image.bank_q[0]) << ',' << value_or_x(state_valid, image.bank_q[1]) << ','
        << +image.bank_next[0] << ',' << +image.bank_next[1] << ','
        << value_or_x(top.combination_valid(), mac.mul0) << ','
        << value_or_x(top.combination_valid(), mac.add0) << ','
        << value_or_x(top.combination_valid(), mac.mul1) << ','
        << value_or_x(top.combination_valid(), mac.add1) << ','
        << value_or_x(top.combination_valid(), mac.mul2) << ','
        << value_or_x(top.combination_valid(), mac.result16) << ','
        << value_or_x(top.combination_valid(), mac.result8) << ','
        << value_or_x(state_valid, fifo_flags.full) << ','
        << value_or_x(state_valid, fifo_flags.empty) << ','
        << value_or_x(state_valid, fifo_flags.wr_ready) << ','
        << value_or_x(state_valid, fifo_flags.rd_available) << ','
        << fifo_events.wr_fire << ',' << fifo_events.rd_fire << ','
        << value_or_x(state_valid, fifo.wr_ptr) << ',' << value_or_x(state_valid, fifo.rd_ptr) << ','
        << value_or_x(state_valid, fifo.count) << ',' << value_or_x(state_valid, fifo.rd_data) << ','
        << value_or_x(state_valid, fifo.rd_data_valid) << ',';
    for (std::size_t i = 0; i < 4; ++i)
        out << value_or_x(state_valid && fifo.memory_valid[i], fifo.memory[i]) << ',';
    for (std::size_t i = 0; i < 4; ++i) out << value_or_x(state_valid, fifo.memory_valid[i]) << ',';
    out << fifo_next.wr_ptr << ',' << fifo_next.rd_ptr << ',' << fifo_next.count << ','
        << +fifo_next.rd_data << ',' << fifo_next.rd_data_valid << ',';
    for (std::size_t i = 0; i < 4; ++i)
        out << value_or_x(fifo_next.memory_valid[i], fifo_next.memory[i]) << ',';
    for (std::size_t i = 0; i < 4; ++i) out << fifo_next.memory_valid[i] << ',';
    out << value_or_x(state_valid, sram.data) << ',' << value_or_x(state_valid, sram.valid) << ','
        << value_or_x(state_valid, sram.initialized) << ','
        << +sram_next.data << ',' << sram_next.valid << ',' << sram_next.initialized << ',';
    for (std::size_t i = 0; i < 4; ++i)
        out << value_or_x(image.sram_initialized[i], image.sram_memory[i]) << ',';
    for (std::size_t i = 0; i < 4; ++i) {
        out << image.sram_initialized[i];
        out << (i == 3 ? '\n' : ',');
    }
}

void require_equal(const test::FrameworkTop& first, const test::FrameworkTop& second,
                   const std::string& context) {
    if (!(first.state_image() == second.state_image()) || !(first.combination() == second.combination()) ||
        first.combination_valid() != second.combination_valid())
        throw std::runtime_error("commit-order independence failed at " + context);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "usage: clocked_module_framework_cmodel <stimulus.hex> <trace.csv>\n";
            return 2;
        }
        test::run_framework_unit_tests();
        const auto vectors = load_stimulus(argv[1]);
        std::ofstream trace(argv[2]);
        if (!trace) throw std::runtime_error("cannot create trace output");
        write_header(trace);

        test::FrameworkTop normal(false);
        test::FrameworkTop reversed(true);
        fw::Clock normal_clock;
        fw::Clock reversed_clock;
        normal_clock.register_participant(normal);
        reversed_clock.register_participant(reversed);
        std::size_t order_checks{};
        std::size_t readonly_checks{};

        for (std::size_t cycle = 0; cycle < vectors.size(); ++cycle) {
            normal.drive(vectors[cycle]);
            reversed.drive(vectors[cycle]);

            normal_clock.step(ftlpu::timing::ClockPhase::LowEvaluate);
            reversed_clock.step(ftlpu::timing::ClockPhase::LowEvaluate);
            require_equal(normal, reversed, "cycle " + std::to_string(cycle) + " phase 0");
            write_row(trace, cycle, 0, normal);

            normal_clock.step(ftlpu::timing::ClockPhase::PosedgeCommit);
            reversed_clock.step(ftlpu::timing::ClockPhase::PosedgeCommit);
            require_equal(normal, reversed, "cycle " + std::to_string(cycle) + " phase 1");
            ++order_checks;
            write_row(trace, cycle, 1, normal);
            const auto committed = normal.state_image();

            normal_clock.step(ftlpu::timing::ClockPhase::PostEdgeSettle);
            reversed_clock.step(ftlpu::timing::ClockPhase::PostEdgeSettle);
            require_equal(normal, reversed, "cycle " + std::to_string(cycle) + " phase 2");
            if (!(normal.state_image() == committed))
                throw std::runtime_error("Phase 2 modified sequential state at cycle " + std::to_string(cycle));
            ++readonly_checks;
            write_row(trace, cycle, 2, normal);

            normal_clock.step(ftlpu::timing::ClockPhase::Negedge);
            reversed_clock.step(ftlpu::timing::ClockPhase::Negedge);
            require_equal(normal, reversed, "cycle " + std::to_string(cycle) + " phase 3");
            if (!(normal.state_image() == committed))
                throw std::runtime_error("Phase 3 modified sequential state at cycle " + std::to_string(cycle));
            write_row(trace, cycle, 3, normal);
        }

        std::cout << "framework_unit_tests=PASS\n"
                  << "stimulus_cycles=" << vectors.size() << "\n"
                  << "trace_rows=" << vectors.size() * 4 << "\n"
                  << "commit_order_checks=" << order_checks << "\n"
                  << "phase2_readonly_checks=" << readonly_checks << "\n"
                  << "result=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "result=FAIL\nerror=" << error.what() << '\n';
        return 1;
    }
}

