#pragma once

#include "cpp_timing_modeling.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ftlpu::timing::framework::test {

struct Stimulus {
    bool reset{};
    bool reg_enable{};
    std::uint8_t reg_data{};
    bool fifo_wr_enable{};
    std::uint8_t fifo_wr_data{};
    bool fifo_rd_enable{};
    SyncSram<std::uint8_t, 4>::RequestKind sram_kind{};
    std::size_t sram_address{};
    std::uint8_t sram_write_data{};

    static Stimulus decode(const std::uint32_t word) {
        const auto raw_kind = static_cast<std::uint8_t>((word >> 10U) & 0x3U);
        if (raw_kind == 3) throw std::runtime_error("stimulus uses reserved SRAM operation 3");
        return {
            .reset = ((word >> 31U) & 1U) != 0,
            .reg_enable = ((word >> 30U) & 1U) != 0,
            .reg_data = static_cast<std::uint8_t>((word >> 22U) & 0xFFU),
            .fifo_wr_enable = ((word >> 21U) & 1U) != 0,
            .fifo_wr_data = static_cast<std::uint8_t>((word >> 13U) & 0xFFU),
            .fifo_rd_enable = ((word >> 12U) & 1U) != 0,
            .sram_kind = static_cast<SyncSram<std::uint8_t, 4>::RequestKind>(raw_kind),
            .sram_address = static_cast<std::size_t>((word >> 8U) & 0x3U),
            .sram_write_data = static_cast<std::uint8_t>(word & 0xFFU),
        };
    }
};

class FrameworkTop final : public ClockedModule {
public:
    using Fifo = SyncFifo<std::uint8_t, 4>;
    using Sram = SyncSram<std::uint8_t, 4>;

    struct StateImage {
        std::uint8_t source_q{};
        std::uint8_t source_next{};
        std::uint8_t destination_q{};
        std::uint8_t destination_next{};
        std::array<std::uint8_t, 2> bank_q{};
        std::array<std::uint8_t, 2> bank_next{};
        Fifo::State fifo_current{};
        Fifo::State fifo_next{};
        Sram::ReadOutput sram_output{};
        Sram::ReadOutput sram_next_output{};
        std::array<std::uint8_t, 4> sram_memory{};
        std::array<bool, 4> sram_initialized{};
        bool trace_state_valid{};
        constexpr bool operator==(const StateImage&) const = default;
    };

    explicit FrameworkTop(const bool reverse_commit = false) : reverse_commit_(reverse_commit) {}

    void drive(const Stimulus stimulus) { stimulus_ = stimulus; }

    void low() override {
        source_.drive(stimulus_.reg_data, stimulus_.reg_enable, stimulus_.reset);
        bank_.drive({stimulus_.reg_data, static_cast<std::uint8_t>(stimulus_.reg_data ^ 0x5AU)},
                    stimulus_.reg_enable, stimulus_.reset);
        fifo_.drive(stimulus_.reset, stimulus_.fifo_wr_enable,
                    stimulus_.fifo_wr_data, stimulus_.fifo_rd_enable);
        switch (stimulus_.sram_kind) {
            case Sram::RequestKind::Idle: sram_.request_idle(stimulus_.reset); break;
            case Sram::RequestKind::Read: sram_.request_read(stimulus_.sram_address, stimulus_.reset); break;
            case Sram::RequestKind::Write:
                sram_.request_write(stimulus_.sram_address, stimulus_.sram_write_data, stimulus_.reset);
                break;
        }

        // Explicit hardware-topology order. Destination samples the result
        // derived from Source's old Current; no register exists inside the MAC.
        source_.low();
        bank_.low();
        fifo_.low();
        sram_.low();
        combination_ = evaluate_mac(source_.q());
        combination_valid_ = trace_state_valid_;
        destination_.drive(combination_.result8, true, stimulus_.reset);
        destination_.low();
        prepared_trace_state_valid_ = trace_state_valid_ || stimulus_.reset;
    }

    void validate_posedge() const override {
        source_.validate_posedge();
        destination_.validate_posedge();
        bank_.validate_posedge();
        fifo_.validate_posedge();
        sram_.validate_posedge();
    }

    void posedge() override {
        if (!reverse_commit_) {
            source_.posedge(); destination_.posedge(); bank_.posedge(); fifo_.posedge(); sram_.posedge();
        } else {
            sram_.posedge(); fifo_.posedge(); bank_.posedge(); destination_.posedge(); source_.posedge();
        }
        trace_state_valid_ = prepared_trace_state_valid_;
    }

    void high() override {
        // Observation only. This may change after Source commits, while no
        // ClockedState or memory is touched.
        combination_ = evaluate_mac(source_.q());
        combination_valid_ = trace_state_valid_;
    }

    void negedge() override {}

    [[nodiscard]] StateImage state_image() const {
        return {
            source_.q(), source_.next(), destination_.q(), destination_.next(),
            bank_.q(), bank_.next(), fifo_.current(), fifo_.next(),
            sram_.read_output(), sram_.next_read_output(), sram_.memory(),
            sram_.initialized(), trace_state_valid_
        };
    }

    [[nodiscard]] const Stimulus& stimulus() const noexcept { return stimulus_; }
    [[nodiscard]] const MacSignals& combination() const noexcept { return combination_; }
    [[nodiscard]] bool combination_valid() const noexcept { return combination_valid_; }
    [[nodiscard]] const Fifo& fifo() const noexcept { return fifo_; }
    [[nodiscard]] const Sram& sram() const noexcept { return sram_; }
    [[nodiscard]] bool trace_state_valid() const noexcept { return trace_state_valid_; }

private:
    Register<std::uint8_t> source_{};
    Register<std::uint8_t> destination_{};
    RegisterBank<std::uint8_t, 2> bank_{};
    Fifo fifo_{};
    Sram sram_{};
    Stimulus stimulus_{};
    MacSignals combination_{};
    bool combination_valid_{};
    bool trace_state_valid_{};
    bool prepared_trace_state_valid_{};
    bool reverse_commit_{};
};

void run_framework_unit_tests();

}  // namespace ftlpu::timing::framework::test

