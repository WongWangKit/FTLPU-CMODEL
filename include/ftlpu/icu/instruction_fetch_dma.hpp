#pragma once

#include "ftlpu/axi/axi4_memory_slave.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace ftlpu {

enum class PhysicalInstructionWordWidth : std::size_t {
    Bits96 = 96,
    Bits128 = 128,
};

constexpr std::size_t physical_instruction_word_bytes(
    PhysicalInstructionWordWidth width) noexcept
{
    return static_cast<std::size_t>(width) / 8;
}

// External program-image ABI for one physical local-iMEM word:
//   * bytes increase with the DRAM byte address;
//   * bytes [3:0] form 32-bit lane 0 in little-endian order;
//   * lane 0 is physical word bits [31:0], matching IcuRawImemWord.
// Only byte_count bytes are valid; the fixed array is not an instruction
// buffer and merely accommodates the largest supported physical word.
struct PhysicalInstructionWord {
    static constexpr std::size_t kMaximumBytes = 16;

    std::size_t index{0};
    PhysicalInstructionWordWidth width{
        PhysicalInstructionWordWidth::Bits96};
    std::size_t byte_count{0};
    std::array<std::uint8_t, kMaximumBytes> bytes{};

    std::uint32_t lane32(std::size_t lane) const
    {
        const auto offset = lane * 4;
        if (offset + 4 > byte_count)
            throw std::out_of_range(
                "physical instruction lane is outside the word");
        return static_cast<std::uint32_t>(bytes[offset])
            | (static_cast<std::uint32_t>(bytes[offset + 1]) << 8)
            | (static_cast<std::uint32_t>(bytes[offset + 2]) << 16)
            | (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
    }
};

struct InstructionFetchProgram {
    std::uint64_t base_address{0};
    std::size_t word_count{0};
    PhysicalInstructionWordWidth word_width{
        PhysicalInstructionWordWidth::Bits96};
};

struct InstructionFetchDmaConfig {
    std::size_t axi_data_width_bits{256};
    std::size_t burst_beats{16};
    // The field remains explicit so the later multi-outstanding version does
    // not need a configuration ABI change. This first implementation accepts
    // exactly one outstanding burst.
    std::size_t max_outstanding{1};
    std::uint32_t axi_id{0};

    std::size_t axi_data_bytes() const noexcept
    {
        return axi_data_width_bits / 8;
    }

    void validate() const
    {
        if (axi_data_width_bits == 0 || axi_data_width_bits % 8 != 0
            || !std::has_single_bit(axi_data_bytes())) {
            throw std::invalid_argument(
                "instruction-fetch AXI width must contain a power-of-two byte count");
        }
        if (burst_beats == 0 || burst_beats > 256)
            throw std::invalid_argument(
                "instruction-fetch burst length must be in [1, 256]");
        if (max_outstanding != 1)
            throw std::invalid_argument(
                "the first instruction-fetch DMA supports one outstanding burst");
    }
};

// Bounded byte reservoir used only to bridge AXI beat and instruction-word
// widths. Its capacity is fixed at construction and it never allocates in
// response to backpressure.
class InstructionWordAssembler {
public:
    InstructionWordAssembler(std::size_t word_bytes,
        std::size_t axi_data_bytes)
        : word_bytes_(word_bytes)
        , storage_(axi_data_bytes + word_bytes - 1, 0)
    {
        if ((word_bytes != 12 && word_bytes != 16)
            || axi_data_bytes == 0) {
            throw std::invalid_argument(
                "instruction assembler requires 12/16-byte words and a non-zero AXI width");
        }
    }

    std::size_t capacity() const noexcept { return storage_.size(); }
    std::size_t occupancy() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    bool word_ready() const noexcept { return size_ >= word_bytes_; }

    bool can_accept(std::size_t byte_count,
        bool remove_word_this_cycle = false) const noexcept
    {
        const auto occupancyAfterRemoval = size_
            - (remove_word_this_cycle ? word_bytes_ : 0);
        return byte_count <= capacity() - occupancyAfterRemoval;
    }

    void append(std::span<const std::uint8_t> bytes)
    {
        if (!can_accept(bytes.size()))
            throw std::logic_error(
                "instruction assembler capacity was exceeded");
        for (const auto byte : bytes) {
            storage_[(head_ + size_) % capacity()] = byte;
            ++size_;
        }
    }

    PhysicalInstructionWord front(std::size_t word_index,
        PhysicalInstructionWordWidth width) const
    {
        if (!word_ready())
            throw std::logic_error(
                "instruction assembler does not hold a complete word");
        auto word = PhysicalInstructionWord {};
        word.index = word_index;
        word.width = width;
        word.byte_count = word_bytes_;
        for (std::size_t byte = 0; byte < word_bytes_; ++byte)
            word.bytes[byte] = storage_[(head_ + byte) % capacity()];
        return word;
    }

    void pop_word()
    {
        if (!word_ready())
            throw std::logic_error(
                "instruction assembler cannot pop an incomplete word");
        head_ = (head_ + word_bytes_) % capacity();
        size_ -= word_bytes_;
    }

private:
    std::size_t word_bytes_{0};
    std::vector<std::uint8_t> storage_{};
    std::size_t head_{0};
    std::size_t size_{0};
};

struct InstructionFetchBurstTrace {
    std::size_t ar_handshake_cycle{0};
    std::uint64_t address{0};
    std::size_t beat_bytes{0};
    std::size_t beat_count{0};
};

struct InstructionFetchDmaTrace {
    std::vector<InstructionFetchBurstTrace> bursts{};
    std::vector<std::size_t> r_beat_cycles{};
    std::vector<std::size_t> word_assembled_cycles{};
    std::vector<std::size_t> word_enqueue_cycles{};
};

struct InstructionFetchDmaStatistics {
    std::size_t cycle_count{0};
    std::size_t ar_handshakes{0};
    std::size_t r_handshakes{0};
    std::size_t bytes_received{0};
    std::size_t words_assembled{0};
    std::size_t words_enqueued{0};
    std::size_t axi_r_backpressure_cycles{0};
    std::size_t queue_backpressure_cycles{0};
    std::size_t assembler_peak_bytes{0};

    double effective_payload_bytes_per_cycle() const noexcept
    {
        return cycle_count == 0
            ? 0.0
            : static_cast<double>(bytes_received)
                / static_cast<double>(cycle_count);
    }
};

struct InstructionFetchDmaOutputs {
    axi::Axi4ReadInputs axi{};
    bool word_valid{false};
    PhysicalInstructionWord word{};
};

struct InstructionFetchDmaCycle {
    std::size_t cycle{0};
    InstructionFetchDmaOutputs outputs{};
    bool ar_fire{false};
    bool r_fire{false};
    bool word_fire{false};
    bool complete{false};
};

class InstructionFetchDma {
public:
    explicit InstructionFetchDma(InstructionFetchDmaConfig config)
        : config_(config)
    {
        config_.validate();
    }

    void start(InstructionFetchProgram program)
    {
        if (started_ && !complete_)
            throw std::logic_error(
                "instruction-fetch DMA is already active");
        validate_program(program);

        program_ = program;
        word_bytes_ = physical_instruction_word_bytes(program.word_width);
        if (program.word_count
            > std::numeric_limits<std::size_t>::max() / word_bytes_) {
            throw std::out_of_range(
                "instruction-fetch program byte count overflows");
        }
        total_bytes_ = program.word_count * word_bytes_;
        if (total_bytes_ != 0
            && total_bytes_ - 1
                > std::numeric_limits<std::uint64_t>::max()
                    - program.base_address) {
            throw std::out_of_range(
                "instruction-fetch program crosses the address space");
        }

        assembler_.emplace(word_bytes_, config_.axi_data_bytes());
        pending_ar_.reset();
        active_burst_.reset();
        bytes_requested_ = 0;
        bytes_received_ = 0;
        words_assembled_ = 0;
        words_enqueued_ = 0;
        cycle_ = 0;
        trace_ = {};
        statistics_ = {};
        started_ = true;
        complete_ = total_bytes_ == 0;
        if (!complete_)
            prepare_next_burst();
    }

    InstructionFetchDmaOutputs evaluate(
        const axi::Axi4ReadOutputs& axi_outputs,
        bool word_ready) const
    {
        require_started();
        auto outputs = InstructionFetchDmaOutputs {};

        if (pending_ar_.has_value()) {
            outputs.axi.arvalid = true;
            outputs.axi.araddr = pending_ar_->address;
            outputs.axi.arlen = static_cast<std::uint8_t>(
                pending_ar_->beat_count - 1);
            outputs.axi.arsize = static_cast<std::uint8_t>(
                std::countr_zero(pending_ar_->beat_bytes));
            outputs.axi.arburst = axi::AxiBurst::Incrementing;
            outputs.axi.arid = config_.axi_id;
        }

        if (assembler_->word_ready()) {
            outputs.word_valid = true;
            outputs.word = assembler_->front(
                words_enqueued_, program_.word_width);
        }

        const auto wordFire = outputs.word_valid && word_ready;
        if (active_burst_.has_value()) {
            outputs.axi.rready = assembler_->can_accept(
                active_burst_->beat_bytes, wordFire);
        }

        (void)axi_outputs;
        return outputs;
    }

    // The caller supplies the AXI slave outputs observed in this cycle and
    // then presents cycle.outputs.axi to Axi4MemorySlave::tick(). Both blocks
    // therefore make handshake decisions from the same pre-commit state.
    InstructionFetchDmaCycle tick(
        const axi::Axi4ReadOutputs& axi_outputs,
        bool word_ready)
    {
        auto result = InstructionFetchDmaCycle {};
        result.cycle = cycle_;
        result.outputs = evaluate(axi_outputs, word_ready);
        result.ar_fire = result.outputs.axi.arvalid
            && axi_outputs.arready;
        result.r_fire = axi_outputs.rvalid
            && result.outputs.axi.rready;
        result.word_fire = result.outputs.word_valid && word_ready;

        if (axi_outputs.rvalid && !result.outputs.axi.rready)
            ++statistics_.axi_r_backpressure_cycles;
        if (result.outputs.word_valid && !word_ready)
            ++statistics_.queue_backpressure_cycles;

        if (result.word_fire) {
            assembler_->pop_word();
            trace_.word_enqueue_cycles.push_back(cycle_);
            ++words_enqueued_;
            ++statistics_.words_enqueued;
        }

        if (result.r_fire)
            accept_r_beat(axi_outputs);

        if (result.ar_fire)
            accept_ar();

        statistics_.assembler_peak_bytes = std::max(
            statistics_.assembler_peak_bytes, assembler_->occupancy());
        complete_ = words_enqueued_ == program_.word_count
            && bytes_received_ == total_bytes_
            && !pending_ar_.has_value() && !active_burst_.has_value()
            && assembler_->empty();

        ++cycle_;
        statistics_.cycle_count = cycle_;
        result.complete = complete_;
        return result;
    }

    bool started() const noexcept { return started_; }
    bool complete() const noexcept { return complete_; }
    std::size_t cycle() const noexcept { return cycle_; }
    std::size_t assembler_capacity() const
    {
        require_started();
        return assembler_->capacity();
    }
    std::size_t assembler_occupancy() const
    {
        require_started();
        return assembler_->occupancy();
    }
    const InstructionFetchDmaTrace& trace() const noexcept
    {
        return trace_;
    }
    const InstructionFetchDmaStatistics& statistics() const noexcept
    {
        return statistics_;
    }

private:
    struct Burst {
        std::uint64_t address{0};
        std::size_t beat_bytes{0};
        std::size_t beat_count{0};
        std::size_t beats_received{0};
    };

    static void validate_program(const InstructionFetchProgram& program)
    {
        const auto bits = static_cast<std::size_t>(program.word_width);
        if (bits != 96 && bits != 128)
            throw std::invalid_argument(
                "instruction-fetch word width must be 96 or 128 bits");
    }

    void require_started() const
    {
        if (!started_)
            throw std::logic_error(
                "instruction-fetch DMA has not been started");
    }

    void prepare_next_burst()
    {
        if (bytes_requested_ == total_bytes_)
            return;
        if (pending_ar_.has_value() || active_burst_.has_value())
            throw std::logic_error(
                "instruction-fetch prepared a second outstanding burst");

        const auto address = program_.base_address + bytes_requested_;
        const auto remaining = total_bytes_ - bytes_requested_;
        const auto bytesTo4k = static_cast<std::size_t>(
            4096 - (address & std::uint64_t {4095}));
        const auto beatLimit = std::min(
            {config_.axi_data_bytes(), remaining, bytesTo4k});
        auto beatBytes = std::bit_floor(beatLimit);
        while (address % beatBytes != 0)
            beatBytes >>= 1;

        const auto beatCount = std::min(
            {config_.burst_beats,
                remaining / beatBytes,
                bytesTo4k / beatBytes,
                std::size_t {256}});
        if (beatBytes == 0 || beatCount == 0)
            throw std::logic_error(
                "instruction-fetch could not form a legal AXI burst");
        pending_ar_ = Burst {address, beatBytes, beatCount, 0};
    }

    void accept_ar()
    {
        if (!pending_ar_.has_value() || active_burst_.has_value())
            throw std::logic_error(
                "instruction-fetch observed an unexpected AR handshake");
        active_burst_ = pending_ar_;
        pending_ar_.reset();
        bytes_requested_ +=
            active_burst_->beat_bytes * active_burst_->beat_count;
        trace_.bursts.push_back(InstructionFetchBurstTrace {
            cycle_, active_burst_->address, active_burst_->beat_bytes,
            active_burst_->beat_count});
        ++statistics_.ar_handshakes;
    }

    void accept_r_beat(const axi::Axi4ReadOutputs& axi_outputs)
    {
        if (!active_burst_.has_value())
            throw std::logic_error(
                "instruction-fetch received R data without an active burst");
        if (axi_outputs.rid != config_.axi_id)
            throw std::logic_error(
                "instruction-fetch received an unexpected AXI ID");
        if (axi_outputs.rresp != axi::AxiResponse::Okay)
            throw std::runtime_error(
                "instruction-fetch received an AXI read error response");
        const auto expectedLast =
            active_burst_->beats_received + 1
            == active_burst_->beat_count;
        if (axi_outputs.rlast != expectedLast)
            throw std::logic_error(
                "instruction-fetch received RLAST on the wrong beat");

        const auto beatAddress = active_burst_->address
            + active_burst_->beats_received * active_burst_->beat_bytes;
        const auto lane = static_cast<std::size_t>(
            beatAddress % config_.axi_data_bytes());
        if (axi_outputs.rdata.size() != config_.axi_data_bytes()
            || lane + active_burst_->beat_bytes
                > axi_outputs.rdata.size()) {
            throw std::logic_error(
                "instruction-fetch RDATA width or lane placement is invalid");
        }

        assembler_->append(std::span<const std::uint8_t>(
            axi_outputs.rdata.data() + lane,
            active_burst_->beat_bytes));
        bytes_received_ += active_burst_->beat_bytes;
        trace_.r_beat_cycles.push_back(cycle_);
        ++statistics_.r_handshakes;
        statistics_.bytes_received += active_burst_->beat_bytes;

        const auto completeWords = bytes_received_ / word_bytes_;
        while (words_assembled_ < completeWords) {
            trace_.word_assembled_cycles.push_back(cycle_);
            ++words_assembled_;
            ++statistics_.words_assembled;
        }

        ++active_burst_->beats_received;
        if (expectedLast) {
            active_burst_.reset();
            prepare_next_burst();
        }
    }

    InstructionFetchDmaConfig config_{};
    InstructionFetchProgram program_{};
    std::optional<InstructionWordAssembler> assembler_{};
    std::optional<Burst> pending_ar_{};
    std::optional<Burst> active_burst_{};
    std::size_t word_bytes_{0};
    std::size_t total_bytes_{0};
    std::size_t bytes_requested_{0};
    std::size_t bytes_received_{0};
    std::size_t words_assembled_{0};
    std::size_t words_enqueued_{0};
    std::size_t cycle_{0};
    bool started_{false};
    bool complete_{false};
    InstructionFetchDmaTrace trace_{};
    InstructionFetchDmaStatistics statistics_{};
};

} // namespace ftlpu
