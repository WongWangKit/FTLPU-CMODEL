#include "ftlpu/axi/memory_backend.hpp"
#include "ftlpu/icu/instruction_fetch_dma.hpp"
#include "ftlpu/icu/macro_bitstream_decoder.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ftlpu;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

std::vector<std::uint8_t> make_machine_code(
    std::size_t byte_count, std::uint8_t seed)
{
    auto bytes = std::vector<std::uint8_t>(byte_count, 0);
    for (std::size_t index = 0; index < byte_count; ++index) {
        bytes[index] = static_cast<std::uint8_t>(
            seed + index * 37 + (index >> 1) * 11);
    }
    return bytes;
}

bool same_word(const PhysicalInstructionWord& lhs,
    const PhysicalInstructionWord& rhs)
{
    return lhs.index == rhs.index && lhs.width == rhs.width
        && lhs.byte_count == rhs.byte_count && lhs.bytes == rhs.bytes;
}

class FiniteInstructionQueueSink {
public:
    FiniteInstructionQueueSink(std::size_t depth,
        std::size_t consume_start_cycle,
        std::size_t consume_period_cycles)
        : depth_(depth)
        , consume_start_cycle_(consume_start_cycle)
        , consume_period_cycles_(consume_period_cycles)
    {
        if (depth_ == 0)
            throw std::invalid_argument("queue sink depth must be non-zero");
    }

    bool ready(std::size_t cycle) const noexcept
    {
        return queue_.size() < depth_ || will_consume(cycle);
    }

    void tick(std::size_t cycle, bool word_valid,
        const PhysicalInstructionWord& word)
    {
        const auto consume = will_consume(cycle);
        const auto accepted = word_valid && ready(cycle);
        if (word_valid && !ready(cycle))
            ++full_wait_cycles_;
        if (consume) {
            consumed_.push_back(queue_.front());
            queue_.pop_front();
        }
        if (accepted) {
            queue_.push_back(word);
            accepted_.push_back(word);
        }
        require(queue_.size() <= depth_,
            "finite queue sink exceeded its configured depth");
        peak_occupancy_ = std::max(peak_occupancy_, queue_.size());
    }

    std::size_t depth() const noexcept { return depth_; }
    std::size_t occupancy() const noexcept { return queue_.size(); }
    std::size_t peak_occupancy() const noexcept
    {
        return peak_occupancy_;
    }
    std::size_t full_wait_cycles() const noexcept
    {
        return full_wait_cycles_;
    }
    const std::vector<PhysicalInstructionWord>& accepted() const noexcept
    {
        return accepted_;
    }

private:
    bool will_consume(std::size_t cycle) const noexcept
    {
        return !queue_.empty() && consume_period_cycles_ != 0
            && cycle >= consume_start_cycle_
            && (cycle - consume_start_cycle_) % consume_period_cycles_ == 0;
    }

    std::size_t depth_{0};
    std::size_t consume_start_cycle_{0};
    std::size_t consume_period_cycles_{0};
    std::deque<PhysicalInstructionWord> queue_{};
    std::vector<PhysicalInstructionWord> accepted_{};
    std::vector<PhysicalInstructionWord> consumed_{};
    std::size_t peak_occupancy_{0};
    std::size_t full_wait_cycles_{0};
};

struct ScenarioConfig {
    std::string name{};
    std::uint64_t base_address{0x1000};
    std::size_t word_count{1};
    PhysicalInstructionWordWidth word_width{
        PhysicalInstructionWordWidth::Bits96};
    std::size_t axi_data_width_bits{256};
    std::size_t burst_beats{4};
    std::size_t response_fifo_beats{4};
    axi::DramConfig dram{};
    std::size_t queue_depth{16};
    std::size_t consume_start_cycle{0};
    std::size_t consume_period_cycles{1};
    std::uint8_t pattern_seed{0x21};
};

struct ScenarioResult {
    std::vector<std::uint8_t> machine_code{};
    std::vector<PhysicalInstructionWord> accepted_words{};
    InstructionFetchDmaTrace dma_trace{};
    InstructionFetchDmaStatistics dma_statistics{};
    axi::Axi4ReadStatistics axi_statistics{};
    std::uint64_t backend_bytes_serviced{0};
    std::size_t assembler_capacity{0};
    std::size_t queue_peak{0};
    std::size_t queue_full_wait_cycles{0};
};

ScenarioResult run_scenario(const ScenarioConfig& config)
{
    const auto wordBytes =
        physical_instruction_word_bytes(config.word_width);
    const auto byteCount = config.word_count * wordBytes;
    const auto machineCode =
        make_machine_code(byteCount, config.pattern_seed);

    auto dram = axi::SimpleDramBackend(config.dram);
    dram.initialize_bytes(config.base_address, machineCode);
    const auto padding = std::vector<std::uint8_t>(64, 0xe7);
    dram.initialize_bytes(config.base_address + byteCount, padding);

    auto axiConfig = axi::Axi4Config {};
    axiConfig.data_width_bits = config.axi_data_width_bits;
    axiConfig.max_burst_beats = config.burst_beats;
    axiConfig.max_read_outstanding = 1;
    axiConfig.read_response_fifo_beats = config.response_fifo_beats;
    auto slave = axi::Axi4MemorySlave(axiConfig, dram);

    auto dmaConfig = InstructionFetchDmaConfig {};
    dmaConfig.axi_data_width_bits = config.axi_data_width_bits;
    dmaConfig.burst_beats = config.burst_beats;
    dmaConfig.max_outstanding = 1;
    dmaConfig.axi_id = 3;
    auto dma = InstructionFetchDma(dmaConfig);
    dma.start(InstructionFetchProgram {
        config.base_address, config.word_count, config.word_width});

    auto sink = FiniteInstructionQueueSink(config.queue_depth,
        config.consume_start_cycle, config.consume_period_cycles);
    auto heldWord = std::optional<PhysicalInstructionWord> {};

    constexpr auto kMaximumCycles = std::size_t {100'000};
    for (std::size_t cycle = 0;
         cycle < kMaximumCycles && !dma.complete(); ++cycle) {
        const auto axiOutputs = slave.evaluate();
        const auto sinkReady = sink.ready(cycle);
        const auto dmaCycle = dma.tick(axiOutputs, sinkReady);
        const auto axiCycle = slave.tick(dmaCycle.outputs.axi);

        require(dmaCycle.cycle == cycle && axiCycle.cycle == cycle,
            "DMA, AXI, and harness cycle counters diverged");
        require(dmaCycle.ar_fire == axiCycle.ar_fire
                && dmaCycle.r_fire == axiCycle.r_fire,
            "DMA and AXI disagreed about a channel handshake");

        if (dmaCycle.outputs.word_valid && !sinkReady) {
            if (!heldWord.has_value())
                heldWord = dmaCycle.outputs.word;
            else
                require(same_word(*heldWord, dmaCycle.outputs.word),
                    "word output changed while queue ready was low");
        }
        if (dmaCycle.word_fire)
            heldWord.reset();

        sink.tick(cycle, dmaCycle.outputs.word_valid,
            dmaCycle.outputs.word);
    }

    require(dma.complete(), config.name + " did not complete");
    require(sink.accepted().size() == config.word_count,
        config.name + " delivered an incorrect word count");
    require(dma.statistics().bytes_received == byteCount
            && dram.bytes_serviced() == byteCount,
        config.name + " read padding or missed program bytes");
    require(dma.statistics().words_assembled == config.word_count
            && dma.statistics().words_enqueued == config.word_count,
        config.name + " word counters are incorrect");
    require(slave.read_outstanding() == 0
            && slave.response_fifo_occupancy() == 0 && dram.idle(),
        config.name + " left an AXI or DRAM request active");
    require(dma.statistics().assembler_peak_bytes
            <= dma.assembler_capacity(),
        config.name + " exceeded the bounded assembler capacity");

    for (std::size_t word = 0; word < config.word_count; ++word) {
        const auto& actual = sink.accepted()[word];
        require(actual.index == word && actual.width == config.word_width
                && actual.byte_count == wordBytes,
            config.name + " changed physical-word metadata");
        for (std::size_t byte = 0; byte < wordBytes; ++byte) {
            require(actual.bytes[byte]
                    == machineCode[word * wordBytes + byte],
                config.name + " lost, duplicated, or reordered a byte");
        }
        for (std::size_t byte = wordBytes;
             byte < actual.bytes.size(); ++byte) {
            require(actual.bytes[byte] == 0,
                config.name + " exposed tail padding as instruction data");
        }
    }

    for (const auto& burst : dma.trace().bursts) {
        const auto burstBytes = burst.beat_bytes * burst.beat_count;
        require(burst.beat_count <= config.burst_beats
                && burst.address % burst.beat_bytes == 0,
            config.name + " emitted an illegal AXI burst shape");
        require((burst.address >> 12)
                == ((burst.address + burstBytes - 1) >> 12),
            config.name + " emitted a burst crossing 4 KiB");
    }

    require(dma.trace().r_beat_cycles.size()
                == dma.statistics().r_handshakes
            && dma.trace().word_assembled_cycles.size()
                == config.word_count
            && dma.trace().word_enqueue_cycles.size()
                == config.word_count,
        config.name + " trace is incomplete");

    std::cout << config.name
              << " ar=" << dma.statistics().ar_handshakes
              << " r=" << dma.statistics().r_handshakes
              << " assembled=" << dma.statistics().words_assembled
              << " enqueued=" << dma.statistics().words_enqueued
              << " queue_wait="
              << dma.statistics().queue_backpressure_cycles
              << " r_wait="
              << dma.statistics().axi_r_backpressure_cycles
              << " assembler_peak="
              << dma.statistics().assembler_peak_bytes
              << " queue_peak=" << sink.peak_occupancy()
              << " cycles=" << dma.statistics().cycle_count
              << " payload_bytes_per_cycle="
              << dma.statistics().effective_payload_bytes_per_cycle()
              << '\n';

    return ScenarioResult {
        machineCode,
        sink.accepted(),
        dma.trace(),
        dma.statistics(),
        slave.statistics(),
        dram.bytes_serviced(),
        dma.assembler_capacity(),
        sink.peak_occupancy(),
        sink.full_wait_cycles(),
    };
}

void test_96_bit_contiguous_and_machine_word_abi()
{
    auto config = ScenarioConfig {};
    config.name = "fetch96";
    config.word_count = 11;
    config.word_width = PhysicalInstructionWordWidth::Bits96;
    config.dram.read_latency_cycles = 2;
    config.dram.latency_jitter_cycles = 0;
    config.dram.bandwidth_bytes_per_cycle = 64;
    const auto result = run_scenario(config);

    const auto& word = result.accepted_words.front();
    require(word.lane32(0)
            == (static_cast<std::uint32_t>(result.machine_code[0])
                | (static_cast<std::uint32_t>(result.machine_code[1]) << 8)
                | (static_cast<std::uint32_t>(result.machine_code[2]) << 16)
                | (static_cast<std::uint32_t>(result.machine_code[3]) << 24)),
        "physical-word lane 0 is not little-endian bits [31:0]");
    const auto raw = IcuRawImemWord96 {{
        word.lane32(0), word.lane32(1), word.lane32(2)}};
    require(raw.bit(0) == ((word.bytes[0] & 1u) != 0)
            && raw.bit(31)
                == (((word.bytes[3] >> 7) & 1u) != 0),
        "physical-word byte ABI disagrees with IcuRawImemWord bit order");
}

void test_arvalid_holds_until_arready()
{
    auto dmaConfig = InstructionFetchDmaConfig {};
    dmaConfig.axi_data_width_bits = 256;
    dmaConfig.burst_beats = 4;
    dmaConfig.max_outstanding = 1;
    dmaConfig.axi_id = 6;
    auto dma = InstructionFetchDma(dmaConfig);
    dma.start(InstructionFetchProgram {
        0x1200, 10, PhysicalInstructionWordWidth::Bits96});

    auto slaveOutputs = axi::Axi4ReadOutputs {};
    auto held = std::optional<axi::Axi4ReadInputs> {};
    for (std::size_t cycle = 0; cycle < 3; ++cycle) {
        const auto result = dma.tick(slaveOutputs, true);
        require(result.outputs.axi.arvalid && !result.ar_fire,
            "ARVALID did not remain asserted while ARREADY was low");
        if (!held.has_value())
            held = result.outputs.axi;
        else
            require(result.outputs.axi.araddr == held->araddr
                    && result.outputs.axi.arlen == held->arlen
                    && result.outputs.axi.arsize == held->arsize
                    && result.outputs.axi.arburst == held->arburst
                    && result.outputs.axi.arid == held->arid,
                "AXI AR fields changed while ARREADY was low");
    }

    slaveOutputs.arready = true;
    const auto accepted = dma.tick(slaveOutputs, true);
    require(accepted.ar_fire && dma.trace().bursts.size() == 1
            && dma.statistics().ar_handshakes == 1,
        "DMA did not commit the held AR request when ARREADY rose");
}

void test_128_bit_contiguous()
{
    auto config = ScenarioConfig {};
    config.name = "fetch128";
    config.base_address = 0x2400;
    config.word_count = 9;
    config.word_width = PhysicalInstructionWordWidth::Bits128;
    config.dram.read_latency_cycles = 1;
    config.dram.latency_jitter_cycles = 0;
    config.dram.bandwidth_bytes_per_cycle = 32;
    (void)run_scenario(config);
}

void test_word_crosses_axi_beat()
{
    auto config = ScenarioConfig {};
    config.name = "cross_beat_96";
    config.base_address = 0x3100;
    config.word_count = 7;
    config.word_width = PhysicalInstructionWordWidth::Bits96;
    config.axi_data_width_bits = 64;
    config.burst_beats = 8;
    config.dram.read_latency_cycles = 0;
    config.dram.latency_jitter_cycles = 0;
    config.dram.bandwidth_bytes_per_cycle = 16;
    const auto result = run_scenario(config);
    require(result.dma_statistics.r_handshakes > config.word_count,
        "cross-beat test did not split words across AXI beats");
}

void test_word_crosses_axi_burst()
{
    auto config = ScenarioConfig {};
    config.name = "cross_burst_96";
    config.base_address = 0x4000;
    config.word_count = 8;
    config.word_width = PhysicalInstructionWordWidth::Bits96;
    config.burst_beats = 1;
    config.dram.read_latency_cycles = 1;
    config.dram.latency_jitter_cycles = 0;
    config.dram.bandwidth_bytes_per_cycle = 64;
    const auto result = run_scenario(config);
    require(result.dma_trace.bursts.size() >= 3,
        "cross-burst test did not create multiple AXI bursts");
}

void test_partial_final_axi_beat_uses_exact_narrow_reads()
{
    auto config = ScenarioConfig {};
    config.name = "partial_tail_96";
    config.base_address = 0x5000;
    config.word_count = 5; // 60 bytes on a 32-byte AXI bus.
    config.word_width = PhysicalInstructionWordWidth::Bits96;
    config.burst_beats = 4;
    config.dram.read_latency_cycles = 0;
    config.dram.latency_jitter_cycles = 0;
    config.dram.bandwidth_bytes_per_cycle = 64;
    const auto result = run_scenario(config);
    require(std::any_of(result.dma_trace.bursts.begin(),
                result.dma_trace.bursts.end(),
                [](const auto& burst) { return burst.beat_bytes < 32; }),
        "partial-tail test did not use a narrow final AXI read");
    require(result.backend_bytes_serviced == 60,
        "partial-tail test fetched padding bytes");
}

void test_four_kilobyte_boundary_split()
{
    auto config = ScenarioConfig {};
    config.name = "boundary_4k";
    config.base_address = 0x0ff0;
    config.word_count = 4; // 48 bytes, crossing the page boundary.
    config.word_width = PhysicalInstructionWordWidth::Bits96;
    config.burst_beats = 8;
    config.dram.read_latency_cycles = 0;
    config.dram.latency_jitter_cycles = 0;
    config.dram.bandwidth_bytes_per_cycle = 64;
    const auto result = run_scenario(config);
    require(result.dma_trace.bursts.size() >= 2
            && result.dma_trace.bursts.front().address == 0x0ff0,
        "4-KiB test did not split the read at the boundary");
}

void test_queue_full_backpressure_and_recovery()
{
    auto config = ScenarioConfig {};
    config.name = "queue_full_recovery";
    config.base_address = 0x6000;
    config.word_count = 48;
    config.word_width = PhysicalInstructionWordWidth::Bits96;
    config.burst_beats = 16;
    config.response_fifo_beats = 4;
    config.queue_depth = 16;
    config.consume_start_cycle = 100;
    config.consume_period_cycles = 1;
    config.dram.read_latency_cycles = 0;
    config.dram.latency_jitter_cycles = 0;
    config.dram.bandwidth_bytes_per_cycle = 256;
    const auto result = run_scenario(config);
    require(result.queue_peak == config.queue_depth
            && result.queue_full_wait_cycles > 0
            && result.dma_statistics.queue_backpressure_cycles > 0,
        "queue-full test did not apply sink backpressure");
    require(result.dma_statistics.axi_r_backpressure_cycles > 0
            && result.axi_statistics.r_stall_cycles > 0,
        "queue-full pressure did not propagate to AXI RREADY");
    require(result.dma_statistics.assembler_peak_bytes
            <= result.assembler_capacity,
        "queue-full recovery exceeded the elastic assembler");
}

void test_dram_latency_jitter_and_bandwidth_variation()
{
    auto config = ScenarioConfig {};
    config.name = "dram_timing_variation";
    config.base_address = 0x8000;
    config.word_count = 13;
    config.word_width = PhysicalInstructionWordWidth::Bits128;
    config.burst_beats = 3;
    config.dram.read_latency_cycles = 7;
    config.dram.latency_jitter_cycles = 9;
    config.dram.bandwidth_bytes_per_cycle = 5;
    config.dram.random_seed = 0x12345678;
    const auto result = run_scenario(config);
    require(result.dma_statistics.cycle_count
            > result.dma_statistics.r_handshakes,
        "DRAM timing variation did not introduce modeled wait cycles");
}

} // namespace

int main()
{
    test_96_bit_contiguous_and_machine_word_abi();
    test_arvalid_holds_until_arready();
    test_128_bit_contiguous();
    test_word_crosses_axi_beat();
    test_word_crosses_axi_burst();
    test_partial_final_axi_beat_uses_exact_narrow_reads();
    test_four_kilobyte_boundary_split();
    test_queue_full_backpressure_and_recovery();
    test_dram_latency_jitter_and_bandwidth_variation();
    return 0;
}
