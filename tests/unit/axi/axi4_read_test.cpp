#include "ftlpu/axi/axi4_memory_slave.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace ftlpu::axi;

static_assert(std::is_same_v<MemoryConfig, DramConfig>);
static_assert(std::is_same_v<ByteMemoryBackend, SimpleDramBackend>);

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

DramConfig dram_config(std::size_t latency,
    std::size_t jitter, std::size_t bandwidth,
    std::size_t queue_depth = 64)
{
    auto config = DramConfig {};
    config.read_latency_cycles = latency;
    config.latency_jitter_cycles = jitter;
    config.bandwidth_bytes_per_cycle = bandwidth;
    config.request_queue_depth = queue_depth;
    config.random_seed = 1;
    return config;
}

Axi4Config axi_config(std::size_t outstanding = 1,
    std::size_t response_fifo_beats = 16)
{
    auto config = Axi4Config {};
    config.max_read_outstanding = outstanding;
    config.read_response_fifo_beats = response_fifo_beats;
    return config;
}

Axi4ReadInputs read_request(std::uint64_t address,
    std::size_t beats, std::uint8_t size,
    std::uint32_t id = 0)
{
    auto input = Axi4ReadInputs {};
    input.arvalid = true;
    input.araddr = address;
    input.arlen = static_cast<std::uint8_t>(beats - 1);
    input.arsize = size;
    input.arburst = AxiBurst::Incrementing;
    input.arid = id;
    input.rready = true;
    return input;
}

std::vector<std::uint8_t> make_pattern(
    std::size_t bytes, std::uint8_t base)
{
    auto data = std::vector<std::uint8_t>(bytes, 0);
    for (std::size_t index = 0; index < data.size(); ++index)
        data[index] = static_cast<std::uint8_t>(base + index);
    return data;
}

void test_single_burst_and_trace()
{
    constexpr auto kAddress = std::uint64_t {0x1000};
    constexpr auto kBeatBytes = std::size_t {32};
    constexpr auto kBeats = std::size_t {4};

    auto backend = SimpleDramBackend(dram_config(2, 0, 32));
    const auto expected = make_pattern(kBeatBytes * kBeats, 0x20);
    backend.initialize_bytes(kAddress, expected);
    auto slave = Axi4MemorySlave(axi_config(), backend);
    auto input = read_request(kAddress, kBeats, 5, 5);

    std::size_t received = 0;
    bool completed = false;
    for (std::size_t elapsed = 0; elapsed < 64; ++elapsed) {
        const auto cycle = slave.tick(input);
        if (cycle.ar_fire)
            input.arvalid = false;
        if (!cycle.r_fire)
            continue;

        require(cycle.outputs.rid == 5
                && cycle.outputs.rresp == AxiResponse::Okay,
            "single burst returned an incorrect RID or RRESP");
        require(cycle.outputs.rdata.size() == kBeatBytes,
            "single burst returned an incorrect RDATA width");
        for (std::size_t byte = 0; byte < kBeatBytes; ++byte)
            require(cycle.outputs.rdata[byte]
                    == expected[received * kBeatBytes + byte],
                "single burst changed payload data");
        require(cycle.outputs.rlast == (received + 1 == kBeats),
            "single burst asserted RLAST on the wrong beat");
        ++received;
        if (cycle.outputs.rlast) {
            completed = true;
            break;
        }
    }

    require(completed && received == kBeats,
        "single burst did not return every beat");
    require(slave.read_outstanding() == 0,
        "single burst did not release outstanding on final R handshake");
    require(slave.traces().size() == 1,
        "single burst did not create exactly one transaction trace");
    const auto& trace = slave.traces().front();
    require(trace.ar_accept_cycle == 0
            && trace.first_rvalid_cycle.has_value()
            && trace.first_r_handshake_cycle
                == trace.first_rvalid_cycle
            && trace.last_r_handshake_cycle.has_value()
            && trace.byte_count == kBeatBytes * kBeats,
        "single burst trace is incomplete");
    const auto& stats = slave.statistics();
    require(stats.transferred_bytes == kBeatBytes * kBeats
            && stats.outstanding_peak == 1
            && stats.r_stall_cycles == 0,
        "single burst statistics are incorrect");

    std::cout << "single_burst ar=" << trace.ar_accept_cycle
              << " first_valid=" << *trace.first_rvalid_cycle
              << " first_fire=" << *trace.first_r_handshake_cycle
              << " last_fire=" << *trace.last_r_handshake_cycle
              << " bytes=" << trace.byte_count
              << " data_wait=" << stats.data_wait_cycles
              << " bytes_per_cycle="
              << stats.effective_bytes_per_cycle() << '\n';
}

void test_narrow_burst_uses_arsize_and_bus_lanes()
{
    constexpr auto kAddress = std::uint64_t {0x2080};
    constexpr auto kBusBytes = std::size_t {32};
    constexpr auto kBeatBytes = std::size_t {8};
    constexpr auto kBeats = std::size_t {4};

    auto backend = SimpleDramBackend(dram_config(0, 0, 32));
    const auto expected = make_pattern(kBeatBytes * kBeats, 0x70);
    backend.initialize_bytes(kAddress, expected);
    auto slave = Axi4MemorySlave(axi_config(), backend);
    auto input = read_request(kAddress, kBeats, 3, 1);

    std::size_t received = 0;
    for (std::size_t elapsed = 0; elapsed < 32 && received < kBeats;
         ++elapsed) {
        const auto cycle = slave.tick(input);
        if (cycle.ar_fire)
            input.arvalid = false;
        if (!cycle.r_fire)
            continue;
        const auto lane = static_cast<std::size_t>(
            (kAddress + received * kBeatBytes) % kBusBytes);
        for (std::size_t byte = 0; byte < kBusBytes; ++byte) {
            const auto expectedByte =
                byte >= lane && byte < lane + kBeatBytes
                ? expected[received * kBeatBytes + byte - lane]
                : 0;
            require(cycle.outputs.rdata[byte] == expectedByte,
                "narrow burst placed bytes in the wrong RDATA lanes");
        }
        require(cycle.outputs.rlast == (received + 1 == kBeats),
            "narrow burst asserted RLAST on the wrong beat");
        ++received;
    }
    require(received == kBeats
            && slave.statistics().transferred_bytes
                == kBeatBytes * kBeats,
        "ARSIZE did not control the narrow-burst byte count");
}

void exercise_outstanding(std::size_t outstanding_limit)
{
    constexpr auto kRequestCount = std::size_t {8};
    constexpr auto kBaseAddress = std::uint64_t {0x4000};

    auto backend = SimpleDramBackend(dram_config(12, 0, 32, 16));
    for (std::size_t request = 0; request < kRequestCount; ++request) {
        auto data = std::vector<std::uint8_t>(32,
            static_cast<std::uint8_t>(0x30 + request));
        backend.initialize_bytes(kBaseAddress + request * 64, data);
    }
    auto slave = Axi4MemorySlave(
        axi_config(outstanding_limit, 16), backend);

    std::size_t submitted = 0;
    std::size_t completed = 0;
    std::size_t previousFireCycle = 0;
    for (std::size_t elapsed = 0;
         elapsed < 512 && completed < kRequestCount; ++elapsed) {
        auto input = Axi4ReadInputs {};
        input.rready = true;
        if (submitted < kRequestCount) {
            input = read_request(
                kBaseAddress + submitted * 64, 1, 5, 2);
        }
        const auto cycle = slave.tick(input);
        if (cycle.ar_fire)
            ++submitted;
        if (!cycle.r_fire)
            continue;

        require(cycle.outputs.rid == 2 && cycle.outputs.rlast,
            "outstanding read returned an incorrect ID or RLAST");
        require(cycle.outputs.rdata.front()
                == static_cast<std::uint8_t>(0x30 + completed),
            "same-ID outstanding responses changed acceptance order");
        if (completed != 0)
            require(cycle.cycle > previousFireCycle,
                "more than one R beat completed in one cycle");
        previousFireCycle = cycle.cycle;
        ++completed;
    }

    require(submitted == kRequestCount && completed == kRequestCount,
        "outstanding test did not complete every request");
    require(slave.statistics().outstanding_peak == outstanding_limit,
        "configured outstanding limit was not reached or enforced");
    require(slave.traces().size() == kRequestCount,
        "outstanding test lost transaction traces");
    for (std::size_t index = 0; index < slave.traces().size(); ++index) {
        const auto& trace = slave.traces()[index];
        require(trace.sequence == index && trace.id == 2
                && trace.last_r_handshake_cycle.has_value(),
            "same-ID outstanding trace ordering is incorrect");
    }
}

void test_consecutive_bursts_and_outstanding_limits()
{
    for (const auto limit : {1U, 2U, 4U, 8U})
        exercise_outstanding(limit);
    std::cout << "outstanding_limits=1,2,4,8 passed\n";
}

void test_rready_backpressure_holds_and_resumes()
{
    constexpr auto kAddress = std::uint64_t {0x8000};
    constexpr auto kBeats = std::size_t {4};
    constexpr auto kStallCycles = std::size_t {2};

    auto backend = SimpleDramBackend(dram_config(0, 0, 32));
    const auto expected = make_pattern(32 * kBeats, 0xa0);
    backend.initialize_bytes(kAddress, expected);
    auto slave = Axi4MemorySlave(axi_config(1, 4), backend);
    auto input = read_request(kAddress, kBeats, 5, 7);
    input.rready = false;

    bool release = false;
    std::size_t stalls = 0;
    std::size_t received = 0;
    std::optional<Axi4ReadOutputs> held{};
    std::vector<std::size_t> fireCycles{};
    for (std::size_t elapsed = 0; elapsed < 64 && received < kBeats;
         ++elapsed) {
        input.rready = release;
        const auto cycle = slave.tick(input);
        if (cycle.ar_fire)
            input.arvalid = false;

        if (cycle.outputs.rvalid && !input.rready) {
            if (!held.has_value())
                held = cycle.outputs;
            else
                require(cycle.outputs.rdata == held->rdata
                        && cycle.outputs.rid == held->rid
                        && cycle.outputs.rlast == held->rlast
                        && cycle.outputs.rresp == held->rresp,
                    "R channel fields changed while RREADY was low");
            ++stalls;
            if (stalls == kStallCycles)
                release = true;
        }

        if (!cycle.r_fire)
            continue;
        fireCycles.push_back(cycle.cycle);
        for (std::size_t byte = 0; byte < 32; ++byte)
            require(cycle.outputs.rdata[byte]
                    == expected[received * 32 + byte],
                "backpressure recovery changed response data");
        ++received;
    }

    require(received == kBeats && stalls == kStallCycles,
        "RREADY backpressure test did not stall and drain as expected");
    require(slave.statistics().r_stall_cycles == kStallCycles,
        "R stall cycle metric is incorrect");
    require(slave.statistics().response_fifo_peak < 4,
        "short RREADY stall unexpectedly filled the response FIFO");
    require(slave.traces().front().first_r_handshake_cycle.value()
            - slave.traces().front().first_rvalid_cycle.value()
            == kStallCycles,
        "RREADY recovery behaved like a new memory latency");
    for (std::size_t index = 1; index < fireCycles.size(); ++index)
        require(fireCycles[index] == fireCycles[index - 1] + 1,
            "buffered R beats did not resume at one beat per cycle");

    std::cout << "backpressure stalls="
              << slave.statistics().r_stall_cycles
              << " response_fifo_peak="
              << slave.statistics().response_fifo_peak << '\n';
}

void exercise_full_fifo_pause_and_resume(std::size_t fifo_depth)
{
    constexpr auto kBeatBytes = std::size_t {32};
    constexpr auto kBeats = std::size_t {8};
    constexpr auto kStallCycles = std::size_t {10};
    const auto address = std::uint64_t {0x9000}
        + fifo_depth * 0x200;

    auto backend = SimpleDramBackend(dram_config(3, 0, 256));
    const auto expected = make_pattern(kBeatBytes * kBeats,
        static_cast<std::uint8_t>(0x40 + fifo_depth));
    backend.initialize_bytes(address, expected);
    auto slave = Axi4MemorySlave(axi_config(1, fifo_depth), backend);
    auto input = read_request(address, kBeats, 5, 6);
    input.rready = false;

    bool fifoFull = false;
    for (std::size_t elapsed = 0; elapsed < 32; ++elapsed) {
        const auto cycle = slave.tick(input);
        if (cycle.ar_fire)
            input.arvalid = false;
        if (cycle.response_fifo_occupancy == fifo_depth) {
            fifoFull = true;
            break;
        }
    }
    require(fifoFull, "DRAM did not fill the bounded response FIFO");
    const auto bytesAtFull = backend.bytes_serviced();
    require(bytesAtFull == fifo_depth * kBeatBytes
            && backend.queued_read_count() == 1
            && slave.read_outstanding() == 1,
        "FIFO became full after the DRAM request had already disappeared");
    require(!backend.traces().front().completion_cycle.has_value(),
        "DRAM completed a burst beyond the response FIFO capacity");

    std::optional<Axi4ReadOutputs> held{};
    for (std::size_t stall = 0; stall < kStallCycles; ++stall) {
        const auto cycle = slave.tick(input);
        require(cycle.outputs.rvalid
                && cycle.response_fifo_occupancy == fifo_depth,
            "full response FIFO did not hold RVALID and its occupancy");
        if (!held.has_value())
            held = cycle.outputs;
        else
            require(cycle.outputs.rdata == held->rdata
                    && cycle.outputs.rid == held->rid
                    && cycle.outputs.rlast == held->rlast
                    && cycle.outputs.rresp == held->rresp,
                "full FIFO changed the stalled response fields");
        require(cycle.backend_bytes_serviced == 0
                && backend.bytes_serviced() == bytesAtFull,
            "DRAM serviced bytes while the response FIFO had no credit");
        require(slave.read_outstanding() == 1
                && !backend.traces().front()
                        .completion_cycle.has_value(),
            "full FIFO retired an outstanding request during backpressure");
    }

    const auto firstResumeCycle = slave.cycle();
    input.rready = true;
    std::size_t received = 0;
    std::vector<std::size_t> fireCycles{};
    for (std::size_t elapsed = 0; elapsed < 64 && received < kBeats;
         ++elapsed) {
        const auto cycle = slave.tick(input);
        if (!cycle.r_fire)
            continue;
        fireCycles.push_back(cycle.cycle);
        for (std::size_t byte = 0; byte < kBeatBytes; ++byte)
            require(cycle.outputs.rdata[byte]
                    == expected[received * kBeatBytes + byte],
                "full-FIFO recovery lost, duplicated, or reordered data");
        require(cycle.outputs.rlast == (received + 1 == kBeats),
            "full-FIFO recovery changed the burst boundary");
        ++received;
    }

    require(received == kBeats
            && fireCycles.front() == firstResumeCycle,
        "FIFO recovery inserted a new DRAM first-read latency");
    for (std::size_t beat = 1; beat < fireCycles.size(); ++beat)
        require(fireCycles[beat] == fireCycles[beat - 1] + 1,
            "FIFO recovery did not continue at one R beat per cycle");
    require(slave.read_outstanding() == 0 && backend.idle()
            && backend.bytes_serviced() == kBeats * kBeatBytes
            && slave.statistics().response_fifo_peak == fifo_depth,
        "full-FIFO recovery did not complete the original request");
}

void test_full_fifo_pause_across_depths()
{
    for (const auto depth : {1U, 2U, 4U})
        exercise_full_fifo_pause_and_resume(depth);
    std::cout << "full_fifo_depths=1,2,4 pause_resume passed\n";
}

void test_full_fifo_with_multiple_outstanding()
{
    constexpr auto kRequests = std::size_t {4};
    constexpr auto kBeatsPerRequest = std::size_t {4};
    constexpr auto kBeatBytes = std::size_t {32};
    constexpr auto kFifoDepth = std::size_t {2};
    constexpr auto kStallCycles = std::size_t {12};
    constexpr auto kBaseAddress = std::uint64_t {0xd000};

    auto backend = SimpleDramBackend(dram_config(3, 0, 256, 8));
    auto expected = std::vector<std::vector<std::uint8_t>> {};
    for (std::size_t request = 0; request < kRequests; ++request) {
        expected.push_back(make_pattern(
            kBeatsPerRequest * kBeatBytes,
            static_cast<std::uint8_t>(request * 0x30)));
        backend.initialize_bytes(
            kBaseAddress + request * 0x100, expected.back());
    }
    auto slave = Axi4MemorySlave(
        axi_config(kRequests, kFifoDepth), backend);

    std::size_t submitted = 0;
    bool fifoFull = false;
    for (std::size_t elapsed = 0; elapsed < 32; ++elapsed) {
        auto input = Axi4ReadInputs {};
        input.rready = false;
        if (submitted < kRequests)
            input = read_request(
                kBaseAddress + submitted * 0x100,
                kBeatsPerRequest, 5, 9);
        input.rready = false;
        const auto cycle = slave.tick(input);
        if (cycle.ar_fire)
            ++submitted;
        if (submitted == kRequests
            && cycle.response_fifo_occupancy == kFifoDepth) {
            fifoFull = true;
            break;
        }
    }
    require(fifoFull && slave.read_outstanding() == kRequests
            && backend.queued_read_count() == kRequests,
        "multiple outstanding requests were not active at FIFO saturation");
    const auto bytesAtFull = backend.bytes_serviced();
    require(bytesAtFull == kFifoDepth * kBeatBytes,
        "DRAM crossed the response FIFO capacity before the long stall");

    auto stalledInput = Axi4ReadInputs {};
    stalledInput.rready = false;
    std::optional<Axi4ReadOutputs> held{};
    for (std::size_t stall = 0; stall < kStallCycles; ++stall) {
        const auto cycle = slave.tick(stalledInput);
        require(cycle.outputs.rvalid
                && cycle.response_fifo_occupancy == kFifoDepth
                && cycle.backend_bytes_serviced == 0,
            "multiple-outstanding FIFO stall leaked backend service");
        if (!held.has_value())
            held = cycle.outputs;
        else
            require(cycle.outputs.rdata == held->rdata
                    && cycle.outputs.rid == held->rid
                    && cycle.outputs.rlast == held->rlast,
                "multiple-outstanding stall changed the FIFO head");
        require(backend.bytes_serviced() == bytesAtFull
                && slave.read_outstanding() == kRequests,
            "multiple-outstanding state changed while the FIFO was full");
    }

    auto readyInput = Axi4ReadInputs {};
    readyInput.rready = true;
    const auto firstResumeCycle = slave.cycle();
    std::size_t received = 0;
    std::vector<std::size_t> fireCycles{};
    const auto totalBeats = kRequests * kBeatsPerRequest;
    for (std::size_t elapsed = 0;
         elapsed < 128 && received < totalBeats; ++elapsed) {
        const auto cycle = slave.tick(readyInput);
        if (!cycle.r_fire)
            continue;
        const auto request = received / kBeatsPerRequest;
        const auto beat = received % kBeatsPerRequest;
        require(cycle.outputs.rid == 9
                && cycle.outputs.rlast
                    == (beat + 1 == kBeatsPerRequest),
            "multiple-outstanding recovery changed RID or RLAST");
        for (std::size_t byte = 0; byte < kBeatBytes; ++byte)
            require(cycle.outputs.rdata[byte]
                    == expected[request][beat * kBeatBytes + byte],
                "multiple-outstanding recovery changed response order");
        fireCycles.push_back(cycle.cycle);
        ++received;
    }

    require(received == totalBeats
            && fireCycles.front() == firstResumeCycle,
        "multiple-outstanding recovery paid a new DRAM latency");
    for (std::size_t beat = 1; beat < fireCycles.size(); ++beat)
        require(fireCycles[beat] == fireCycles[beat - 1] + 1,
            "multiple-outstanding recovery introduced an R-channel gap");
    require(slave.read_outstanding() == 0 && backend.idle()
            && slave.statistics().outstanding_peak == kRequests
            && slave.traces().size() == kRequests,
        "multiple-outstanding recovery did not retire every transaction");
    for (const auto& trace : slave.traces())
        require(trace.last_r_handshake_cycle.has_value(),
            "multiple-outstanding trace missed final R handshake");

    std::cout << "full_fifo_multi_outstanding=" << kRequests
              << " stall_cycles=" << kStallCycles
              << " total_beats=" << totalBeats << '\n';
}

void test_backend_bandwidth_inserts_beat_gaps()
{
    constexpr auto kAddress = std::uint64_t {0xa000};
    constexpr auto kBeats = std::size_t {3};

    auto backend = SimpleDramBackend(dram_config(0, 0, 8));
    backend.initialize_bytes(
        kAddress, make_pattern(32 * kBeats, 0x10));
    auto slave = Axi4MemorySlave(axi_config(1, 4), backend);
    auto input = read_request(kAddress, kBeats, 5);

    auto fireCycles = std::vector<std::size_t> {};
    for (std::size_t elapsed = 0;
         elapsed < 64 && fireCycles.size() < kBeats; ++elapsed) {
        const auto cycle = slave.tick(input);
        if (cycle.ar_fire)
            input.arvalid = false;
        if (cycle.r_fire)
            fireCycles.push_back(cycle.cycle);
    }

    require(fireCycles.size() == kBeats,
        "bandwidth-limited backend did not complete the burst");
    for (std::size_t index = 1; index < fireCycles.size(); ++index)
        require(fireCycles[index] - fireCycles[index - 1] == 4,
            "8-byte/cycle backend did not take four cycles per 32-byte beat");
    require(slave.statistics().backend_bytes_serviced == 96
            && slave.statistics().transferred_bytes == 96,
        "backend and AXI byte counters disagree");

    std::cout << "bandwidth_limited r_fire_cycles="
              << fireCycles[0] << ',' << fireCycles[1] << ','
              << fireCycles[2] << " bytes_per_cycle="
              << slave.statistics().effective_bytes_per_cycle() << '\n';
}

struct JitterResult {
    std::vector<std::size_t> configured_latencies{};
    std::vector<std::size_t> first_valid_cycles{};
};

JitterResult run_jitter(std::uint64_t seed)
{
    constexpr auto kRequests = std::size_t {8};
    constexpr auto kBaseAddress = std::uint64_t {0xc000};

    auto config = dram_config(3, 7, 256, 16);
    config.random_seed = seed;
    auto backend = SimpleDramBackend(config);
    auto slave = Axi4MemorySlave(axi_config(8, 16), backend);

    std::size_t submitted = 0;
    std::size_t completed = 0;
    for (std::size_t elapsed = 0;
         elapsed < 256 && completed < kRequests; ++elapsed) {
        auto input = Axi4ReadInputs {};
        input.rready = true;
        if (submitted < kRequests)
            input = read_request(
                kBaseAddress + submitted * 64, 1, 5, 3);
        const auto cycle = slave.tick(input);
        if (cycle.ar_fire)
            ++submitted;
        if (cycle.r_fire)
            ++completed;
    }
    require(completed == kRequests,
        "jitter run did not complete every request");

    auto result = JitterResult {};
    for (const auto& trace : backend.traces()) {
        require(trace.initial_latency_cycles >= 3
                && trace.initial_latency_cycles <= 10,
            "backend jitter escaped its configured range");
        result.configured_latencies.push_back(
            trace.initial_latency_cycles);
    }
    for (const auto& trace : slave.traces()) {
        require(trace.first_rvalid_cycle.has_value(),
            "AXI trace missed first RVALID under jitter");
        result.first_valid_cycles.push_back(
            *trace.first_rvalid_cycle);
    }
    return result;
}

void test_latency_jitter_is_reproducible()
{
    const auto first = run_jitter(0x1234);
    const auto second = run_jitter(0x1234);
    const auto differentSeed = run_jitter(0x5678);
    require(first.configured_latencies == second.configured_latencies
            && first.first_valid_cycles == second.first_valid_cycles,
        "fixed-seed latency jitter is not reproducible");
    require(first.configured_latencies
            != differentSeed.configured_latencies,
        "changing the latency jitter seed had no effect");
}

void test_illegal_bursts_are_rejected_without_state_change()
{
    auto backend = SimpleDramBackend(dram_config(0, 0, 32));
    auto config = axi_config();
    config.max_burst_beats = 4;
    auto slave = Axi4MemorySlave(config, backend);

    const auto reject = [&slave](Axi4ReadInputs input,
                            const char* description) {
        bool rejected = false;
        try {
            (void)slave.tick(input);
        } catch (const std::invalid_argument&) {
            rejected = true;
        } catch (const std::out_of_range&) {
            rejected = true;
        }
        require(rejected, std::string("illegal AXI request accepted: ")
                + description);
    };

    reject(read_request(0x1000, 5, 5), "burst exceeds maximum");
    reject(read_request(0x1000, 1, 6), "ARSIZE exceeds bus width");
    reject(read_request(0x0fe0, 2, 5), "burst crosses 4 KB");
    auto fixed = read_request(0x1000, 1, 5);
    fixed.arburst = AxiBurst::Fixed;
    reject(fixed, "FIXED burst is unsupported");
    reject(read_request(0x1000, 1, 5, 16), "ARID exceeds width");
    reject(read_request(0x1004, 1, 5), "address is not beat-aligned");

    require(slave.cycle() == 0 && slave.read_outstanding() == 0
            && backend.traces().empty(),
        "rejected AXI requests changed model state");
}

} // namespace

int main()
try {
    test_single_burst_and_trace();
    test_narrow_burst_uses_arsize_and_bus_lanes();
    test_consecutive_bursts_and_outstanding_limits();
    test_rready_backpressure_holds_and_resumes();
    test_full_fifo_pause_across_depths();
    test_full_fifo_with_multiple_outstanding();
    test_backend_bandwidth_inserts_beat_gaps();
    test_latency_jitter_is_reproducible();
    test_illegal_bursts_are_rejected_without_state_change();
    return 0;
} catch (const std::exception& error) {
    std::cerr << "axi4_read_test failed: " << error.what() << '\n';
    return 1;
}
