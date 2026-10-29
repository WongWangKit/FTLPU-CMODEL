#include "ftlpu/system/c2c_dma_system.hpp"

#include <iostream>
#include <memory>
#include <sstream>

using namespace ftlpu;

namespace {
void require(bool value, const char* message)
{ if (!value) throw std::runtime_error(message); }
template <typename F> void rejects(F action)
{
    try { action(); } catch (const std::exception&) { return; }
    throw std::runtime_error("invalid program load was accepted");
}

Ddr4Config ddr_config()
{
    Ddr4Config config;
    config.read_latency_cycles = 2;
    config.read_latency_jitter_cycles = 0;
    config.peak_bandwidth_bytes_per_second = config.lpu_clock_hz * 32;
    return config;
}

std::vector<std::uint8_t> image_bytes(bool mem, std::size_t words)
{
    // Independent golden physical fixtures, also used by raw decoder tests.
    const std::vector<std::uint32_t> lanes = mem
        ? std::vector<std::uint32_t>{0x27,0,0,0x02000021,0x180,0x00303200,0x400,0,0}
        : std::vector<std::uint32_t>{0x17,0,0,0,0x28,0x04000008,0,0};
    std::vector<std::uint8_t> bytes(words * (mem ? 12 : 16));
    for (std::size_t b = 0; b < bytes.size(); ++b)
        bytes[b] = b < lanes.size() * 4
            ? static_cast<std::uint8_t>(lanes[b / 4] >> (8 * (b % 4)))
            : static_cast<std::uint8_t>(b * 37 + 11);
    return bytes;
}

void stage(C2cDmaSystem& system, std::uint64_t address,
    const std::vector<std::uint8_t>& bytes)
{
    for (std::size_t offset = 0; offset < bytes.size(); offset += 32) {
        C2cVector vector{};
        for (std::size_t b = 0; b < 32; ++b)
            vector.payload[b / hw::kLanesPerTile][b % hw::kLanesPerTile] =
                offset + b < bytes.size() ? bytes[offset + b] : 0xa5;
        system.ddr4().initialize_vector(address + offset, vector);
    }
}

void wait_ready(C2cDmaSystem& system)
{
    for (std::size_t timeout = 0; !system.program_loader().ready_to_start(); ++timeout) {
        require(timeout < 20000, "program load did not finish");
        system.tick();
    }
}

void compare(C2cDmaSystem& system, const ProgramImage& image,
    const std::vector<std::uint8_t>& bytes)
{
    for (std::size_t b = 0; b < bytes.size(); ++b) {
        const auto lane = (b % image.word_bytes()) / 4;
        const auto shift = (b % 4) * 8;
        const auto word = b / image.word_bytes();
        const auto value = image.kind == IcuMacroQueueKind::Mem
            ? system.chip().icu().mem_iq(image.queue_index).read_raw_imem(word).lanes[lane]
            : system.chip().icu().mxm_load_iq(image.queue_index).read_raw_imem(word).lanes[lane];
        require(static_cast<std::uint8_t>(value >> shift) == bytes[b], "i-MEM byte mismatch");
    }
}

ProgramLoadStatistics sweep(ProgramLoadConfig config, std::size_t depth = 16,
    bool weight_reads = false)
{
    auto system = std::make_unique<C2cDmaSystem>(ddr_config(), depth,
        SystemHardwareConfiguration{}, config);
    auto& loader = system->program_loader();
    const auto bytes = image_bytes(true, 160);
    stage(*system, 3, bytes); // Deliberately unaligned DDR source.
    loader.submit({{3, IcuMacroQueueKind::Mem, 0, bytes.size()}}, system->cycle());
    C2cDmaEngine weights(system->ddr4(), 16);
    if (weight_reads) weights.issue(C2cDmaInstruction::Load(0x10000, 120));
    loader.start(); // Early start must stay held until every gate is satisfied.
    for (std::size_t timeout = 0; !loader.ready_to_start(); ++timeout) {
        require(timeout < 20000, "program load did not finish");
        const auto words = loader.statistics().written_words;
        if (weight_reads) {
            weights.tick();
            while (weights.receive_ready()) weights.pop_received();
        }
        system->tick();
        require(loader.statistics().written_words <= words + 1,
            "target committed more than one word per cycle");
        if (!loader.statistics().images[0].first_write_committed)
            rejects([&] { system->chip().icu().mem_iq(0).read_raw_imem(0); });
    }
    require(!loader.statistics().program_start, "start bypassed dependency gate");
    compare(*system, loader.statistics().images[0].image, bytes);
    const auto stats = loader.statistics();
    if (weight_reads)
        require(system->ddr4().read_bytes_transferred() > stats.dma_bytes
            && system->ddr4().read_bytes_transferred() <= system->cycle() * 32,
            "shared DDR byte budget was not enforced across DMA clients");
    require(*stats.images[0].first_write_committed - *stats.images[0].first_write_accepted
            == config.imem_write_commit_latency_cycles
        && *stats.images[0].image_committed - *stats.images[0].last_write_accepted
            == config.imem_write_commit_latency_cycles,
        "i-MEM content or completion became visible outside its commit cycle");
    require(stats.peak_reassembly_bytes <= 64, "reassembly overflow");
    require(stats.peak_dma_vectors <= depth, "DMA FIFO/credit overflow");
    require(*stats.images[0].source_done <= *stats.images[0].image_committed,
        "source_done occurred after image_committed");
    require(*stats.frontend_ready > *stats.all_images_committed, "prefetch did not consume time");
    require(stats.effective_bytes == bytes.size() && stats.dma_bytes == 1920
        && stats.written_words == 160, "incorrect physical transfer accounting");
    loader.set_dependencies_ready(true);
    const auto start = system->cycle();
    system->tick();
    require(loader.started() && loader.statistics().program_start == start, "unified start failed");
    return stats;
}
}

int main() try
{
    ProgramLoadConfig config;
    config.bus_bytes_per_cycle = 32;
    config.route_latency_cycles = config.imem_write_ii_cycles
        = config.imem_write_commit_latency_cycles = 1;
    const auto baseline = sweep(config);
    const auto configured = sweep(ProgramLoadConfig{}, hw::config::kDmaFifoDepthVectors);
    std::cout << "configured_load_cycles="
        << *configured.all_images_committed - *configured.request_accepted
        << " bus_bytes_per_cycle=" << ProgramLoadConfig{}.bus_bytes_per_cycle
        << " dma_fifo_depth_vectors=" << hw::config::kDmaFifoDepthVectors << '\n';
    config.bus_bytes_per_cycle = 4;
    const auto narrow = sweep(config);
    require(*narrow.all_images_committed > *baseline.all_images_committed
        && narrow.bus_wait_cycles > 0, "bus serialization has no timing effect");
    config.bus_bytes_per_cycle = 32;
    config.imem_write_ii_cycles = 4;
    const auto slow_write = sweep(config, 1);
    require(*slow_write.all_images_committed > *baseline.all_images_committed
        && slow_write.target_wait_cycles > 0 && slow_write.dma_credit_wait_cycles > 0,
        "write II did not backpressure DMA");
    config.imem_write_ii_cycles = 1;
    config.imem_write_commit_latency_cycles = 4;
    const auto slow_commit = sweep(config);
    require(*slow_commit.all_images_committed == *baseline.all_images_committed + 3,
        "commit latency is not pipelined");
    config.imem_write_commit_latency_cycles = 1;
    config.route_latency_cycles = 4;
    const auto slow_route = sweep(config);
    require(*slow_route.all_images_committed == *baseline.all_images_committed + 3,
        "route latency is not pipelined");
    config.route_latency_cycles = 1;
    const auto shared = sweep(config, 16, true);
    require(*shared.all_images_committed > *baseline.all_images_committed,
        "program and weight reads bypassed shared DDR bandwidth");

    auto system = std::make_unique<C2cDmaSystem>(ddr_config(), 1,
        SystemHardwareConfiguration{}, config);
    auto& loader = system->program_loader();
    const auto mem = image_bytes(true, 9), mxm = image_bytes(false, 7);
    stage(*system, 0x1000, mem); stage(*system, 0x2000, mxm);
    const ProgramImage a{0x1000, IcuMacroQueueKind::Mem, 0, mem.size()};
    const ProgramImage b{0x2000, IcuMacroQueueKind::MxmLoad, 0, mxm.size()};
    rejects([&] { auto invalid = a; invalid.byte_count = 13; loader.submit({invalid},0); });
    rejects([&] { auto invalid = a; invalid.mode = IcuQueueMode::Native; loader.submit({invalid},0); });
    rejects([&] { auto invalid = a; invalid.queue_index = InstructionControlUnit::kMemQueues;
        loader.submit({invalid},0); });
    rejects([&] { loader.submit({a,a},0); });
    require(system->cycle() == 0, "validation advanced physical time");
    loader.submit({a,b}, system->cycle());
    rejects([&] { loader.submit({a}, system->cycle()); });
    rejects([&] { system->chip().icu().mem_iq(0).read_raw_imem(0); });
    while (!loader.ready_to_start()) {
        system->tick();
        if (loader.statistics().images[0].image_committed && !loader.all_images_committed())
            require(system->chip().icu().mem_iq(0).macro_decoder_statistics().fetched_words == 0,
                "a frontend fetched before all queue images committed");
        require(system->cycle() < 20000, "multi-queue load timed out");
    }
    const auto previous = loader.statistics();
    require(*previous.images[1].first_ddr_request > *previous.images[0].image_committed,
        "queue switch preceded the last commit");
    compare(*system, a, mem); compare(*system, b, mxm);
    loader.start(); loader.set_dependencies_ready(true); system->tick();
    const auto nextCycle = system->cycle();
    const auto shortMem = image_bytes(true, 3);
    stage(*system, 0x3000, shortMem);
    loader.submit({{0x3000,IcuMacroQueueKind::Mem,0,shortMem.size()}}, nextCycle);
    require(!loader.all_images_committed() && !loader.statistics().program_start,
        "old completion leaked into next program");
    rejects([&] { system->chip().icu().mem_iq(0).read_raw_imem(8); });
    wait_ready(*system);
    require(loader.statistics().request_accepted == nextCycle
        && system->ddr4().cycle() == system->cycle(), "physical time was reset on switch");
    require(loader.statistics().written_words == 3, "long image leaked into short program");
    std::ostringstream report; loader.statistics().print(report);
    require(report.str().find("source_done=") != std::string::npos, "missing load report");

    for (int field = 0; field < 4; ++field) {
        auto invalid = config;
        if (field == 0) invalid.bus_bytes_per_cycle = 0;
        if (field == 1) invalid.route_latency_cycles = 0;
        if (field == 2) invalid.imem_write_ii_cycles = 0;
        if (field == 3) invalid.imem_write_commit_latency_cycles = 0;
        rejects([&] { invalid.validate(); });
    }
    std::cout << "program_loader_test passed\n";
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "program_loader_test failed: " << ex.what() << '\n'; return 1;
}
