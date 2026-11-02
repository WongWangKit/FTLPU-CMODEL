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
void barrier_storage_and_activation() {
    auto system=std::make_unique<C2cDmaSystem>(ddr_config(),1);
    auto encoded=std::make_unique<InstructionControlUnit>();encoded->mem_iq(0).append_control(IcuControlInstruction::Nop(8));
    ProgramImage native{0x1000,IcuMacroQueueKind::Mem,0,12,IcuQueueMode::Native};
    ProgramImage barr;barr.source_address=0x2000;barr.storage=ProgramStorage::Barr;barr.byte_count=8;barr.generation=20;
    ProgramImage events;events.source_address=0x3000;events.storage=ProgramStorage::BarrierEvents;events.byte_count=12;events.generation=20;
    const std::vector<std::uint8_t> table{1,0,0,0,5,0,0,0},bindings{5,0,0,0,1,0,0,0,42,0,0,0};
    stage(*system,native.source_address,encoded->read_program_image(native));stage(*system,barr.source_address,table);stage(*system,events.source_address,bindings);
    auto& loader=system->program_loader();loader.submit({native,barr,events},0);loader.start();loader.set_dependencies_ready(true);
    while(!loader.ready_to_start()) {
        system->tick();require(system->cycle()<2000,"BARR prime timed out");
        const auto& stats=loader.statistics();
        require(system->chip().icu().mem_iq(0).cycle()==0,"BARR program ran before control commit");
        if(stats.images[2].first_write_accepted && !stats.images[2].first_write_committed)
            require(system->chip().icu().read_program_image(events).empty(),"event RAM changed before commit");
    }
    require(system->chip().icu().read_program_image(barr)==table && system->chip().icu().read_program_image(events)==bindings,"BARR/config RAM mismatch");
    system->tick();system->tick();
    require(system->chip().icu().common_barrier_holding() && !loader.kernel_done(),"BARR did not wait for committed binding");
    system->chip().icu().notify_common_mem_commit(19,42);system->tick();
    require(system->chip().icu().common_barrier_holding(),"old generation released BARR");
    system->chip().icu().notify_common_mem_commit(20,42);
    for(std::size_t n=0;!loader.kernel_done();++n) {require(n<100,"BARR kernel did not complete");system->tick();}
    require(loader.statistics().stores[static_cast<std::size_t>(ProgramStorage::Barr)].written_words==1,"missing per-storage accounting");
    auto bad=native;bad.storage=static_cast<ProgramStorage>(99);rejects([&]{loader.submit({bad},system->cycle());});
    bad=native;bad.byte_count=12*(hw::kIcuMemImemDepth+1);rejects([&]{loader.submit({bad},system->cycle());});
    require(loader.can_load_program(),"invalid image changed completed device");
    rejects([&]{loader.submit({native,barr,events},system->cycle());});
    barr.generation=events.generation=21;barr.byte_count=events.byte_count=0;
    loader.submit({barr,events},system->cycle());
    system->chip().icu().notify_common_mem_commit(20,42); // Before new event RAM commits.
    wait_ready(*system);loader.start();loader.set_dependencies_ready(true);system->tick();
    require(loader.kernel_done(),"empty control task retained old generation");

}

}

int main() try
{
    barrier_storage_and_activation();
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
    rejects([&] { auto invalid = a; invalid.mode = static_cast<IcuQueueMode>(3); loader.submit({invalid},0); });
    rejects([&] { auto invalid = a; invalid.queue_index = InstructionControlUnit::kMemQueues;
        loader.submit({invalid},0); });
    rejects([&] { loader.submit({a,a},0); });
    ProgramImage qctl;qctl.storage=ProgramStorage::Qctl;qctl.generation=1;
    rejects([&]{loader.submit({a,qctl},0);});
    auto mixed=a;mixed.storage=ProgramStorage::Barr;mixed.byte_count=0;mixed.generation=1;
    rejects([&]{loader.submit({a,qctl,mixed},0);});
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
    // A fresh, executable Native program exercises safe long/short switching.
    system=std::make_unique<C2cDmaSystem>(ddr_config(),1,SystemHardwareConfiguration{},config);
    auto& safe=system->program_loader();
    auto native=std::make_unique<InstructionControlUnit>();
    for(unsigned i=0;i<5;++i) native->mem_iq(0).append_control(IcuControlInstruction::Nop(4));
    ProgramImage longImage{0x3000,IcuMacroQueueKind::Mem,0,60,IcuQueueMode::Native};
    const auto nativeBytes=native->read_program_image(longImage);
    stage(*system,longImage.source_address,nativeBytes);safe.submit({longImage},system->cycle());
    wait_ready(*system);safe.start();safe.set_dependencies_ready(true);system->tick();
    const auto accepted=safe.statistics().request_accepted;
    rejects([&]{safe.submit({longImage},system->cycle());});
    require(safe.statistics().request_accepted==accepted && safe.started(),"unsafe resubmit changed old task");
    for(std::size_t n=0;!safe.kernel_done();++n) {require(n<100,"Native kernel failed to drain");system->tick();}
    require(!safe.busy() && safe.can_load_program(),"completed kernel cannot switch");
    const auto done=safe.statistics().kernel_done_cycle;
    // Returned DMA vectors and unconsumed DDR responses remain live resources.
    auto& data=system->dma(Hemisphere::East);data.issue(C2cDmaInstruction::Load(0x8000,1));
    while(!data.idle()) system->tick();
    require(!safe.can_load_program() && safe.kernel_done(),"DMA FIFO was excluded from drain");
    rejects([&]{safe.submit({},system->cycle());});data.pop_received();
    const auto request=system->ddr4().request_read(0x9000);
    while(!system->ddr4().idle()) system->tick();
    require(!safe.can_load_program(),"DDR response was excluded from drain");
    system->ddr4().pop_read_completion(request);
    require(safe.can_load_program() && safe.statistics().kernel_done_cycle==done,"completion latch changed");
    const auto nextCycle=system->cycle();
    auto shortImage=longImage;shortImage.source_address=0x4000;shortImage.byte_count=12;
    stage(*system,shortImage.source_address,std::vector<std::uint8_t>(nativeBytes.begin(),nativeBytes.begin()+12));
    safe.submit({shortImage},nextCycle);wait_ready(*system);
    require(safe.statistics().request_accepted==nextCycle && system->ddr4().cycle()==system->cycle(),
        "program switch reset physical time");
    rejects([&]{system->chip().icu().mem_iq(0).read_committed_imem(4);});
    std::ostringstream report; safe.statistics().print(report);
    require(report.str().find("kernel_done=") != std::string::npos, "missing load report");

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
