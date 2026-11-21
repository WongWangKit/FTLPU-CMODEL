#include "ftlpu/c2c/dma.hpp"
#include "ftlpu/core/timing.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace {

using namespace ftlpu;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

C2cVector make_vector(std::uint8_t base)
{
    auto vector = C2cVector {};
    for (std::size_t byte = 0; byte < hw::kPhysicalVectorBytes; ++byte) {
        vector.payload[byte / hw::kLanesPerTile]
            [byte % hw::kLanesPerTile] =
            static_cast<std::uint8_t>(base + byte);
    }
    return vector;
}

Ddr4Config zero_latency_ddr()
{
    auto config = Ddr4Config {};
    config.beat_bytes = hw::kPhysicalVectorBytes;
    config.read_latency_cycles = 0;
    config.write_latency_cycles = 0;
    config.read_latency_jitter_cycles = 0;
    config.write_latency_jitter_cycles = 0;
    config.transfer_channels = 1;
    config.peak_bandwidth_bytes_per_second =
        config.lpu_clock_hz * hw::kPhysicalVectorBytes;
    return config;
}

class RejectingModule final : public timing::ClockedModule {
public:
    void low() override { prepared = true; }
    void validate_posedge() const override
    {
        if (prepared)
            throw std::logic_error("intentional validation failure");
    }
    void posedge() override { ++commits; }

    bool prepared{};
    std::size_t commits{};
};

void test_ddr_memory_changes_only_at_posedge()
{
    constexpr auto kAddress = std::uint64_t {0x1000};
    auto ddr = Ddr4Model(zero_latency_ddr());
    ddr.enable_clocked_mode();
    const auto expected = make_vector(0x20);
    const auto request = ddr.request_write(kAddress, expected);

    // A real AXI write has independent AW/W/B handshakes. Zero DRAM
    // latency does not imply completion on the submission edge.
    for(std::size_t n=0;!ddr.write_completion_ready(request);++n) {
        require(n<32,"AXI write failed to complete");
        const auto cycle=ddr.cycle(),bytes=ddr.write_bytes_transferred();
        const auto before=ddr.read_vector(kAddress).payload;
        ddr.low();
        require(ddr.cycle()==cycle && ddr.write_bytes_transferred()==bytes
            && ddr.read_vector(kAddress).payload==before,"DDR published Next during LOW");
        ddr.validate_posedge();ddr.posedge();
        require(ddr.cycle()==cycle+1,"DDR did not advance once");
        const auto committed=ddr.write_bytes_transferred();ddr.high();
        require(ddr.cycle()==cycle+1 && ddr.write_bytes_transferred()==committed,"HIGH advanced time");
    }
    require(ddr.write_bytes_transferred()==hw::kPhysicalVectorBytes
        && ddr.read_vector(kAddress).payload==expected.payload,"AXI write data mismatch");
    require(ddr.axi().counters().aw==1 && ddr.axi().counters().w==1
        && ddr.axi().counters().b==1,"write completed before B");
}

void test_dma_and_ddr_share_one_commit_boundary()
{
    constexpr auto kAddress = std::uint64_t {0x2000};
    auto ddr = Ddr4Model(zero_latency_ddr());
    auto dma = C2cDmaEngine(ddr, 2);
    ddr.initialize_vector(kAddress, make_vector(0x40));
    ddr.enable_clocked_mode();
    dma.enable_clocked_mode();

    auto clock = timing::Clock {};
    auto phases = timing::SystemPhase {};
    // Deliberately register DDR before DMA. The explicit LOW topology below,
    // not Clock registration order, preserves DMA-request-before-DDR service.
    clock.register_module(ddr);
    clock.register_module(dma);
    phases.add_node("DMA", [&] { dma.low(); }, [&] { dma.high(); });
    phases.add_node("DDR", [&] { ddr.low(); }, [&] { ddr.high(); });

    dma.issue(C2cDmaInstruction::Load(kAddress));
    for(std::size_t n=0;!dma.receive_ready();++n) {
        require(n<32,"AXI read failed to reach DMA");
        const auto prior_r=ddr.axi().counters().r;
        phases.low();
        require(dma.cycle()==n && ddr.cycle()==n && !dma.receive_ready(),
            "LOW exposed next DMA data/time");
        clock.complete_low();clock.validate_all();clock.commit_all();
        require(dma.cycle()==n+1 && ddr.cycle()==n+1,"clocks diverged");
        if(dma.receive_ready())require(prior_r!=0,"DMA consumed a same-edge DDR return");
        const auto count=dma.receive_queue_size();phases.high();
        require(dma.receive_queue_size()==count,"HIGH repeated receive");
        clock.complete_high();clock.negedge_all();
    }
    require(dma.front_received().payload==make_vector(0x40).payload,"DMA payload mismatch");
}

void test_validation_rejection_blocks_dma_and_memory_commit()
{
    constexpr auto kAddress = std::uint64_t {0x3000};
    auto ddr = Ddr4Model(zero_latency_ddr());
    auto dma = C2cDmaEngine(ddr, 2);
    auto reject = RejectingModule {};
    const auto expected = make_vector(0x60);
    ddr.enable_clocked_mode();
    dma.enable_clocked_mode();

    auto clock = timing::Clock {};
    clock.register_module(dma);
    clock.register_module(ddr);
    clock.register_module(reject);
    dma.issue(C2cDmaInstruction::Store(kAddress));
    dma.send(expected);
    dma.low();
    ddr.low();
    reject.low();
    clock.complete_low();

    auto rejected = false;
    try {
        clock.validate_all();
    } catch (const std::logic_error&) {
        rejected = true;
    }
    require(rejected, "Clock accepted an intentionally invalid barrier");
    require(dma.cycle() == 0 && ddr.cycle() == 0,
        "DMA or DDR committed before all participants validated");
    require(ddr.read_vector(kAddress).payload != expected.payload,
        "DDR sparse Memory changed after rejected validation");
    require(reject.commits == 0,
        "Clock committed another module after validation failed");
}

} // namespace

int main()
{
    test_ddr_memory_changes_only_at_posedge();
    test_dma_and_ddr_share_one_commit_boundary();
    test_validation_rejection_blocks_dma_and_memory_commit();
    return 0;
}
