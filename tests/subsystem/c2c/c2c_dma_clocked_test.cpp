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

    ddr.low();
    require(ddr.cycle() == 0 && ddr.write_bytes_transferred() == 0,
        "DDR exposed Next counters during LOW");
    require(ddr.read_vector(kAddress).payload != expected.payload,
        "DDR sparse Memory write became visible before POSEDGE");
    require(!ddr.write_completion_ready(request),
        "DDR completion became visible before POSEDGE");

    ddr.validate_posedge();
    ddr.posedge();
    require(ddr.cycle() == 1
            && ddr.write_bytes_transferred() == hw::kPhysicalVectorBytes,
        "DDR did not publish counters at POSEDGE");
    require(ddr.read_vector(kAddress).payload == expected.payload,
        "DDR sparse Memory write was not committed at POSEDGE");
    require(ddr.write_completion_ready(request),
        "DDR completion was not committed with the final beat");

    ddr.high();
    require(ddr.cycle() == 1
            && ddr.write_bytes_transferred() == hw::kPhysicalVectorBytes,
        "DDR HIGH repeated sequential progression");
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
    phases.low();
    require(dma.cycle() == 0 && ddr.cycle() == 0,
        "DMA or DDR exposed Next cycle state during LOW");
    require(!dma.receive_ready(),
        "DMA exposed a returned vector during request LOW");
    clock.complete_low();
    clock.validate_all();
    clock.commit_all();
    require(dma.cycle() == 1 && ddr.cycle() == 1,
        "DMA and DDR did not commit the first edge together");
    require(!dma.receive_ready(),
        "DMA consumed a DDR completion before a second LOW");
    phases.high();
    clock.complete_high();
    clock.negedge_all();

    phases.low();
    require(!dma.receive_ready(),
        "DMA exposed the returned vector before the second POSEDGE");
    clock.complete_low();
    clock.validate_all();
    clock.commit_all();
    require(dma.cycle() == 2 && ddr.cycle() == 2
            && dma.receive_ready(),
        "DMA return and DDR completion pop did not commit together");
    const auto returned = dma.front_received();
    require(returned.payload == make_vector(0x40).payload,
        "DMA returned the wrong DDR vector");
    phases.high();
    require(dma.cycle() == 2 && dma.receive_queue_size() == 1,
        "DMA HIGH repeated receive or state progression");
    clock.complete_high();
    clock.negedge_all();
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
