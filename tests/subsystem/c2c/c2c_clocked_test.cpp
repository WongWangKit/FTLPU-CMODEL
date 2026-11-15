#include "ftlpu/c2c/link.hpp"
#include "ftlpu/c2c/slice.hpp"
#include "ftlpu/core/timing.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace ftlpu;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

class VectorTransport {
public:
    bool can_send() const noexcept { return true; }
    void send(C2cVector vector) { sent.push_back(std::move(vector)); }
    bool receive_ready() const noexcept { return !received.empty(); }
    C2cVector pop_received()
    {
        auto vector = std::move(received.front());
        received.pop_front();
        return vector;
    }

    std::deque<C2cVector> received{};
    std::vector<C2cVector> sent{};
};

class RejectingModule final : public timing::ClockedModule {
public:
    void low() override { prepared = true; }
    void validate_posedge() const override
    {
        if (prepared) throw std::logic_error("intentional validation failure");
    }
    void posedge() override { ++commits; }

    bool prepared{};
    std::size_t commits{};
};

void test_link_current_changes_only_at_posedge()
{
    auto link = C2cLink(C2cLinkConfig {
        hw::kPhysicalVectorBytes, 0, 2});
    link.send(C2cVector {});

    link.low();
    require(link.cycle() == 0,
        "C2C link exposed Next cycle state during LOW");
    require(!link.receive_ready(),
        "C2C link exposed a Next RX entry during LOW");

    link.validate_posedge();
    link.posedge();
    require(link.cycle() == 1 && link.receive_ready(),
        "C2C link did not publish state at POSEDGE");

    link.high();
    require(link.cycle() == 1 && link.receive_queue_size() == 1,
        "C2C link HIGH changed sequential state");
}

void test_validation_barrier_prevents_partial_link_commit()
{
    auto link = C2cLink(C2cLinkConfig {
        hw::kPhysicalVectorBytes, 0, 2});
    auto reject = RejectingModule {};
    auto clock = timing::Clock {};
    clock.register_module(link);
    clock.register_module(reject);

    link.send(C2cVector {});
    link.low();
    reject.low();
    clock.complete_low();

    auto rejected = false;
    try {
        clock.validate_all();
    } catch (const std::logic_error&) {
        rejected = true;
    }
    require(rejected, "Clock accepted an intentionally invalid barrier");
    require(link.cycle() == 0 && !link.receive_ready(),
        "C2C link committed before all participants validated");
    require(reject.commits == 0,
        "Clock committed another module after validation failed");
}

void test_endpoint_system_order_is_independent_of_clock_order()
{
    constexpr auto kStream = std::size_t {3};
    auto fabric = StreamRegisterFabric(1);
    auto endpoint = C2cEndpoint(C2cStreamPortMap::EastEdge(0));
    auto transport = VectorTransport {};
    auto system_phase = timing::SystemPhase {};
    auto clock = timing::Clock {};

    // Deliberately reverse the sequential registration order.  SystemPhase,
    // not Clock registration, still owns fabric-before-endpoint LOW topology.
    clock.register_module(endpoint);
    clock.register_module(fabric);
    system_phase.add_node("stream fabric", [&] { fabric.low(); },
        [&] { fabric.high(); });
    system_phase.add_node("C2C endpoint", [&] {
        endpoint.evaluate(fabric, transport, transport);
    }, [&] { endpoint.high(); });

    endpoint.issue(C2cInstruction::Send(kStream));
    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
        fabric.initialize_cell(0, 0, lane, StreamId::East(kStream),
            StreamCell::Valid(static_cast<std::uint8_t>(lane),
                lane + 1 == hw::kLanesPerTile, 7));
    }

    system_phase.low();
    require(endpoint.tx().gathered_tile_count() == 0,
        "C2C endpoint exposed its Next gather pipeline during LOW");
    clock.complete_low();
    clock.validate_all();
    clock.commit_all();
    require(endpoint.tx().gathered_tile_count() == 1,
        "C2C endpoint did not commit its prepared gather pipeline");

    system_phase.high();
    require(endpoint.tx().gathered_tile_count() == 1,
        "C2C endpoint HIGH repeated sequential progression");
    clock.complete_high();
    clock.negedge_all();
}

} // namespace

int main()
{
    test_link_current_changes_only_at_posedge();
    test_validation_barrier_prevents_partial_link_commit();
    test_endpoint_system_order_is_independent_of_clock_order();
    return 0;
}
