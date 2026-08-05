#include "ftlpu/c2c/c2c.hpp"

#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace {

void initialize_vector(
    ftlpu::StreamRegisterFabric& fabric,
    std::size_t column,
    ftlpu::StreamId stream,
    std::uint64_t tag)
{
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            const auto value = static_cast<std::uint8_t>(tile * 16 + lane);
            fabric.initialize_cell(
                column,
                tile,
                lane,
                stream,
                ftlpu::StreamCell::Valid(value, lane + 1 == 16, tag));
        }
    }
}

void test_one_vector_per_cycle()
{
    ftlpu::StreamRegisterFabric sender_fabric(1);
    ftlpu::StreamRegisterFabric receiver_fabric(1);

    initialize_vector(sender_fabric, 0, ftlpu::StreamId::East(3), 91);

    ftlpu::C2cLink link(ftlpu::C2cLinkConfig{
        ftlpu::hw::kPhysicalVectorBytes,
        0,
        2,
    });

    ftlpu::C2cTxSlice tx(
        {0, ftlpu::StreamDirection::East},
        "sender TX");
    ftlpu::C2cRxSlice rx(
        {0, ftlpu::StreamDirection::West},
        "receiver RX");

    tx.enqueue_send(3);

    sender_fabric.begin_cycle();
    tx.evaluate(sender_fabric, link);
    sender_fabric.commit_cycle();

    assert(!link.receive_ready());
    link.tick();
    assert(link.receive_ready());
    assert(link.config().serialization_cycles() == 1);

    rx.enqueue_receive(9);
    receiver_fabric.begin_cycle();
    rx.evaluate(receiver_fabric, link);
    receiver_fabric.commit_cycle();

    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            const auto& cell = receiver_fabric.cell(
                0,
                tile,
                lane,
                ftlpu::StreamId::West(9));
            assert(cell.valid);
            assert(cell.data == static_cast<std::uint8_t>(tile * 16 + lane));
            assert(cell.vector_tag == 91);
        }
    }
}

void test_configurable_beat_width()
{
    ftlpu::C2cLink link(ftlpu::C2cLinkConfig{
        160,
        0,
        1,
    });

    ftlpu::C2cVector vector{};
    link.send(vector);

    link.tick();
    assert(!link.receive_ready());

    link.tick();
    assert(link.receive_ready());
    assert(link.config().serialization_cycles() == 2);
}

void test_early_receive_is_rejected()
{
    ftlpu::StreamRegisterFabric receiver_fabric(1);
    ftlpu::C2cLink link;
    ftlpu::C2cRxSlice rx(
        {0, ftlpu::StreamDirection::West},
        "receiver RX");

    rx.enqueue_receive(0);
    receiver_fabric.begin_cycle();

    bool rejected = false;
    try {
        rx.evaluate(receiver_fabric, link);
    } catch (const std::logic_error&) {
        rejected = true;
    }

    assert(rejected);
    receiver_fabric.commit_cycle();
}

} // namespace

int main()
{
    test_one_vector_per_cycle();
    test_configurable_beat_width();
    test_early_receive_is_rejected();
    return 0;
}
