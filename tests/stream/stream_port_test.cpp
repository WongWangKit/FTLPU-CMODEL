#include "ftlpu/core/stream_port.hpp"

#include <cassert>
#include <cstdint>

int main()
{
    ftlpu::StreamRegisterFabric fabric(2);
    ftlpu::StreamPayloadSegment16 payload{};
    for (std::size_t lane = 0; lane < payload.size(); ++lane) {
        payload[lane] = static_cast<std::uint8_t>(lane);
    }

    fabric.begin_cycle();
    ftlpu::StreamOutputPort producer(
        fabric, 0, ftlpu::StreamDirection::East, "producer");
    constexpr auto kTile = ftlpu::hw::kTileRows - 1;
    producer.write_payload_segment(kTile, 7, payload, 123);
    fabric.commit_cycle();

    fabric.begin_cycle();
    ftlpu::StreamInputPort consumer_a(
        fabric, 0, ftlpu::StreamDirection::East, "consumer A");
    ftlpu::StreamInputPort consumer_b(
        fabric, 0, ftlpu::StreamDirection::East, "consumer B");
    assert(consumer_a.segment_valid(kTile, 7));
    const auto segment_a = consumer_a.consume_segment(kTile, 7);
    assert(consumer_b.segment_valid(kTile, 7));
    const auto segment_b = consumer_b.consume_segment(kTile, 7);
    for (std::size_t lane = 0; lane < segment_a.size(); ++lane) {
        assert(segment_a[lane].data == lane);
        assert(segment_a[lane].vector_tag == 123);
        assert(segment_b[lane].data == segment_a[lane].data);
        assert(segment_b[lane].vector_tag == segment_a[lane].vector_tag);
    }
    fabric.stage_linear_links();
    fabric.commit_cycle();

    // Broadcast consumption still suppresses passive forwarding once.
    assert(!fabric.cell(1, kTile, 0, ftlpu::StreamId::East(7)).valid);
    return 0;
}
