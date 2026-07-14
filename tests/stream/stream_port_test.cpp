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
    producer.write_payload_segment(4, 7, payload, 123);
    fabric.commit_cycle();

    fabric.begin_cycle();
    ftlpu::StreamInputPort consumer(
        fabric, 0, ftlpu::StreamDirection::East, "consumer");
    assert(consumer.segment_valid(4, 7));
    const auto segment = consumer.consume_segment(4, 7);
    for (std::size_t lane = 0; lane < segment.size(); ++lane) {
        assert(segment[lane].data == lane);
        assert(segment[lane].vector_tag == 123);
    }
    fabric.stage_linear_links();
    fabric.commit_cycle();

    // A consumed operand does not pass through the passive SR link.
    assert(!fabric.cell(1, 4, 0, ftlpu::StreamId::East(7)).valid);
    return 0;
}
