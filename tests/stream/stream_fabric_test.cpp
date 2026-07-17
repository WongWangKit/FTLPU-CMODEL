#include "ftlpu/core/stream_fabric.hpp"

#include <cassert>
#include <cstdint>
#include <stdexcept>

int main()
{
    ftlpu::StreamRegisterFabric fabric(3);
    const auto stream = ftlpu::StreamId::East(7);

    // One StreamId denotes the same logical stream across all physical lanes.
    fabric.begin_cycle();
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        ftlpu::StreamPayloadSegment16 segment{};
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            segment[lane] = static_cast<std::uint8_t>(tile * 16 + lane);
        }
        fabric.stage_payload_segment(0, tile, stream, segment, 123, "test producer");
    }
    fabric.commit_cycle();

    const auto vector = fabric.vector(0, stream);
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            assert(vector[tile][lane].valid);
            assert(vector[tile][lane].data == static_cast<std::uint8_t>(tile * 16 + lane));
            assert(vector[tile][lane].vector_tag == 123);
        }
    }

    // Passive eastward propagation is one SR column per cycle.
    fabric.begin_cycle();
    fabric.stage_linear_links();
    fabric.commit_cycle();
    assert(!fabric.cell(0, 0, 0, stream).valid);
    assert(fabric.cell(1, 0, 0, stream).valid);
    assert(fabric.cell(1, 0, 0, stream).data == 0);

    // A consumer suppresses forwarding of the consumed physical segment.
    fabric.begin_cycle();
    fabric.consume_segment(1, 0, stream, "test consumer");
    fabric.stage_linear_links();
    fabric.commit_cycle();
    assert(!fabric.cell(2, 0, 0, stream).valid);
    // Other tile rows still propagate.
    assert(fabric.cell(2, 1, 0, stream).valid);

    // Valid zero and invalid are distinct states.
    assert(fabric.cell(2, 1, 0, stream).valid);
    assert(fabric.cell(2, 1, 0, stream).data == 16);
    assert(!fabric.cell(2, 0, 0, stream).valid);

    // Two producers cannot write the same physical SR cell in one cycle.
    bool collision = false;
    fabric.begin_cycle();
    fabric.stage_write(0, 0, 0, stream, ftlpu::StreamCell::Valid(1), "producer A");
    try {
        fabric.stage_write(0, 0, 0, stream, ftlpu::StreamCell::Valid(2), "producer B");
    } catch (const std::logic_error&) {
        collision = true;
    }
    assert(collision);

    return 0;
}
