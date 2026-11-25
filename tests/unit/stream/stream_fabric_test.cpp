#include "ftlpu/core/stream_fabric.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>

int main()
{
    ftlpu::StreamRegisterFabric fabric(3);
    const auto stream = ftlpu::StreamId::East(7);

    // One StreamId denotes the same logical stream across all physical lanes.
    fabric.low();
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        ftlpu::StreamPayloadTileSegment segment{};
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            segment[lane] = static_cast<std::uint8_t>(tile * 16 + lane);
        }
        fabric.stage_payload_segment(0, tile, stream, segment, 123, "test producer");
    }
    fabric.validate_posedge(); fabric.posedge();

    const auto vector = fabric.vector(0, stream);
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            assert(vector[tile][lane].valid);
            assert(vector[tile][lane].data == static_cast<std::uint8_t>(tile * 16 + lane));
            assert(vector[tile][lane].vector_tag == 123);
        }
    }

    // Passive eastward propagation is one SR column per cycle.
    fabric.low();
    fabric.stage_linear_links();
    fabric.validate_posedge(); fabric.posedge();
    assert(!fabric.cell(0, 0, 0, stream).valid);
    assert(fabric.cell(1, 0, 0, stream).valid);
    assert(fabric.cell(1, 0, 0, stream).data == 0);

    // A consumer suppresses forwarding of the consumed physical segment.
    fabric.low();
    fabric.consume_segment(1, 0, stream, "test consumer");
    fabric.stage_linear_links();
    fabric.validate_posedge(); fabric.posedge();
    assert(!fabric.cell(2, 0, 0, stream).valid);
    // Other tile rows still propagate.
    assert(fabric.cell(2, 1, 0, stream).valid);

    // Valid zero and invalid are distinct states.
    assert(fabric.cell(2, 1, 0, stream).valid);
    assert(fabric.cell(2, 1, 0, stream).data == 16);
    assert(!fabric.cell(2, 0, 0, stream).valid);

    // Two producers have no winner for the collided cell. The RTL reports a
    // leaf status signal and still commits unrelated cells in the same edge.
    fabric.low();
    fabric.stage_write(0, 0, 0, stream, ftlpu::StreamCell::Valid(1), "producer A");
    fabric.stage_write(0, 0, 0, stream, ftlpu::StreamCell::Valid(2), "producer B");
    fabric.stage_write(0, 0, 1, stream, ftlpu::StreamCell::Valid(3), "producer A");
    const auto& collision_view = fabric.hardware_interface_view();
    assert(std::any_of(collision_view.collision_o.begin(),
        collision_view.collision_o.end(),
        [](bool value) { return value; }));
    assert(collision_view.inject_valid_i.size() == 3);
    fabric.validate_posedge(); fabric.posedge();
    assert(!fabric.cell(0, 0, 0, stream).valid);
    assert(fabric.cell(0, 0, 1, stream).valid);
    assert(fabric.cell(0, 0, 1, stream).data == 3);

    // Invalid consume is an observable status, not an exception or a global
    // cycle abort. An unrelated producer still commits.
    fabric.low();
    fabric.consume(0, 0, 0, stream, "invalid consumer");
    fabric.stage_write(0, 1, 0, stream, ftlpu::StreamCell::Valid(4), "producer A");
    const auto& invalid_view = fabric.hardware_interface_view();
    assert(std::any_of(invalid_view.invalid_consume_o.begin(),
        invalid_view.invalid_consume_o.end(),
        [](bool value) { return value; }));
    assert(invalid_view.consume_i.size() == 1);
    fabric.validate_posedge(); fabric.posedge();
    assert(fabric.cell(0, 1, 0, stream).valid);
    assert(fabric.cell(0, 1, 0, stream).data == 4);

    return 0;
}
