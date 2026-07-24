#include "ftlpu/vxm/slice.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace {

void put_int32(ftlpu::VxmLane::StreamBytes& streams, std::size_t base,
               std::int32_t value)
{
    const auto bytes = ftlpu::VxmLane::pack_int32(value);
    for (std::size_t byte = 0; byte < 4; ++byte) streams[base + byte] = bytes[byte];
}

}

int main()
{
    using namespace ftlpu;
    auto slice = VxmSlice{};
    slice.set_chain_depth(VxmChainDepth::Two);

    slice.issue_south(0, {VxmAluOpcode::Add,
        VxmLaneOperand::StreamInt32(), VxmLaneOperand::StreamInt32()});
    auto tail = VxmLaneAluInstruction{VxmAluOpcode::Bypass,
        VxmLaneOperand::Previous()};
    tail.output_type = VxmCastTarget::Int8;
    tail.output_scale = 1.0f;
    tail.output_stream = 0;
    slice.issue_south(1, tail);

    auto input = VxmSlice::StreamMatrix{};
    for (std::size_t lane = 0; lane < VxmSuperlane::kLaneCount; ++lane) {
        put_int32(input[lane], 0, static_cast<std::int32_t>(lane));
        put_int32(input[lane], 4, 10);
    }
    slice.prepare_cycle();
    assert(slice.required_streams_at(0));
    for (std::size_t stream = 0; stream < 8; ++stream) {
        assert((*slice.required_streams_at(0))[stream]);
    }

    slice.tick(); // instruction reaches tile 0 and spends one cycle decoding
    assert(!slice.output_at(0));
    slice.set_stream_inputs(0, input);
    slice.tick();
    assert(!slice.output_at(0));
    slice.tick();
    assert(slice.output_at(0));
    assert(slice.output_at(0)->stream == 0);
    for (std::size_t lane = 0; lane < VxmSuperlane::kLaneCount; ++lane) {
        assert(slice.output_at(0)->values[lane] == static_cast<std::int8_t>(lane + 10));
    }

    // The decoded instruction row propagates one superlane per cycle.
    assert(slice.instruction_at(0, 3).has_value());
    assert(slice.cycle() == 3);

    // A repeat-count Current Config remains the source of stream requirements
    // after its one instruction packet has moved away from this tile.
    auto repeated = VxmSlice{};
    repeated.set_chain_depth(VxmChainDepth::Two);
    auto repeated_head = VxmLaneAluInstruction{VxmAluOpcode::Bypass,
        VxmLaneOperand::StreamInt32()};
    repeated_head.repeat_count = 2;
    repeated.issue_south(0, repeated_head);
    auto repeated_tail = VxmLaneAluInstruction{VxmAluOpcode::Bypass,
        VxmLaneOperand::Previous()};
    repeated_tail.output_type = VxmCastTarget::Float32;
    repeated_tail.output_stream = 0;
    repeated_tail.repeat_count = 2;
    repeated.issue_south(1, repeated_tail);

    auto repeated_input = VxmSlice::StreamMatrix{};
    for (std::size_t lane = 0; lane < VxmSuperlane::kLaneCount; ++lane) {
        put_int32(repeated_input[lane], 0, static_cast<std::int32_t>(lane));
    }
    repeated.tick(); // instruction reaches tile 0 and starts decoding
    repeated.prepare_cycle();
    assert(repeated.required_streams_at(0));
    for (std::size_t stream = 0; stream < 4; ++stream) {
        assert((*repeated.required_streams_at(0))[stream]);
    }
    repeated.set_stream_inputs(0, repeated_input);
    repeated.tick();
    repeated.set_stream_inputs(0, repeated_input);
    repeated.tick();
    assert(repeated.output_at(0));
    repeated.tick();
    assert(repeated.output_at(0));
    return 0;
}
