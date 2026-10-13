#include "ftlpu/vxm/sr_slice.hpp"

#include <cassert>

int main()
{
    using namespace ftlpu;
    VxmSrSlice slice {};

    VxmSrInstruction add {};
    add.operation = VxmAluOpcode::Add;
    add.lhs = VxmSrOperand::StreamFloat16(StreamId::East(0).packed());
    add.rhs = VxmSrOperand::StreamFloat16(StreamId::East(2).packed());
    add.result_sr = static_cast<std::uint8_t>(StreamId::East(4).packed());
    add.result_type = VxmCastTarget::Float16;
    slice.issue_south(0, VxmSrCompactInstructionCodec::encode(add));

    auto lhs = StreamTileSegment {};
    auto lhsHigh = StreamTileSegment {};
    auto rhs = StreamTileSegment {};
    auto rhsHigh = StreamTileSegment {};
    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
        const auto a = VxmDataFormat::float_to_fp16_bits(
            1.0f + static_cast<float>(lane));
        const auto b = VxmDataFormat::float_to_fp16_bits(2.0f);
        lhs[lane] = StreamCell::Valid(static_cast<std::uint8_t>(a));
        lhsHigh[lane] = StreamCell::Valid(static_cast<std::uint8_t>(a >> 8));
        rhs[lane] = StreamCell::Valid(static_cast<std::uint8_t>(b));
        rhsHigh[lane] = StreamCell::Valid(static_cast<std::uint8_t>(b >> 8));
    }

    // Cycle 0 transports both the config and data into tile 0 / SR0.
    slice.begin_cycle();
    slice.stage_ingress(0, StreamId::East(0), lhs);
    slice.stage_ingress(0, StreamId::East(1), lhsHigh);
    slice.stage_ingress(0, StreamId::East(2), rhs);
    slice.stage_ingress(0, StreamId::East(3), rhsHigh);
    slice.tick();
    assert(slice.config_entry_count(0, 0) == 1);

    // Cycle 1 executes C0 and writes the named result to the ordinary SR
    // immediately after C0.
    slice.tick();
    const auto& fabric = slice.datapath().fabric().registers();
    assert(fabric.segment_valid(1, 0, StreamId::East(4)));
    assert(fabric.segment_valid(1, 0, StreamId::East(5)));
    assert(!fabric.segment_valid(1, 0, StreamId::East(0)));
    assert(slice.last_execution()[0][0]);
    assert(slice.config_entry_count(0, 0) == 0);
    return 0;
}
