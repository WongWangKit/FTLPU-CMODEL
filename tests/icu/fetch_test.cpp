#include "ftlpu/icu/icu.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace {

using PacketArray = ftlpu::IcuFetchBuffer::PacketArray;

PacketArray make_mem_program()
{
    PacketArray packets{};
    for (std::size_t packet = 0; packet < packets.size(); ++packet) {
        packets[packet] = ftlpu::isa::encode_packet(
            ftlpu::MemInstruction::Read(
                packet,
                ftlpu::StreamId::West(packet % ftlpu::hw::kStreamsPerDirection)));
    }
    return packets;
}

void stage_vector(
    ftlpu::StreamRegisterFabric& fabric,
    const PacketArray& packets,
    std::size_t vector,
    ftlpu::StreamId stream,
    std::uint64_t tag,
    std::size_t tile_count = ftlpu::hw::kTileRows)
{
    for (std::size_t tile = 0; tile < tile_count; ++tile) {
        const auto& packet = packets[vector * ftlpu::hw::kTileRows + tile];
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            fabric.initialize_cell(
                0,
                tile,
                lane,
                stream,
                ftlpu::StreamCell::Valid(packet.bytes[lane], lane == 15, tag));
        }
    }
}

template <typename Fn>
bool throws(Fn&& fn)
{
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

} // namespace

int main()
{
    const auto packets = make_mem_program();
    const auto stream = ftlpu::StreamId::East(3);
    auto fabric = std::make_unique<ftlpu::StreamRegisterFabric>(1);
    ftlpu::SliceIcu<ftlpu::MemInstruction> icu;
    const auto already_queued = ftlpu::MemInstruction::Write(77, ftlpu::StreamId::West(1));
    icu.load({ftlpu::IcuControlInstruction::Fetch(stream), already_queued});

    // Fetch retires and reserves 640 bytes.  The following old IQ entry can
    // dispatch while the frontend receives the two vectors.
    assert(!icu.dispatch().has_value());
    assert(icu.fetch_active());
    assert(icu.reserved_bytes() == ftlpu::hw::kIcuFetchBufferBytes);

    stage_vector(*fabric, packets, 0, stream, 100);
    fabric->begin_cycle();
    icu.evaluate_fetch(*fabric, 0);
    const auto parallel = icu.dispatch();
    assert(parallel.has_value() && parallel->opcode == already_queued.opcode);
    fabric->commit_cycle();
    icu.commit_fetch();
    assert(icu.fetch_active());
    assert(!icu.fetch_complete_pending_commit());

    stage_vector(*fabric, packets, 1, stream, 101);
    fabric->begin_cycle();
    icu.evaluate_fetch(*fabric, 0);
    assert(icu.fetch_complete_pending_commit());
    // Decoded next-state is not dispatch-visible in its completion cycle.
    assert(!icu.dispatch().has_value());
    fabric->commit_cycle();
    icu.commit_fetch();
    assert(!icu.fetch_active());
    assert(icu.iq_occupancy() == ftlpu::hw::kIcuFetchPackets);
    const auto first_fetched = icu.dispatch();
    assert(first_fetched.has_value());
    assert(first_fetched->address == ftlpu::MemLocalWordAddress13(0));

    // A second Fetch reaching the head before the first completes is a static
    // scheduling error rather than an implicit second private buffer.
    {
        ftlpu::SliceIcu<ftlpu::MemInstruction> second;
        second.load({
            ftlpu::IcuControlInstruction::Fetch(stream),
            ftlpu::IcuControlInstruction::Fetch(stream),
        });
        assert(!second.dispatch().has_value());
        assert(throws([&] { (void)second.dispatch(); }));
    }

    // Incomplete stream data leaves the frontend pending and the IQ intact.
    {
        auto partial_fabric = std::make_unique<ftlpu::StreamRegisterFabric>(1);
        ftlpu::SliceIcu<ftlpu::MemInstruction> partial;
        partial.load({ftlpu::IcuControlInstruction::Fetch(stream)});
        assert(!partial.dispatch().has_value());
        stage_vector(*partial_fabric, packets, 0, stream, 200, 1);
        partial_fabric->begin_cycle();
        partial.evaluate_fetch(*partial_fabric, 0);
        partial_fabric->commit_cycle();
        partial.commit_fetch();
        assert(partial.fetch_active());
        assert(partial.iq_occupancy() == 0);
        assert(partial.fetch_state()->buffer.received_mask(0).count() == 1);
    }

    return 0;
}
