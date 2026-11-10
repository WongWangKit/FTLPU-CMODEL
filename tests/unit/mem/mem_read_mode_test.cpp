#include "ftlpu/mem/mem_array.hpp"

#include <cassert>
#include <iostream>
#include <memory>

int main()
{
    using namespace ftlpu;
    using Timing = MemArrayModel::ReadTiming;
    assert(std::make_unique<MemArrayModel>()->read_timing() == Timing::Main);
    for (const auto mode : {Timing::Main, Timing::RegisteredR1}) {
        const std::size_t delay = mode == Timing::RegisteredR1 ? 1 : 0;
        for (const bool transport : {false, true}) {
            auto mem = std::make_unique<MemArrayModel>(MemStreamPortMap::BetweenBoundaries(),
                MemArrayModel::MissingStreamPolicy::Error, mode);
            StreamRegisterFabric fabric(hw::kMemBoundaryStreamRegisterColumns);
            auto read = MemInstruction::Read(3, StreamId::East(0));
            if (transport) read.transport_generation = 7;
            for (std::size_t t = 0; t < hw::kTileRows; ++t)
                for (std::size_t l = 0; l < hw::kLanesPerTile; ++l)
                    mem->set_sram_lane_byte(0, 0, t, 3, l, std::uint8_t(1 + t * 8 + l));
            mem->enqueue_instruction(0, read);
            for (std::size_t c = 0; c <= hw::kTileRows; ++c) {
                fabric.begin_cycle();
                mem->evaluate(fabric, false);
                assert(mem->executed_transfers().empty());
                for (std::size_t t = 0; t < hw::kTileRows; ++t)
                    assert(fabric.staged_segment_valid(1, t, StreamId::East(0))
                        == (c >= delay && t == c - delay));
                fabric.commit_cycle();
                if (c >= delay && c - delay < hw::kTileRows) {
                    const auto t = c - delay;
                    for (std::size_t l = 0; l < hw::kLanesPerTile; ++l)
                        assert(fabric.cell(1, t, l, StreamId::East(0)).data
                            == std::uint8_t(1 + t * 8 + l));
                }
                if (c == hw::kTileRows - 1) {
                    assert(mem->idle() == (delay == 0));
                    assert(mem->compute_idle() == (delay == 0 || transport));
                    assert(mem->transport_idle() == (delay == 0 || !transport));
                    assert(mem->has_pending_read_responses() == (delay != 0));
                }
            }
            assert(mem->idle() && mem->compute_idle() && mem->transport_idle());
            mem->reset_execution_state();
            assert(mem->read_timing() == mode);
            assert(mem->read_responses().empty() && mem->staged_read_responses().empty());
        }

        // Upstream transport commit accounting must survive either read mode.
        for (const bool tap : {false, true}) {
            auto mem = std::make_unique<MemArrayModel>(MemStreamPortMap::BetweenBoundaries(),
                MemArrayModel::MissingStreamPolicy::Error, mode);
            StreamRegisterFabric fabric(hw::kMemBoundaryStreamRegisterColumns);
            mem->configure_transport_commit(7, 1, 0, 0, 1);
            auto write = tap ? MemInstruction::WriteTap(5, StreamId::East(0))
                             : MemInstruction::Write(5, StreamId::East(0));
            write.transport_generation = 7;
            write.transport_launch = 1;
            mem->enqueue_instruction(0, write);
            for (std::size_t t = 0; t < hw::kTileRows; ++t) {
                fabric.begin_cycle();
                StreamPayloadTileSegment bytes{};
                bytes.fill(std::uint8_t(0x30 + t));
                StreamOutputPort source(fabric, 0, StreamDirection::East, "transport", true);
                source.write_payload_segment(t, 0, bytes);
                fabric.commit_cycle();
                fabric.begin_cycle();
                mem->evaluate(fabric, false);
                assert(mem->last_transport_commit_segments() == 1);
                assert(mem->transport_committed_segments(7, 1, 0, 0) == t + 1);
                assert(mem->consumed_transfers().size() == (tap ? 0 : 1));
                assert(mem->sram_lane_byte(0, t, 5, 0) == std::uint8_t(0x30 + t));
                fabric.stage_linear_links();
                fabric.commit_cycle();
            }
            assert(mem->idle());
            mem->reset_execution_state();
            assert(mem->last_transport_commit_segments() == 0);
            assert(mem->idle() && mem->consumed_transfers().empty());
        }

        for (const bool wrong_provenance : {false, true}) {
            auto mem = std::make_unique<MemArrayModel>(MemStreamPortMap::BetweenBoundaries(),
                MemArrayModel::MissingStreamPolicy::Error, mode);
            StreamRegisterFabric fabric(hw::kMemBoundaryStreamRegisterColumns);
            mem->configure_transport_commit(7, 1, 0, 0, 1);
            auto write = MemInstruction::Write(5, StreamId::East(0));
            write.transport_generation = 7;
            write.transport_launch = 1;
            mem->enqueue_instruction(0, write);
            if (wrong_provenance) {
                fabric.begin_cycle();
                StreamOutputPort source(fabric, 0, StreamDirection::East, "computation");
                source.write_payload_segment(0, 0, StreamPayloadTileSegment{});
                fabric.commit_cycle();
            }
            fabric.begin_cycle();
            bool rejected = false;
            try { mem->evaluate(fabric, false); }
            catch (const std::logic_error&) { rejected = true; }
            assert(rejected && mem->last_transport_commit_segments() == 0);
            assert(mem->transport_committed_segments(7, 1, 0, 0) == 0);
        }
    }
    std::cout << "MEM_READ_MODE PASS default_main R1_phase idle_domains transport_commit reset negatives\n";
}
