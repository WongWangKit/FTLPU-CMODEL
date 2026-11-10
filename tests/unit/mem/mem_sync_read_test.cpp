#include "ftlpu/mem/mem_array.hpp"

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

namespace {
using namespace ftlpu;
using Transfer = MemArrayModel::MemTransfer;

void check_committed(const StreamRegisterFabric& fabric, const std::vector<Transfer>& expected)
{
    for (const auto& r : expected) {
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const auto& cell = fabric.cell(r.sr_column, r.tile, lane, r.stream);
            assert(cell.valid && cell.data == r.bytes[lane]);
            assert(cell.vector_tag == r.vector_tag);
        }
    }
}

void tick(MemArrayModel& mem, StreamRegisterFabric& fabric, bool capture = true)
{
    const auto before = mem.read_responses();
    fabric.begin_cycle();
    mem.evaluate(fabric, capture);
    assert(mem.staged_read_responses().size() == before.size());
    fabric.commit_cycle();
    check_committed(fabric, before);
}
}

int main()
{
    using namespace ftlpu;
    // Cold zero, A/B phase, tail response, snapshot independent of later SRAM
    // changes, and trace-disabled execution. Debug writes stress the snapshot;
    // separate tests below use legal successive Read/Write sample edges.
    {
        auto mem = std::make_unique<MemArrayModel>(MemStreamPortMap::BetweenBoundaries(),
            MemArrayModel::MissingStreamPolicy::Error, MemArrayModel::ReadTiming::RegisteredR1);
        StreamRegisterFabric fabric(16);
        mem->enqueue_instruction(0, 0, MemInstruction::Read(0, StreamId::East(0)));
        tick(*mem, fabric, false);
        assert(mem->executed_transfers().empty());
        assert(mem->read_responses().size() == 1);
        assert(mem->read_responses()[0].bytes == StreamPayloadTileSegment{});
        assert(!fabric.cell(1, 0, 0, StreamId::East(0)).valid);
        mem->set_sram_lane_byte(0, 0, 0, 0, 0, 0xa5);
        assert(mem->read_responses()[0].bytes[0] == 0);
        tick(*mem, fabric, false);
        assert(fabric.cell(1, 0, 0, StreamId::East(0)).data == 0);
        for (std::size_t t = 2; t < hw::kTileRows; ++t) tick(*mem, fabric);
        for (std::size_t t = 0; t < hw::kTileRows; ++t) assert(!mem->instruction_at(0, 0, t));
        assert(mem->has_pending_read_responses()); // command wave empty != drained
        assert(!mem->idle() && !mem->compute_idle() && mem->transport_idle());
        tick(*mem, fabric);
        assert(!mem->has_pending_read_responses());
        assert(mem->idle() && mem->compute_idle() && mem->transport_idle());
        assert(fabric.cell(1, hw::kTileRows-1, 0, StreamId::East(0)).valid);
        tick(*mem, fabric);
        assert(!fabric.cell(1, hw::kTileRows-1, 0, StreamId::East(0)).valid);
    }
    // Consecutive read throughput, both banks, directions, row/stream/tile
    // metadata, and true bubbles. No delay wrapper around MemArrayModel.
    {
        auto mem = std::make_unique<MemArrayModel>(MemStreamPortMap::BetweenBoundaries(),
            MemArrayModel::MissingStreamPolicy::Error, MemArrayModel::ReadTiming::RegisteredR1);
        StreamRegisterFabric fabric(16);
        std::size_t responses = 0;
        for (std::size_t b = 0; b < 2; ++b)
            for (std::size_t t = 0; t < hw::kTileRows; ++t)
                for (std::size_t r = 0; r < 32; ++r)
                    for (std::size_t l = 0; l < 8; ++l)
                        mem->set_sram_lane_byte(48, b, t, r, l, b*91+t*17+r+l);
        for (std::size_t c = 0; c < 40; ++c) {
            if (c < 32 && c % 7 != 6) {
                const auto stream = c % 2 ? StreamId::West(c % 32) : StreamId::East(c % 32);
                mem->enqueue_instruction(48, c % 2, MemInstruction::Read(c, stream));
            }
            tick(*mem, fabric, c % 2 == 0);
            for (const auto& r : mem->read_responses()) {
                ++responses;
                assert(r.mem_slice == 48 && r.bank == r.address % 2);
                assert(r.stream.index() == r.address % 32);
                assert(r.sr_column == (r.address % 2 ? 12 : 13));
                for (std::size_t l = 0; l < 8; ++l)
                    assert(r.bytes[l] == std::uint8_t(r.bank*91+r.tile*17+r.address+l));
            }
        }
        assert(responses == 28*hw::kTileRows && !mem->has_pending_read_responses());
    }
    // Legal single-port same-address Read then Write on successive edges.
    // At B the old Read is staged while Write consumes B's current, not next.
    for (const bool tap : {false, true}) {
        auto mem = std::make_unique<MemArrayModel>(MemStreamPortMap::BetweenBoundaries(),
            MemArrayModel::MissingStreamPolicy::Error, MemArrayModel::ReadTiming::RegisteredR1);
        StreamRegisterFabric fabric(16);
        const auto output = StreamId::East(1), input = StreamId::East(2);
        mem->enqueue_instruction(0, MemInstruction::Read(7, output));
        tick(*mem, fabric);
        fabric.reset(); fabric.begin_cycle();
        for (std::size_t t = 0; t < hw::kTileRows; ++t) {
            StreamPayloadTileSegment bytes{}; bytes.fill(std::uint8_t(0x40+t));
            fabric.stage_payload_segment(0, t, input, bytes);
        }
        fabric.commit_cycle();
        mem->enqueue_instruction(0, tap ? MemInstruction::WriteTap(7, input) : MemInstruction::Write(7, input));
        fabric.begin_cycle(); mem->evaluate(fabric);
        assert(mem->consumed_transfers().size() == (tap ? 0 : 1));
        assert(mem->sram_lane_byte(0, 0, 7, 0) == 0x40);
        assert(mem->staged_read_responses()[0].bytes[0] == 0);
        fabric.stage_linear_links(); fabric.commit_cycle();
        assert(fabric.cell(1, 0, 0, output).valid && fabric.cell(1, 0, 0, output).data == 0);
        assert(fabric.cell(1, 0, 0, input).valid == tap);
        // In-flight response/commands cancel but SRAM persists.
        mem->reset_execution_state();
        assert(mem->read_responses().empty() && mem->staged_read_responses().empty());
        assert(mem->consumed_transfers().empty());
        fabric.reset(); tick(*mem, fabric);
        assert(!fabric.cell(1, 1, 0, output).valid);
        assert(mem->sram_lane_byte(0, 0, 7, 0) == 0x40);
        assert(mem->sram_lane_byte(0, 1, 7, 0) == 0);
    }
    // Active domain and exceptions stay in the existing c7 implementation.
    {
        auto mem = std::make_unique<MemArrayModel>(MemStreamPortMap::BetweenBoundaries(),
            MemArrayModel::MissingStreamPolicy::Error, MemArrayModel::ReadTiming::RegisteredR1);
        StreamRegisterFabric fabric(16);
        mem->set_sram_depth_rows(64);
        mem->enqueue_instruction(0, MemInstruction::Read(64, 0));
        fabric.begin_cycle(); bool rejected = false;
        try { mem->evaluate(fabric); } catch (const std::out_of_range&) { rejected = true; }
        assert(rejected && mem->read_responses().empty());
        mem->reset_execution_state(); fabric.reset();
        mem->enqueue_instruction(0, MemInstruction::Read(63, 0)); tick(*mem, fabric);
        assert(mem->read_responses()[0].bytes == StreamPayloadTileSegment{});
        mem->reset();
        assert(mem->read_responses().empty());
    }
    std::cout << "MEM_SYNC_MODEL TEST_PASS A_response B_commit snapshot throughput reset trace_off active_domain\n";
}
