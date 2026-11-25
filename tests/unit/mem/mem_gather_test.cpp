#include "ftlpu/mem/mem_array.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
#ifdef FTLPU_TEST_MEM_R1
constexpr auto kReadTiming = ftlpu::MemArrayModel::ReadTiming::RegisteredR1;
constexpr std::size_t kReadLatency = 1;
#else
constexpr auto kReadTiming = ftlpu::MemArrayModel::ReadTiming::Main;
constexpr std::size_t kReadLatency = 0;
#endif

// Group 2 of the hemisphere: east input SR2, east output SR3.
constexpr std::size_t kMemSlice = 8;
constexpr std::size_t kInputColumn = 2;
constexpr std::size_t kOutputColumn = 3;
constexpr std::size_t kIssueCycle = 3;
constexpr std::size_t kTiles = ftlpu::hw::kTileRows;
constexpr std::size_t kLanes = ftlpu::hw::kLanesPerTile;

std::uint8_t row_payload(std::size_t row, std::size_t tile, std::size_t lane)
{
    return static_cast<std::uint8_t>(row * 37 + tile * 11 + lane * 3 + 1);
}

std::uint8_t stream_payload(std::size_t tile, std::size_t lane)
{
    return static_cast<std::uint8_t>(0x80 + tile * 8 + lane);
}

// The map contract carries one little-endian 16-bit row index per tile in
// bytes 0..1; bytes 2..7 stay zero for a well-formed segment.
ftlpu::StreamPayloadTileSegment map_segment(std::size_t row)
{
    ftlpu::StreamPayloadTileSegment bytes{};
    bytes[0] = static_cast<std::uint8_t>(row & 0xffu);
    bytes[1] = static_cast<std::uint8_t>((row >> 8) & 0xffu);
    return bytes;
}

// One cycle of the shared commit protocol: every functional slice evaluates
// against current SR state, then links stage, then the fabric commits.
template <typename StageFn>
void advance(
    ftlpu::MemArrayModel& mem,
    ftlpu::StreamRegisterFabric& fabric,
    std::size_t cycle,
    StageFn stage_inputs)
{
    fabric.begin_cycle();
    mem.evaluate(fabric);
    fabric.stage_linear_links();
    if (cycle + 1 >= kIssueCycle && cycle + 1 < kIssueCycle + kTiles) {
        stage_inputs(cycle + 1 - kIssueCycle);
    }
    fabric.commit_cycle();
}

void test_gather_reads_one_indirect_row_per_tile()
{
    ftlpu::StreamRegisterFabric fabric(
        ftlpu::hw::kMemBoundaryStreamRegisterColumns);
    ftlpu::MemArrayModel mem(
        ftlpu::MemStreamPortMap::BetweenBoundaries(),
        ftlpu::MemArrayModel::MissingStreamPolicy::Error, kReadTiming);
    const auto data_stream = ftlpu::StreamId::East(5);
    const auto map_stream = ftlpu::StreamId::East(6);

    std::array<std::size_t, kTiles> rows{};
    for (std::size_t tile = 0; tile < kTiles; ++tile) {
        rows[tile] = 40 + tile * 7;
        for (std::size_t lane = 0; lane < kLanes; ++lane) {
            mem.set_sram_lane_byte(kMemSlice, 0, tile, rows[tile], lane,
                row_payload(rows[tile], tile, lane));
        }
    }

    std::vector<std::size_t> gathered;
    std::size_t outputs_checked = 0;
    for (std::size_t cycle = 0; cycle < kIssueCycle + kTiles + 4; ++cycle) {
        if (cycle == kIssueCycle) {
            mem.enqueue_instruction(kMemSlice, 0,
                ftlpu::MemInstruction::Gather(data_stream, map_stream));
        }
        advance(mem, fabric, cycle, [&](std::size_t tile) {
            fabric.stage_payload_segment(kInputColumn, tile, map_stream,
                map_segment(rows[tile]), cycle, "gather map");
        });
        for (const auto& transfer : mem.executed_transfers()) {
            if (transfer.kind
                == ftlpu::MemArrayModel::MemTransfer::Kind::LoadSramToStream) {
                gathered.push_back(transfer.address);
            }
        }

        // The map segment was consumed by the tile that sampled it, so it
        // must not continue to the next boundary. Reading after commit_cycle
        // observes the state the next cycle will start from.
        if (cycle >= kIssueCycle) {
            const auto tile = cycle - kIssueCycle;
            if (tile < kTiles) {
                assert(!fabric.cell(kInputColumn + 1, tile, 0, map_stream).valid);
            }
        }

        // A gathered tile commits its 8-byte segment one edge after sampling,
        // plus the extra registered response edge in R1 mode.
        if (cycle >= kIssueCycle + kReadLatency) {
            const auto tile = cycle - kIssueCycle - kReadLatency;
            if (tile < kTiles) {
                for (std::size_t lane = 0; lane < kLanes; ++lane) {
                    const auto& cell =
                        fabric.cell(kOutputColumn, tile, lane, data_stream);
                    assert(cell.valid);
                    assert(cell.data == row_payload(rows[tile], tile, lane));
                }
                ++outputs_checked;
            }
        }
    }

    assert(gathered.size() == kTiles);
    for (std::size_t tile = 0; tile < kTiles; ++tile) {
        assert(gathered[tile] == rows[tile]);
    }
    assert(outputs_checked == kTiles);
}

void test_scatter_writes_one_indirect_row_per_tile()
{
    ftlpu::StreamRegisterFabric fabric(
        ftlpu::hw::kMemBoundaryStreamRegisterColumns);
    ftlpu::MemArrayModel mem(
        ftlpu::MemStreamPortMap::BetweenBoundaries(),
        ftlpu::MemArrayModel::MissingStreamPolicy::Error, kReadTiming);
    const auto data_stream = ftlpu::StreamId::East(7);
    const auto map_stream = ftlpu::StreamId::East(6);

    std::array<std::size_t, kTiles> rows{};
    for (std::size_t tile = 0; tile < kTiles; ++tile) {
        rows[tile] = 60 + tile * 5;
    }

    std::vector<std::size_t> scattered;
    for (std::size_t cycle = 0; cycle < kIssueCycle + kTiles + 4; ++cycle) {
        if (cycle == kIssueCycle) {
            mem.enqueue_instruction(kMemSlice, 0,
                ftlpu::MemInstruction::Scatter(data_stream, map_stream));
        }
        advance(mem, fabric, cycle, [&](std::size_t tile) {
            fabric.stage_payload_segment(kInputColumn, tile, map_stream,
                map_segment(rows[tile]), cycle, "scatter map");
            ftlpu::StreamPayloadTileSegment bytes{};
            for (std::size_t lane = 0; lane < kLanes; ++lane) {
                bytes[lane] = stream_payload(tile, lane);
            }
            fabric.stage_payload_segment(kInputColumn, tile, data_stream,
                bytes, cycle, "scatter data");
        });
        for (const auto& transfer : mem.consumed_transfers()) {
            if (transfer.kind
                == ftlpu::MemArrayModel::MemTransfer::Kind::StoreStreamToSram) {
                scattered.push_back(transfer.address);
            }
        }
    }

    assert(scattered.size() == kTiles);
    for (std::size_t tile = 0; tile < kTiles; ++tile) {
        assert(scattered[tile] == rows[tile]);
        for (std::size_t lane = 0; lane < kLanes; ++lane) {
            assert(mem.sram_lane_byte(kMemSlice, 0, tile, rows[tile], lane)
                == stream_payload(tile, lane));
        }
    }
}

// Drives one indirect instruction to completion and returns the fault message,
// or an empty string when it completed without faulting. The stage callback
// receives the fabric so each case can build exactly the malformed input it
// targets.
template <typename Prepare, typename StageFn>
std::string fault_message(
    ftlpu::MemInstruction instruction, Prepare prepare, StageFn stage_inputs)
{
    ftlpu::StreamRegisterFabric fabric(
        ftlpu::hw::kMemBoundaryStreamRegisterColumns);
    ftlpu::MemArrayModel mem(
        ftlpu::MemStreamPortMap::BetweenBoundaries(),
        ftlpu::MemArrayModel::MissingStreamPolicy::Error, kReadTiming);
    prepare(mem);
    try {
        for (std::size_t cycle = 0; cycle < kIssueCycle + kTiles + 4; ++cycle) {
            if (cycle == kIssueCycle) {
                mem.enqueue_instruction(kMemSlice, 0, instruction);
            }
            advance(mem, fabric, cycle, [&](std::size_t tile) {
                stage_inputs(fabric, tile, cycle);
            });
        }
    } catch (const std::logic_error& error) {
        return error.what();
    }
    return {};
}

bool contains(const std::string& text, const char* needle)
{
    return text.find(needle) != std::string::npos;
}

void test_missing_map_segment_faults()
{
    const auto message = fault_message(
        ftlpu::MemInstruction::Gather(
            ftlpu::StreamId::East(5), ftlpu::StreamId::East(6)),
        [](ftlpu::MemArrayModel&) {},
        [](ftlpu::StreamRegisterFabric&, std::size_t, std::size_t) {});
    assert(contains(message, "invalid map segment"));
}

void test_map_data_above_the_index_width_faults()
{
    const auto map_stream = ftlpu::StreamId::East(6);
    const auto message = fault_message(
        ftlpu::MemInstruction::Gather(ftlpu::StreamId::East(5), map_stream),
        [](ftlpu::MemArrayModel&) {},
        [&](ftlpu::StreamRegisterFabric& fabric, std::size_t tile,
            std::size_t cycle) {
            auto bytes = map_segment(8);
            bytes[3] = 0x01;
            fabric.stage_payload_segment(
                kInputColumn, tile, map_stream, bytes, cycle, "gather map");
        });
    assert(contains(message, "above the 16-bit row index"));
}

void test_row_index_outside_sram_faults()
{
    const auto map_stream = ftlpu::StreamId::East(6);
    const auto message = fault_message(
        ftlpu::MemInstruction::Gather(ftlpu::StreamId::East(5), map_stream),
        [](ftlpu::MemArrayModel&) {},
        [&](ftlpu::StreamRegisterFabric& fabric, std::size_t tile,
            std::size_t cycle) {
            fabric.stage_payload_segment(kInputColumn, tile, map_stream,
                map_segment(ftlpu::hw::kSramDepthRows), cycle, "gather map");
        });
    assert(contains(message, "outside the configured SRAM"));
}

void test_nonzero_direct_address_faults()
{
    auto instruction = ftlpu::MemInstruction::Gather(
        ftlpu::StreamId::East(5), ftlpu::StreamId::East(6));
    instruction.address = 4;
    const auto map_stream = ftlpu::StreamId::East(6);
    const auto message = fault_message(
        instruction,
        [](ftlpu::MemArrayModel&) {},
        [&](ftlpu::StreamRegisterFabric& fabric, std::size_t tile,
            std::size_t cycle) {
            fabric.stage_payload_segment(kInputColumn, tile, map_stream,
                map_segment(8), cycle, "gather map");
        });
    assert(contains(message, "do not use the direct row address field"));
}

void test_scatter_without_data_segment_faults()
{
    const auto map_stream = ftlpu::StreamId::East(6);
    const auto message = fault_message(
        ftlpu::MemInstruction::Scatter(
            ftlpu::StreamId::East(7), map_stream),
        [](ftlpu::MemArrayModel&) {},
        [&](ftlpu::StreamRegisterFabric& fabric, std::size_t tile,
            std::size_t cycle) {
            fabric.stage_payload_segment(kInputColumn, tile, map_stream,
                map_segment(16), cycle, "scatter map");
        });
    assert(contains(message, "invalid data stream segment"));
}

} // namespace

int main()
{
    test_gather_reads_one_indirect_row_per_tile();
    test_scatter_writes_one_indirect_row_per_tile();
    test_missing_map_segment_faults();
    test_map_data_above_the_index_width_faults();
    test_row_index_outside_sram_faults();
    test_nonzero_direct_address_faults();
    test_scatter_without_data_segment_faults();
    return 0;
}
