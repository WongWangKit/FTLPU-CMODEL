#include "ftlpu/icu/fetch.hpp"
#include "ftlpu/mem/mem_array.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace {

ftlpu::StreamPayloadVector320 vector_pattern(std::uint8_t seed)
{
    ftlpu::StreamPayloadVector320 vector{};
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            vector[tile][lane] = static_cast<std::uint8_t>(seed + tile * 7 + lane);
        }
    }
    return vector;
}

ftlpu::StreamSegment16 make_segment(
    const ftlpu::StreamPayloadSegment16& bytes,
    std::uint64_t tag)
{
    ftlpu::StreamSegment16 segment{};
    for (std::size_t lane = 0; lane < segment.size(); ++lane) {
        segment[lane] = ftlpu::StreamCell::Valid(bytes[lane], lane == 15, tag);
    }
    return segment;
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
    auto memory = std::make_unique<ftlpu::MemArrayModel>();
    auto fabric = std::make_unique<ftlpu::StreamRegisterFabric>(
        ftlpu::hw::kMemBoundaryStreamRegisterColumns);
    const auto first = vector_pattern(3);
    const auto second = vector_pattern(97);
    memory->write_sram_vector(0, ftlpu::MemLocalWordAddress13(0), first);
    memory->write_sram_vector(0, ftlpu::MemLocalWordAddress13(1), second);
    memory->enqueue_instruction(0, ftlpu::MemInstruction::Read(0, ftlpu::StreamId::East(2)));
    memory->enqueue_instruction(0, ftlpu::MemInstruction::Read(1, ftlpu::StreamId::East(2)));

    std::array<std::optional<std::uint64_t>, 2> tags{};
    std::array<std::size_t, 2> tile_counts{};
    for (std::size_t cycle = 0; cycle < ftlpu::hw::kTileRows + 1; ++cycle) {
        fabric->begin_cycle();
        memory->evaluate(*fabric);
        for (const auto& transfer : memory->executed_transfers()) {
            if (transfer.kind != ftlpu::MemArrayModel::MemTransfer::Kind::LoadSramToStream) {
                continue;
            }
            const auto vector = transfer.address.word();
            assert(vector < 2);
            if (!tags[vector].has_value()) tags[vector] = transfer.vector_tag;
            assert(transfer.vector_tag == *tags[vector]);
            assert(transfer.bytes == (vector == 0 ? first[transfer.tile] : second[transfer.tile]));
            ++tile_counts[vector];
        }
        fabric->commit_cycle();
    }
    assert(tags[0].has_value() && tags[1].has_value());
    assert(*tags[0] != *tags[1]);
    assert(tile_counts[0] == ftlpu::hw::kTileRows);
    assert(tile_counts[1] == ftlpu::hw::kTileRows);

    // FetchBuffer accepts interleaved arrival from two stable tags and
    // reconstructs the original two 320-byte vectors byte-for-byte.
    ftlpu::IcuFetchBuffer buffer;
    buffer.reset();
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        buffer.accept_segment(tile, make_segment(first[tile], *tags[0]));
        buffer.accept_segment(tile, make_segment(second[tile], *tags[1]));
    }
    assert(buffer.complete());
    const auto packets = buffer.packets();
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            assert(packets[tile].bytes[lane] == first[tile][lane]);
            assert(packets[ftlpu::hw::kTileRows + tile].bytes[lane] == second[tile][lane]);
        }
    }

    ftlpu::IcuFetchBuffer invalid;
    invalid.reset();
    const auto segment = make_segment(first[0], 10);
    invalid.accept_segment(0, segment);
    assert(throws([&] { invalid.accept_segment(0, segment); }));
    assert(throws([&] { (void)invalid.packets(); }));

    ftlpu::IcuFetchBuffer mixed;
    mixed.reset();
    auto mixed_segment = make_segment(first[0], 20);
    mixed_segment[7].vector_tag = 21;
    assert(throws([&] { mixed.accept_segment(0, mixed_segment); }));

    ftlpu::IcuFetchBuffer third_tag;
    third_tag.reset();
    third_tag.accept_segment(0, make_segment(first[0], 30));
    third_tag.accept_segment(1, make_segment(first[1], 31));
    assert(throws([&] { third_tag.accept_segment(2, make_segment(first[2], 32)); }));

    return 0;
}
