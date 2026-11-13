#include "ftlpu/icu/sxm_macro_instruction.hpp"
#include "ftlpu/mem/mem_array.hpp"
#include "ftlpu/sxm/slice.hpp"

#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
ftlpu::SxmInstruction::StreamList streams(std::size_t base)
{
    ftlpu::SxmInstruction::StreamList result;
    for (std::size_t i = 0; i < 2 * ftlpu::hw::kLanesPerTile; ++i)
        result.push_back({base + i});
    return result;
}
}

int main() try
{
    using namespace ftlpu;
    const auto ports = MemStreamPortMap::BetweenBoundaries();
    for (std::size_t slice = 0; slice < hw::kMemSliceColumns; ++slice) {
        const auto group = slice / hw::kMemSlicesPerGroup;
        require(ports.input_column(slice, StreamDirection::East) == group,
            "MEM grouping changed with physical tile count");
        require(ports.output_column(slice, StreamDirection::West) == group,
            "reverse MEM grouping changed with physical tile count");
    }
    SxmInstruction::PermuteMap map{};
    SxmSlice::StreamVector<int> input{};
    for (std::size_t i = 0; i < map.size(); ++i) {
        map[i] = map.size() - 1 - i;
        input[i / hw::kLanesPerTile][i % hw::kLanesPerTile] = static_cast<int>(i);
    }
    auto instruction = SxmInstruction::Permute(streams(0), streams(16), map);
    instruction.output_tile = hw::kTileRows - 1;
    using Codec = IcuSxmActiveMacroContextCodec<false>;
    IcuMacroSchedule schedule{0,3,2,0,2,13,0,IcuInductionTarget::None};
    const auto packet = Codec::pack_initial(schedule,instruction);
    auto decoded = Codec::unpack(packet);
    require(decoded.instruction.permute_map == map
            && decoded.instruction.output_tile == hw::kTileRows - 1,
        "SXM Macro lost high lane indices or last tile");
    const auto permuted = SxmSlice::permute(input, decoded.instruction.permute_map);
    for (std::size_t i = 0; i < map.size(); ++i)
        require(permuted[i / hw::kLanesPerTile][i % hw::kLanesPerTile]
                    == static_cast<int>(map[i]), "SXM physical permutation mismatch");
    instruction.output_tile = SxmInstruction::kAllOutputTiles;
    decoded = Codec::unpack(Codec::pack_initial(schedule,instruction));
    require(decoded.instruction.output_tile == SxmInstruction::kAllOutputTiles,
        "SXM all-tiles sentinel did not round trip");
    const auto raw = isa::decode_sxm_instruction(isa::encode_sxm_instruction(instruction));
    require(raw.permute_map == map && raw.output_tile == instruction.output_tile,
        "legacy SXM codec lost widened permutation or sentinel");
    instruction.permute_map.back() = SxmInstruction::kTotalLanes;
    bool rejected = false;
    try { (void)Codec::pack_initial(schedule,instruction); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "SXM accepted an out-of-range source lane");
    if constexpr (hw::kPhysicalVectorBytes == 32)
        require(Codec::bit_count==384,"LPU32 permute context exceeds 384 bits");
    if constexpr (hw::kPhysicalVectorBytes == 64) {
        require(hw::kMemSliceColumns == 52 && hw::kMemSlicesPerGroup == 4
            && hw::kMemGroups == 13 && hw::kSystemStreamRegisterColumns == 16,
            "8-tile target changed the MEM/SR topology");
    }
    std::cout << "geometry_encoding_test passed: tiles=" << hw::kTileRows
        << " mem_group=" << hw::kMemSlicesPerGroup << " lanes=" << map.size()
        << " sxm_words=" << Codec::word_count << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
