#include "ftlpu/system/icu.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ftlpu;

static void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

template <typename Instruction, typename Expand, typename Encode>
static auto timeline(const Instruction& instruction, Expand expand, Encode encode)
{
    using Native = decltype(encode(expand(instruction, IcuCoordinate3D{})));
    std::vector<std::pair<std::size_t, Native>> result;
    IcuCoordinate3D coordinate{};
    do {
        result.emplace_back(detail::icu_loop_3d_issue_cycle(instruction.loop, coordinate),
            encode(expand(instruction, coordinate)));
    } while (detail::advance_icu_coordinate_3d(instruction.loop, coordinate));
    return result;
}

static auto mem_timeline(const MemIcuInstruction& instruction)
{
    return timeline(instruction, detail::expand_mem_icu_instruction, isa::encode_mem_instruction);
}

int main() try
{
    // Every placement of inactive axes, all MEM operations, negative address
    // strides and nontrivial wait. Compare emitted FU words and exact cycles.
    for (unsigned mask = 0; mask < 8; ++mask) {
        IcuLoop3D loop{0, {1, 1, 1}, {2, 7, 23}, 5};
        for (unsigned d = 0; d < 3; ++d) if (mask & (1U << d)) loop.counts[d] = 2;
        for (unsigned op = 0; op < 3; ++op) {
            MemIcuInstruction instruction{static_cast<MemIcuOpcode>(op), loop,
                MemIcuAddress3D::Affine(128, {-1, 8, 32}), StreamId::East(0).packed()};
            const auto packet = isa::encode_mem_icu_shortest_instruction(instruction);
            const auto decoded = isa::decode_mem_icu_3d_instruction(packet);
            require(mem_timeline(instruction) == mem_timeline(decoded), "MEM short codec changed FU timeline");
            const auto expectedWords = mask == 0 ? 1 : mask == 7 ? 3 : 2;
            require(isa::fu_loop_packet_word_count(packet) == expectedWords, "wrong MEM physical length");
            const auto relocated = isa::patch_mem_icu_base_address(packet, 64);
            auto relocatedExpected = instruction;
            relocatedExpected.address.base_address = 64;
            require(isa::fu_loop_packet_word_count(relocated) == isa::fu_loop_packet_word_count(packet),
                "MEM relocation changed physical packet length");
            require(mem_timeline(isa::decode_mem_icu_3d_instruction(relocated)) == mem_timeline(relocatedExpected),
                "MEM relocation failed to replace old address bits");
            auto invalid = packet;
            invalid.words[0].lanes[0] ^= 0x80U;
            bool rejected = false;
            try { (void)isa::decode_mem_icu_3d_instruction(invalid); }
            catch (const std::exception&) { rejected = true; }
            require(rejected, "short MEM accepted invalid header version");
        }

        for (unsigned terminal = 0; terminal < 4; ++terminal) {
            MxmComputeIcuInstruction instruction{};
            instruction.loop = loop;
            instruction.weight_buffer_mode = MxmIcuBufferMode::ToggleDimension2;
            instruction.accumulator_address_strides = {0, 8, 32};
            instruction.terminal_dimension = terminal;
            instruction.terminal_mode = {MxmAccumulatorDestination::Stream, true, MxmAccumulatorOutputFormat::BFloat16};
            const auto decoded = isa::decode_mxm_compute_icu_3d_instruction(
                isa::encode_mxm_compute_icu_shortest_instruction(instruction));
            require(timeline(instruction, detail::expand_mxm_compute_icu_instruction, isa::encode_mxm_instruction)
                    == timeline(decoded, detail::expand_mxm_compute_icu_instruction, isa::encode_mxm_instruction),
                "MXM short codec changed parity/terminal/accumulator behavior");
        }
        MxmDequantIcuInstruction dequant{loop, MxmDequantInstruction::ScaleBits(0x3f80)};
        const auto decodedDequant = isa::decode_mxm_dequant_icu_3d_instruction(
            isa::encode_mxm_dequant_icu_shortest_instruction(dequant));
        const auto scale = [](const MxmDequantIcuInstruction& i, IcuCoordinate3D) { return i.instruction; };
        require(timeline(dequant, scale, isa::encode_mxm_dequant_instruction)
                == timeline(decodedDequant, scale, isa::encode_mxm_dequant_instruction),
            "MXM dequant short codec changed scale/cycles");
        const auto scalePacket = isa::encode_mxm_dequant_icu_shortest_instruction(dequant);
        const auto rescaled = isa::patch_mxm_dequant_icu_scale(scalePacket, MxmDequantInstruction::ScaleBits(0x3f00));
        require(isa::fu_loop_packet_word_count(rescaled) == isa::fu_loop_packet_word_count(scalePacket)
                && isa::encode_mxm_dequant_instruction(isa::decode_mxm_dequant_icu_3d_instruction(rescaled).instruction) == 0x3f00,
            "MXM relocation changed length or failed to clear previous scale bits");
        MxmComputeIcuInstruction decodeCompute{};
        decodeCompute.opcode = MxmComputeIcuOpcode::DecodeStreamCompute3D;
        decodeCompute.loop = loop;
        decodeCompute.accumulator_address_strides = {1, 8, 32};
        decodeCompute.weight_buffer_mode = MxmIcuBufferMode::ToggleDimension2;
        const auto decodedCompute = isa::decode_mxm_compute_icu_3d_instruction(
            isa::encode_mxm_compute_icu_shortest_instruction(decodeCompute));
        require(timeline(decodeCompute, detail::expand_mxm_compute_icu_instruction, isa::encode_mxm_instruction)
                == timeline(decodedCompute, detail::expand_mxm_compute_icu_instruction, isa::encode_mxm_instruction),
            "decode MXM short codec changed FU words");
        MxmLoadIcuInstruction load{};
        load.loop = loop;
        load.weight_column_strides = {0, 0, 0};
        load.weight_buffer_mode = MxmIcuBufferMode::ToggleDimension1;
        const auto decoded = isa::decode_mxm_load_icu_3d_instruction(isa::encode_mxm_load_icu_shortest_instruction(load));
        require(timeline(load, detail::expand_mxm_load_icu_instruction, isa::encode_mxm_instruction)
                == timeline(decoded, detail::expand_mxm_load_icu_instruction, isa::encode_mxm_instruction),
            "MXM load short codec changed buffer parity");
    }

    const auto foldable = MemIcuInstruction::Read3D({0, {4, 2, 3}, {1, 4, 8}, 0},
        MemIcuAddress3D::Affine(0, {1, 4, 8}), StreamId::East(0));
    const auto folded = isa::encode_mem_icu_shortest_instruction(foldable);
    require(isa::short_icu_rank(folded.words[0]) == 1, "affine axes were not folded");
    require(mem_timeline(foldable) == mem_timeline(isa::decode_mem_icu_3d_instruction(folded)), "fold changed timeline");

    const auto blocked = MemIcuInstruction::Read3D({0, {2, 2, 3}, {1, 5, 17}, 0},
        MemIcuAddress3D::BlockedOuter(0, 1, 8, 2, 32, 128), StreamId::East(0));
    const auto blockedPacket = isa::encode_mem_icu_shortest_instruction(blocked);
    require(isa::short_icu_rank(blockedPacket.words[0]) < 0, "non-affine blocked domain lost its 3D encoding");
    require(mem_timeline(blocked) == mem_timeline(isa::decode_mem_icu_3d_instruction(blockedPacket)), "blocked timeline changed");

    // Real finite iMEM/IQ decode: mixed 1-, 2-, 3-word instructions.
    InstructionControlUnit icu;
    auto& queue = icu.mem_iq(0);
    const auto singleton = MemIcuInstruction::Write3D({}, MemIcuAddress3D::Affine(512, {}), StreamId::West(0));
    queue.push_encoded_3d_packet(isa::encode_mem_icu_shortest_instruction(singleton));
    queue.push_encoded_3d_packet(folded);
    queue.push_encoded_3d_packet(blockedPacket);
    require(queue.imem_occupancy() == 6, "short packet padding entered physical iMEM");
    std::vector<std::uint64_t> expected, actual;
    for (const auto& instruction : {singleton, foldable, blocked})
        for (const auto& [cycle, native] : mem_timeline(instruction)) expected.push_back(native);
    for (unsigned cycle = 0; cycle < 200 && !queue.done(); ++cycle)
        if (const auto issued = queue.tick()) actual.push_back(isa::encode_mem_instruction(*issued));
    require(queue.done() && actual == expected, "finite ICU failed mixed short/3D execution");
    require(queue.peak_active_3d_contexts() == 1, "short packets introduced interleaved contexts");
    std::cout << "short ICU codecs and sequential physical execution passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
