#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/sxm/instruction.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ftlpu::isa {

enum class InstructionPacketKind : std::uint8_t {
    Mem = 1,
    Mxm = 2,
    Vxm = 3,
    IcuControl = 4,
    Sxm = 5,
};

// Stable 16-byte little-endian program representation.  bytes[0..3] are an
// explicit packet header; bytes[4..15] hold a 4- or 12-byte ISA payload.
struct EncodedInstructionPacket {
    std::array<std::uint8_t, hw::kEncodedInstructionPacketBytes> bytes{};

    friend bool operator==(const EncodedInstructionPacket&, const EncodedInstructionPacket&) = default;
};

static_assert(sizeof(EncodedInstructionPacket) == hw::kEncodedInstructionPacketBytes);

namespace packet_detail {

constexpr std::size_t kHeaderBytes = 4;

inline void write_u32_le(
    EncodedInstructionPacket& packet,
    std::size_t offset,
    std::uint32_t value)
{
    if (offset + 4 > packet.bytes.size()) {
        throw std::out_of_range("instruction packet u32 write exceeds 16 bytes");
    }
    for (std::size_t byte = 0; byte < 4; ++byte) {
        packet.bytes[offset + byte] = static_cast<std::uint8_t>(value >> (8 * byte));
    }
}

inline std::uint32_t read_u32_le(
    const EncodedInstructionPacket& packet,
    std::size_t offset)
{
    if (offset + 4 > packet.bytes.size()) {
        throw std::out_of_range("instruction packet u32 read exceeds 16 bytes");
    }
    std::uint32_t value = 0;
    for (std::size_t byte = 0; byte < 4; ++byte) {
        value |= static_cast<std::uint32_t>(packet.bytes[offset + byte]) << (8 * byte);
    }
    return value;
}

inline EncodedInstructionPacket make_packet(
    InstructionPacketKind kind,
    std::uint8_t payload_bytes)
{
    EncodedInstructionPacket packet{};
    packet.bytes[0] = static_cast<std::uint8_t>(kind);
    packet.bytes[1] = payload_bytes;
    return packet;
}

inline void require_packet(
    const EncodedInstructionPacket& packet,
    InstructionPacketKind expected,
    std::uint8_t payload_bytes)
{
    if (packet.bytes[0] != static_cast<std::uint8_t>(expected)
        || packet.bytes[1] != payload_bytes
        || packet.bytes[2] != 0
        || packet.bytes[3] != 0) {
        throw std::logic_error("instruction packet header does not match decoder");
    }
    const auto payload_end = kHeaderBytes + payload_bytes;
    for (std::size_t byte = payload_end; byte < packet.bytes.size(); ++byte) {
        if (packet.bytes[byte] != 0) {
            throw std::logic_error("instruction packet has non-zero padding bytes");
        }
    }
}

} // namespace packet_detail

inline InstructionPacketKind packet_kind(const EncodedInstructionPacket& packet)
{
    switch (static_cast<InstructionPacketKind>(packet.bytes[0])) {
    case InstructionPacketKind::Mem:
    case InstructionPacketKind::Mxm:
    case InstructionPacketKind::Vxm:
    case InstructionPacketKind::IcuControl:
    case InstructionPacketKind::Sxm:
        return static_cast<InstructionPacketKind>(packet.bytes[0]);
    }
    throw std::logic_error("instruction packet has an unknown kind");
}

inline EncodedInstructionPacket encode_packet(const MemInstruction& instruction)
{
    if (instruction.opcode == MemOpcode::ReadWrite
        || instruction.opcode == MemOpcode::Accumulate) {
        auto packet =
            packet_detail::make_packet(InstructionPacketKind::Mem, 12);
        const auto encoded = encode_extended_mem_instruction(instruction);
        for (std::size_t word = 0; word < encoded.words.size(); ++word) {
            packet_detail::write_u32_le(
                packet, 4 + word * 4, encoded.words[word]);
        }
        return packet;
    }
    auto packet = packet_detail::make_packet(InstructionPacketKind::Mem, 4);
    packet_detail::write_u32_le(packet, 4, encode_mem_instruction(instruction));
    return packet;
}

inline EncodedInstructionPacket encode_packet(const MxmControlInstruction& instruction)
{
    auto packet = packet_detail::make_packet(InstructionPacketKind::Mxm, 4);
    packet_detail::write_u32_le(packet, 4, encode_mxm_instruction(instruction));
    return packet;
}

inline EncodedInstructionPacket encode_packet(const VxmLaneAluInstruction& instruction)
{
    auto packet = packet_detail::make_packet(InstructionPacketKind::Vxm, 12);
    const auto encoded = encode_vxm_instruction(instruction);
    for (std::size_t word = 0; word < encoded.words.size(); ++word) {
        packet_detail::write_u32_le(packet, 4 + word * 4, encoded.words[word]);
    }
    return packet;
}

namespace packet_detail {

inline std::pair<std::size_t, std::size_t> compact_stream_range(
    const SxmInstruction::StreamList& streams,
    const char* field)
{
    if (streams.empty() || streams.size() > 32) {
        throw std::logic_error(field);
    }
    const auto first = streams.front().stream;
    if (first >= hw::kStreams
        || first + streams.size() > hw::kStreams) {
        throw std::out_of_range(field);
    }
    for (std::size_t index = 0; index < streams.size(); ++index) {
        if (streams[index].stream != first + index) {
            throw std::logic_error(
                "SXM packet encoding requires contiguous stream lists");
        }
    }
    return {first, streams.size()};
}

inline bool block_diagonal_permute(
    const SxmInstruction::PermuteMap& map,
    std::size_t diagonal)
{
    for (std::size_t destination = 0;
         destination < hw::kTileRows;
         ++destination) {
        const auto source =
            (diagonal + hw::kTileRows - destination)
            % hw::kTileRows;
        for (std::size_t lane = 0;
             lane < hw::kLanesPerTile;
             ++lane) {
            if (map[destination * hw::kLanesPerTile + lane]
                != source * hw::kLanesPerTile + lane) {
                return false;
            }
        }
    }
    return true;
}

inline SxmInstruction::StreamList compact_stream_list(
    std::size_t first,
    std::size_t count)
{
    auto streams = SxmInstruction::StreamList{};
    streams.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        streams.push_back(SxmStreamId{first + index});
    }
    return streams;
}

} // namespace packet_detail

// The fixed 16-byte packet supports the transformer schedule's compact SXM
// forms: contiguous stream ranges, arbitrary non-zero-fill Distribute maps,
// and block-diagonal Permute maps. A fully arbitrary 320-entry permutation
// deliberately fails encoding instead of being hidden in a side table.
inline EncodedInstructionPacket encode_packet(const SxmInstruction& instruction)
{
    const auto [src_base, src_count] =
        packet_detail::compact_stream_range(
            instruction.src_streams,
            "SXM source stream range is not packet encodable");
    const auto [dst_base, dst_count] =
        packet_detail::compact_stream_range(
            instruction.dst_streams,
            "SXM destination stream range is not packet encodable");
    if (instruction.shift_distance > 31) {
        throw std::out_of_range(
            "SXM shift distance exceeds the compact packet field");
    }
    auto words = std::array<std::uint32_t, 3>{};
    words[0] =
        static_cast<std::uint32_t>(instruction.opcode)
        | (static_cast<std::uint32_t>(instruction.shift_source) << 2)
        | (static_cast<std::uint32_t>(instruction.shift_distance) << 4)
        | (static_cast<std::uint32_t>(src_base) << 9)
        | (static_cast<std::uint32_t>(dst_base) << 15)
        | (static_cast<std::uint32_t>(src_count - 1) << 21)
        | (static_cast<std::uint32_t>(dst_count - 1) << 26);

    if (instruction.opcode == SxmOpcode::Distribute) {
        if (src_count != 1 || dst_count != 1) {
            throw std::logic_error(
                "SXM Distribute packet requires one source and destination");
        }
        std::uint64_t packed_map = 0;
        for (std::size_t lane = 0;
             lane < hw::kLanesPerTile;
             ++lane) {
            if (instruction.lane_map[lane] >= hw::kLanesPerTile
                || instruction.lane_map[lane] > 15) {
                throw std::logic_error(
                    "SXM packet cannot encode zero-fill/out-of-range lane maps");
            }
            packed_map |=
                static_cast<std::uint64_t>(instruction.lane_map[lane])
                << (lane * 4);
        }
        words[1] = static_cast<std::uint32_t>(packed_map);
        words[2] = static_cast<std::uint32_t>(packed_map >> 32);
    } else if (instruction.opcode == SxmOpcode::Permute) {
        auto diagonal = hw::kTileRows;
        for (std::size_t candidate = 0;
             candidate < hw::kTileRows;
             ++candidate) {
            if (packet_detail::block_diagonal_permute(
                    instruction.permute_map, candidate)) {
                diagonal = candidate;
                break;
            }
        }
        if (diagonal == hw::kTileRows) {
            throw std::logic_error(
                "SXM packet supports block-diagonal Permute maps only");
        }
        words[1] = static_cast<std::uint32_t>(diagonal);
    }

    auto packet =
        packet_detail::make_packet(InstructionPacketKind::Sxm, 12);
    for (std::size_t word = 0; word < words.size(); ++word) {
        packet_detail::write_u32_le(
            packet, 4 + word * 4, words[word]);
    }
    return packet;
}

inline EncodedInstructionPacket encode_packet(const IcuControlInstruction& instruction)
{
    auto packet = packet_detail::make_packet(InstructionPacketKind::IcuControl, 4);
    packet_detail::write_u32_le(packet, 4, encode_icu_control_instruction(instruction));
    return packet;
}

inline MemInstruction decode_mem_packet(const EncodedInstructionPacket& packet)
{
    if (packet.bytes[1] == 4) {
        packet_detail::require_packet(
            packet, InstructionPacketKind::Mem, 4);
        return decode_mem_instruction(
            packet_detail::read_u32_le(packet, 4));
    }
    if (packet.bytes[1] == 12) {
        packet_detail::require_packet(
            packet, InstructionPacketKind::Mem, 12);
        EncodedExtendedMemInstruction encoded{};
        for (std::size_t word = 0; word < encoded.words.size(); ++word) {
            encoded.words[word] =
                packet_detail::read_u32_le(packet, 4 + word * 4);
        }
        return decode_extended_mem_instruction(encoded);
    }
    throw std::logic_error(
        "MEM instruction packet has an unsupported payload size");
}

inline MxmControlInstruction decode_mxm_packet(const EncodedInstructionPacket& packet)
{
    packet_detail::require_packet(packet, InstructionPacketKind::Mxm, 4);
    return decode_mxm_instruction(packet_detail::read_u32_le(packet, 4));
}

inline VxmLaneAluInstruction decode_vxm_packet(const EncodedInstructionPacket& packet)
{
    packet_detail::require_packet(packet, InstructionPacketKind::Vxm, 12);
    EncodedVxmInstruction encoded{};
    for (std::size_t word = 0; word < encoded.words.size(); ++word) {
        encoded.words[word] = packet_detail::read_u32_le(packet, 4 + word * 4);
    }
    return decode_vxm_instruction(encoded);
}

inline SxmInstruction decode_sxm_packet(
    const EncodedInstructionPacket& packet)
{
    packet_detail::require_packet(
        packet, InstructionPacketKind::Sxm, 12);
    const auto control = packet_detail::read_u32_le(packet, 4);
    const auto word1 = packet_detail::read_u32_le(packet, 8);
    const auto word2 = packet_detail::read_u32_le(packet, 12);
    const auto opcode = static_cast<SxmOpcode>(control & 0x3u);
    const auto shift_source =
        static_cast<SxmShiftSource>((control >> 2) & 0x3u);
    const auto shift_distance =
        static_cast<std::size_t>((control >> 4) & 0x1fu);
    const auto src_base =
        static_cast<std::size_t>((control >> 9) & 0x3fu);
    const auto dst_base =
        static_cast<std::size_t>((control >> 15) & 0x3fu);
    const auto src_count =
        static_cast<std::size_t>((control >> 21) & 0x1fu) + 1;
    const auto dst_count =
        static_cast<std::size_t>((control >> 26) & 0x1fu) + 1;
    if (src_base + src_count > hw::kStreams
        || dst_base + dst_count > hw::kStreams) {
        throw std::logic_error(
            "encoded SXM stream range exceeds the stream set");
    }
    auto srcs =
        packet_detail::compact_stream_list(src_base, src_count);
    auto dsts =
        packet_detail::compact_stream_list(dst_base, dst_count);

    switch (opcode) {
    case SxmOpcode::Distribute: {
        if (src_count != 1 || dst_count != 1) {
            throw std::logic_error(
                "encoded SXM Distribute has invalid stream counts");
        }
        const auto packed =
            static_cast<std::uint64_t>(word1)
            | (static_cast<std::uint64_t>(word2) << 32);
        auto map = SxmInstruction::LaneMap{};
        for (std::size_t lane = 0;
             lane < hw::kLanesPerTile;
             ++lane) {
            map[lane] =
                static_cast<std::size_t>(
                    (packed >> (lane * 4)) & 0xfu);
            if (map[lane] >= hw::kLanesPerTile) {
                throw std::logic_error(
                    "encoded SXM lane map exceeds configured lanes");
            }
        }
        return SxmInstruction::Distribute(
            srcs.front(), dsts.front(), map);
    }
    case SxmOpcode::Transpose:
        if (word1 != 0 || word2 != 0) {
            throw std::logic_error(
                "encoded SXM Transpose has non-zero map words");
        }
        return SxmInstruction::Transpose(
            std::move(srcs), std::move(dsts));
    case SxmOpcode::ShiftSelect:
        if (word1 != 0 || word2 != 0
            || static_cast<std::size_t>(shift_source)
                > static_cast<std::size_t>(
                    SxmShiftSource::SouthShifted)) {
            throw std::logic_error(
                "encoded SXM ShiftSelect has invalid fields");
        }
        return SxmInstruction::ShiftSelect(
            std::move(srcs),
            std::move(dsts),
            shift_source,
            shift_distance);
    case SxmOpcode::Permute: {
        if (word1 >= hw::kTileRows || word2 != 0) {
            throw std::logic_error(
                "encoded SXM block diagonal is invalid");
        }
        auto map = SxmInstruction::PermuteMap{};
        for (std::size_t destination = 0;
             destination < hw::kTileRows;
             ++destination) {
            const auto source =
                (word1 + hw::kTileRows - destination)
                % hw::kTileRows;
            for (std::size_t lane = 0;
                 lane < hw::kLanesPerTile;
                 ++lane) {
                map[destination * hw::kLanesPerTile + lane] =
                    source * hw::kLanesPerTile + lane;
            }
        }
        return SxmInstruction::Permute(
            std::move(srcs), std::move(dsts), map);
    }
    }
    throw std::logic_error("encoded SXM opcode is invalid");
}

inline IcuControlInstruction decode_icu_packet(const EncodedInstructionPacket& packet)
{
    packet_detail::require_packet(packet, InstructionPacketKind::IcuControl, 4);
    return decode_icu_control_instruction(packet_detail::read_u32_le(packet, 4));
}

} // namespace ftlpu::isa
