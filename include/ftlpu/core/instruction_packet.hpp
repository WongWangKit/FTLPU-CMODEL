#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/instruction_codec.hpp"

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
        return static_cast<InstructionPacketKind>(packet.bytes[0]);
    }
    throw std::logic_error("instruction packet has an unknown kind");
}

inline EncodedInstructionPacket encode_packet(const MemInstruction& instruction)
{
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

inline EncodedInstructionPacket encode_packet(const IcuControlInstruction& instruction)
{
    auto packet = packet_detail::make_packet(InstructionPacketKind::IcuControl, 4);
    packet_detail::write_u32_le(packet, 4, encode_icu_control_instruction(instruction));
    return packet;
}

inline MemInstruction decode_mem_packet(const EncodedInstructionPacket& packet)
{
    packet_detail::require_packet(packet, InstructionPacketKind::Mem, 4);
    return decode_mem_instruction(packet_detail::read_u32_le(packet, 4));
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

inline IcuControlInstruction decode_icu_packet(const EncodedInstructionPacket& packet)
{
    packet_detail::require_packet(packet, InstructionPacketKind::IcuControl, 4);
    return decode_icu_control_instruction(packet_detail::read_u32_le(packet, 4));
}

} // namespace ftlpu::isa
