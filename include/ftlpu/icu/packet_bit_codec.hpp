#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

// Shared packet payload access without imposing MEM/MXM codec geometry.
namespace ftlpu::isa::fu_3d_codec_detail {

constexpr std::size_t kExtendedSubtypePhysicalOffset = 88;
constexpr unsigned kExtendedSubtypeBits = 4;

struct PacketBitLocation {
    std::size_t word{0};
    std::size_t lane{0};
    unsigned bit{0};
};

template <typename Packet>
constexpr std::size_t packet_payload_bits() noexcept
{
    return Packet::kPayloadBitsPerWord * Packet::kWordCount
        - kExtendedSubtypeBits;
}

template <typename Packet>
PacketBitLocation packet_bit_location(std::size_t bit)
{
    if (bit >= packet_payload_bits<Packet>())
        throw std::logic_error("invalid FU 3-D codec payload bit");
    auto remaining = bit;
    for (std::size_t word = 0; word < Packet::kWordCount; ++word) {
        for (std::size_t physical_bit = Packet::kHeaderBits;
             physical_bit < Packet::kWordBits; ++physical_bit) {
            if (word == 0
                && physical_bit >= kExtendedSubtypePhysicalOffset
                && physical_bit < kExtendedSubtypePhysicalOffset
                        + kExtendedSubtypeBits)
                continue;
            if (remaining == 0)
                return {word, physical_bit / 32,
                    static_cast<unsigned>(physical_bit % 32)};
            --remaining;
        }
    }
    throw std::logic_error("invalid FU 3-D codec payload mapping");
}

template <typename Packet>
void write_unsigned(Packet& packet, std::size_t offset,
    unsigned width, std::uint64_t value, const char* field)
{
    if (width == 0 || width > 63
        || offset + width > packet_payload_bits<Packet>())
        throw std::logic_error("invalid FU 3-D codec field layout");
    const auto maximum = (std::uint64_t {1} << width) - 1;
    if (value > maximum) throw std::out_of_range(field);
    for (unsigned bit = 0; bit < width; ++bit) {
        if (((value >> bit) & 1U) == 0) continue;
        const auto position = offset + bit;
        const auto location = packet_bit_location<Packet>(position);
        packet.words[location.word].lanes[location.lane]
            |= std::uint32_t {1} << location.bit;
    }
}

template <typename Packet>
std::uint64_t read_unsigned(
    const Packet& packet, std::size_t offset, unsigned width)
{
    if (width == 0 || width > 63
        || offset + width > packet_payload_bits<Packet>())
        throw std::logic_error("invalid FU 3-D codec field layout");
    std::uint64_t value = 0;
    for (unsigned bit = 0; bit < width; ++bit) {
        const auto position = offset + bit;
        const auto location = packet_bit_location<Packet>(position);
        if (((packet.words[location.word].lanes[location.lane]
                 >> location.bit)
                & 1U)
            != 0)
            value |= std::uint64_t {1} << bit;
    }
    return value;
}

template <typename Packet>
void write_signed(Packet& packet, std::size_t offset,
    unsigned width, std::int64_t value, const char* field)
{
    if (width < 2 || width > 63)
        throw std::logic_error("invalid FU 3-D signed field width");
    const auto minimum = -(std::int64_t {1} << (width - 1));
    const auto maximum = (std::int64_t {1} << (width - 1)) - 1;
    if (value < minimum || value > maximum) throw std::out_of_range(field);
    const auto mask = (std::uint64_t {1} << width) - 1;
    write_unsigned(packet, offset, width,
        static_cast<std::uint64_t>(value) & mask, field);
}

template <typename Packet>
std::int64_t read_signed(
    const Packet& packet, std::size_t offset, unsigned width)
{
    const auto raw = read_unsigned(packet, offset, width);
    const auto sign = std::uint64_t {1} << (width - 1);
    if ((raw & sign) == 0) return static_cast<std::int64_t>(raw);
    const auto mask = (std::uint64_t {1} << width) - 1;
    return -1 - static_cast<std::int64_t>((~raw) & mask);
}

template <typename Packet>
void require_reserved_zero(
    const Packet& packet, std::size_t used_bits, const char* message)
{
    for (auto bit = used_bits; bit < packet_payload_bits<Packet>(); ++bit) {
        if (read_unsigned(packet, bit, 1) != 0)
            throw std::logic_error(message);
    }
}

} // namespace ftlpu::isa::fu_3d_codec_detail
