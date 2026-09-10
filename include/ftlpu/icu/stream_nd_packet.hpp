#pragma once

#include "ftlpu/icu/instruction.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ftlpu {

// Fixed target packet consumed by a local MEM/MXM ICU decoder.  The packet is
// ten 32-bit transfer words regardless of rank, so instruction fetch does not
// depend on a host-side extension-word envelope.
enum class IcuStreamNdUnit : std::uint8_t {
    Mem = 0,
    MxmLoad = 1,
    MxmCompute = 2,
    MxmDequant = 3,
};

struct IcuStreamNdPacket {
    static constexpr std::size_t kWordCount = 10;
    static constexpr std::size_t kBitCount = kWordCount * 32;
    static constexpr std::uint8_t kOpcode = 8;
    static constexpr std::uint8_t kVersion = 0;

    std::array<std::uint32_t, kWordCount> words{};
};

struct IcuDecodedStreamNdPacket {
    IcuStreamNdUnit unit{IcuStreamNdUnit::Mem};
    IcuStreamNdSchedule schedule{};
    std::uint64_t native_instruction{0};
};

namespace detail {

inline constexpr std::uint64_t icu_stream_nd_mask(unsigned width)
{
    return width == 64
        ? std::numeric_limits<std::uint64_t>::max()
        : (std::uint64_t {1} << width) - 1;
}

inline void icu_stream_nd_put(IcuStreamNdPacket& packet,
    std::size_t bit, unsigned width, std::uint64_t value)
{
    if (width == 0 || width > 64 || bit + width > packet.kBitCount
        || (width != 64 && value > icu_stream_nd_mask(width)))
        throw std::invalid_argument("ICU STREAM_ND field does not fit packet");
    for (unsigned offset = 0; offset < width; ++offset) {
        if (((value >> offset) & 1u) != 0)
            packet.words[(bit + offset) / 32]
                |= std::uint32_t {1} << ((bit + offset) % 32);
    }
}

inline std::uint64_t icu_stream_nd_get(const IcuStreamNdPacket& packet,
    std::size_t bit, unsigned width)
{
    if (width == 0 || width > 64 || bit + width > packet.kBitCount)
        throw std::invalid_argument("ICU STREAM_ND field is outside packet");
    std::uint64_t value = 0;
    for (unsigned offset = 0; offset < width; ++offset) {
        if (((packet.words[(bit + offset) / 32]
                 >> ((bit + offset) % 32))
                & 1u)
            != 0)
            value |= std::uint64_t {1} << offset;
    }
    return value;
}

inline bool icu_stream_nd_fits_signed(std::int64_t value, unsigned width)
{
    const auto limit = std::int64_t {1} << (width - 1);
    return value >= -limit && value < limit;
}

inline std::int64_t icu_stream_nd_sign_extend(
    std::uint64_t value, unsigned width)
{
    const auto sign = std::uint64_t {1} << (width - 1);
    if ((value & sign) == 0) return static_cast<std::int64_t>(value);
    return static_cast<std::int64_t>(value | ~icu_stream_nd_mask(width));
}

inline void validate_icu_stream_nd_unit(
    IcuStreamNdUnit unit, IcuInductionTarget target)
{
    switch (unit) {
    case IcuStreamNdUnit::Mem:
        if (target != IcuInductionTarget::MemAddress)
            throw std::invalid_argument(
                "MEM STREAM_ND packet requires MEM-address induction");
        return;
    case IcuStreamNdUnit::MxmLoad:
        if (target != IcuInductionTarget::None
            && target != IcuInductionTarget::MxmWeightColumn)
            throw std::invalid_argument(
                "MXM-load STREAM_ND packet has an invalid induction target");
        return;
    case IcuStreamNdUnit::MxmCompute:
        if (target != IcuInductionTarget::None
            && target != IcuInductionTarget::MxmAccumulatorAddress)
            throw std::invalid_argument(
                "MXM-compute STREAM_ND packet has an invalid induction target");
        return;
    case IcuStreamNdUnit::MxmDequant:
        if (target != IcuInductionTarget::None)
            throw std::invalid_argument(
                "MXM-dequant STREAM_ND packet cannot induce an operand");
        return;
    }
    throw std::invalid_argument("STREAM_ND packet has an unknown unit kind");
}

inline void validate_icu_stream_nd_schedule(
    const IcuStreamNdSchedule& schedule)
{
    if (schedule.rank == 0
        || schedule.rank > IcuStreamNdSchedule::kMaxRank)
        throw std::invalid_argument(
            "STREAM_ND packet rank must be between one and three");
    if (schedule.start_cycle >= (std::size_t {1} << 24))
        throw std::invalid_argument(
            "STREAM_ND packet start cycle exceeds 24 bits");
    std::size_t lower_span = 0;
    for (std::size_t dimension = 0;
         dimension < IcuStreamNdSchedule::kMaxRank; ++dimension) {
        const auto count = schedule.counts[dimension];
        const auto cycle_stride = schedule.cycle_strides[dimension];
        const auto operand_stride = schedule.operand_strides[dimension];
        if (count == 0 || count > 65536
            || cycle_stride == 0 || cycle_stride >= (std::size_t {1} << 24)
            || !icu_stream_nd_fits_signed(operand_stride, 18))
            throw std::invalid_argument(
                "STREAM_ND packet dimension exceeds its fixed field width");
        if (dimension >= schedule.rank
            && (count != 1 || cycle_stride != 1 || operand_stride != 0))
            throw std::invalid_argument(
                "STREAM_ND packet has non-canonical inactive dimensions");
        if (dimension >= schedule.rank) continue;
        if (dimension != 0 && count > 1 && cycle_stride <= lower_span)
            throw std::invalid_argument(
                "STREAM_ND packet dimensions overlap in issue time");
        const auto steps = count - 1;
        if (steps != 0
            && cycle_stride > (std::numeric_limits<std::size_t>::max()
                    - lower_span)
                / steps)
            throw std::overflow_error(
                "STREAM_ND packet cycle span overflows");
        lower_span += steps * cycle_stride;
    }
    if (schedule.start_cycle
        > std::numeric_limits<std::size_t>::max() - lower_span)
        throw std::overflow_error(
            "STREAM_ND packet final issue cycle overflows");
}

} // namespace detail

inline IcuStreamNdPacket encode_icu_stream_nd_packet(
    const IcuDecodedStreamNdPacket& decoded)
{
    detail::validate_icu_stream_nd_unit(
        decoded.unit, decoded.schedule.induction_target);
    detail::validate_icu_stream_nd_schedule(decoded.schedule);

    IcuStreamNdPacket packet;
    std::uint32_t header = IcuStreamNdPacket::kOpcode;
    header |= static_cast<std::uint32_t>(IcuStreamNdPacket::kVersion) << 4;
    header |= static_cast<std::uint32_t>(decoded.unit) << 6;
    header |= static_cast<std::uint32_t>(decoded.schedule.rank - 1) << 9;
    header |= static_cast<std::uint32_t>(
                  decoded.schedule.induction_target)
        << 11;
    packet.words[0] = header;

    std::size_t bit = 32;
    detail::icu_stream_nd_put(
        packet, bit, 24, decoded.schedule.start_cycle);
    bit += 24;
    detail::icu_stream_nd_put(
        packet, bit, 64, decoded.native_instruction);
    bit += 64;
    for (std::size_t dimension = 0;
         dimension < IcuStreamNdSchedule::kMaxRank; ++dimension) {
        detail::icu_stream_nd_put(packet, bit, 16,
            decoded.schedule.counts[dimension] - 1);
        bit += 16;
        detail::icu_stream_nd_put(packet, bit, 24,
            decoded.schedule.cycle_strides[dimension]);
        bit += 24;
        detail::icu_stream_nd_put(packet, bit, 18,
            static_cast<std::uint64_t>(
                decoded.schedule.operand_strides[dimension])
                & detail::icu_stream_nd_mask(18));
        bit += 18;
    }
    return packet;
}

inline IcuDecodedStreamNdPacket decode_icu_stream_nd_packet(
    const IcuStreamNdPacket& packet)
{
    const std::uint32_t header = packet.words[0];
    if ((header & 0xfu) != IcuStreamNdPacket::kOpcode)
        throw std::invalid_argument("ICU packet is not STREAM_ND");
    if (((header >> 4) & 0x3u) != IcuStreamNdPacket::kVersion)
        throw std::invalid_argument("unsupported ICU STREAM_ND packet version");
    if ((header >> 13) != 0)
        throw std::invalid_argument(
            "ICU STREAM_ND packet reserved header bits are non-zero");

    IcuDecodedStreamNdPacket decoded;
    const auto unit = (header >> 6) & 0x7u;
    if (unit > static_cast<std::uint32_t>(IcuStreamNdUnit::MxmDequant))
        throw std::invalid_argument("ICU STREAM_ND packet unit is invalid");
    decoded.unit = static_cast<IcuStreamNdUnit>(unit);
    decoded.schedule.rank = ((header >> 9) & 0x3u) + 1;
    const auto induction = (header >> 11) & 0x3u;
    decoded.schedule.induction_target =
        static_cast<IcuInductionTarget>(induction);

    std::size_t bit = 32;
    decoded.schedule.start_cycle =
        detail::icu_stream_nd_get(packet, bit, 24);
    bit += 24;
    decoded.native_instruction =
        detail::icu_stream_nd_get(packet, bit, 64);
    bit += 64;
    for (std::size_t dimension = 0;
         dimension < IcuStreamNdSchedule::kMaxRank; ++dimension) {
        decoded.schedule.counts[dimension] =
            detail::icu_stream_nd_get(packet, bit, 16) + 1;
        bit += 16;
        decoded.schedule.cycle_strides[dimension] =
            detail::icu_stream_nd_get(packet, bit, 24);
        bit += 24;
        decoded.schedule.operand_strides[dimension] =
            detail::icu_stream_nd_sign_extend(
                detail::icu_stream_nd_get(packet, bit, 18), 18);
        bit += 18;
    }
    if (bit > IcuStreamNdPacket::kBitCount)
        throw std::logic_error("ICU STREAM_ND packet layout overflow");
    for (; bit < IcuStreamNdPacket::kBitCount; ++bit) {
        if (detail::icu_stream_nd_get(packet, bit, 1) != 0)
            throw std::invalid_argument(
                "ICU STREAM_ND packet padding bits are non-zero");
    }
    detail::validate_icu_stream_nd_unit(
        decoded.unit, decoded.schedule.induction_target);
    detail::validate_icu_stream_nd_schedule(decoded.schedule);
    return decoded;
}

} // namespace ftlpu
