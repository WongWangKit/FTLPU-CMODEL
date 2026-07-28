#pragma once

#include "ftlpu/core/hardware_params.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ftlpu {

class MemLocalWordAddress13;

namespace address_detail {

constexpr std::size_t bits_for_count(std::size_t count)
{
    return count <= 1 ? 0 : std::bit_width(count - 1);
}

constexpr std::size_t mask_for_bits(std::size_t bits)
{
    return bits == 0 ? 0 : (std::size_t {1} << bits) - 1;
}

} // namespace address_detail

// Byte address after a MEM slice has been routed.  The historical class name
// is retained for source compatibility; kBits and all field positions are
// derived from the active SRAM configuration.
class MemSliceByteAddress17 {
public:
    static constexpr std::size_t kByteBits =
        address_detail::bits_for_count(hw::kSramWordBytes);
    static constexpr std::size_t kWordBits =
        address_detail::bits_for_count(hw::kSramWordsPerBank);
    static constexpr std::size_t kBankBits =
        address_detail::bits_for_count(hw::kSramBanksPerTileBlock);
    static constexpr std::size_t kBits = kByteBits + kWordBits + kBankBits;
    static constexpr std::size_t kLimit = std::size_t {1} << kBits;

    constexpr MemSliceByteAddress17() noexcept = default;

    explicit constexpr MemSliceByteAddress17(std::size_t encoded)
        : encoded_(checked(encoded))
    {
    }

    static constexpr MemSliceByteAddress17 FromFields(
        std::size_t bank,
        std::size_t word,
        std::size_t byte_offset)
    {
        if (bank >= hw::kSramBanksPerTileBlock) {
            throw std::out_of_range("MEM bank does not fit slice-local byte address");
        }
        if (word >= hw::kSramWordsPerBank) {
            throw std::out_of_range("MEM word does not fit slice-local byte address");
        }
        if (byte_offset >= hw::kSramWordBytes) {
            throw std::out_of_range("MEM byte offset is outside the 16-byte word");
        }
        return MemSliceByteAddress17(
            (bank << (kWordBits + kByteBits))
            | (word << kByteBits)
            | byte_offset);
    }

    constexpr std::size_t encoded() const noexcept { return encoded_; }
    constexpr std::size_t bank() const noexcept
    {
        return encoded_ >> (kWordBits + kByteBits);
    }
    constexpr std::size_t word() const noexcept
    {
        return (encoded_ >> kByteBits)
            & address_detail::mask_for_bits(kWordBits);
    }
    constexpr std::size_t byte_offset() const noexcept
    {
        return encoded_ & address_detail::mask_for_bits(kByteBits);
    }
    constexpr bool word_aligned() const noexcept { return byte_offset() == 0; }

    constexpr MemLocalWordAddress13 local_word_address() const;

    friend constexpr bool operator==(
        MemSliceByteAddress17,
        MemSliceByteAddress17) = default;

private:
    static constexpr std::uint32_t checked(std::size_t encoded)
    {
        if (encoded >= kLimit) {
            throw std::out_of_range("MEM slice byte address exceeds the configured field width");
        }
        return static_cast<std::uint32_t>(encoded);
    }

    std::uint32_t encoded_{0};
};

// Address carried by MEM Read/Write. The physical pipeline stage selects the
// tile-local block.  The historical class name is retained for compatibility.
class MemLocalWordAddress13 {
public:
    static constexpr std::size_t kWordBits =
        address_detail::bits_for_count(hw::kSramWordsPerBank);
    static constexpr std::size_t kBankBits =
        address_detail::bits_for_count(hw::kSramBanksPerTileBlock);
    static constexpr std::size_t kBits = kWordBits + kBankBits;
    static constexpr std::size_t kLimit = std::size_t {1} << kBits;

    constexpr MemLocalWordAddress13() noexcept = default;

    explicit constexpr MemLocalWordAddress13(std::size_t encoded)
        : encoded_(checked(encoded))
    {
    }

    static constexpr MemLocalWordAddress13 FromFields(
        std::size_t bank,
        std::size_t word)
    {
        if (bank >= hw::kSramBanksPerTileBlock) {
            throw std::out_of_range("MEM bank does not fit local word address");
        }
        if (word >= hw::kSramWordsPerBank) {
            throw std::out_of_range("MEM word does not fit local word address");
        }
        return MemLocalWordAddress13((bank << kWordBits) | word);
    }

    constexpr std::size_t encoded() const noexcept { return encoded_; }
    constexpr std::size_t bank() const noexcept { return encoded_ >> kWordBits; }
    constexpr std::size_t word() const noexcept
    {
        return encoded_ & address_detail::mask_for_bits(kWordBits);
    }

    constexpr MemLocalWordAddress13 next_word() const
    {
        return advance_words(1);
    }

    constexpr MemLocalWordAddress13 advance_words(std::size_t count) const
    {
        constexpr auto kWordsPerSlice =
            hw::kSramBanksPerTileBlock * hw::kSramWordsPerBank;
        const auto linear_word = bank() * hw::kSramWordsPerBank + word();
        if (count > kWordsPerSlice - 1 - linear_word) {
            throw std::out_of_range("MEM word address advance exceeds the two-bank slice capacity");
        }
        const auto advanced = linear_word + count;
        return FromFields(
            advanced / hw::kSramWordsPerBank,
            advanced % hw::kSramWordsPerBank);
    }

    constexpr MemSliceByteAddress17 slice_byte_address(
        std::size_t byte_offset = 0) const
    {
        return MemSliceByteAddress17::FromFields(bank(), word(), byte_offset);
    }

    friend constexpr bool operator==(
        MemLocalWordAddress13,
        MemLocalWordAddress13) = default;

private:
    static constexpr std::uint32_t checked(std::size_t encoded)
    {
        if (encoded >= kLimit) {
            throw std::out_of_range("MEM local word address exceeds the configured field width");
        }
        return static_cast<std::uint32_t>(encoded);
    }

    std::uint32_t encoded_{0};
};

constexpr MemLocalWordAddress13 MemSliceByteAddress17::local_word_address() const
{
    if (!word_aligned()) {
        throw std::invalid_argument("MEM Read/Write address must be 16-byte aligned");
    }
    return MemLocalWordAddress13::FromFields(bank(), word());
}

// Software-visible byte address.  The historical class name is retained for
// compatibility; narrower vector rows require a wider local SRAM address.
class MemGlobalAddress24 {
public:
    static constexpr std::size_t kLocalBits = MemSliceByteAddress17::kBits;
    static constexpr std::size_t kSliceBits =
        address_detail::bits_for_count(hw::kMemSliceColumns);
    static constexpr std::size_t kHemisphereBits =
        address_detail::bits_for_count(hw::kHemispheres);
    static constexpr std::size_t kBits =
        kLocalBits + kSliceBits + kHemisphereBits;
    static constexpr std::size_t kLimit = std::size_t {1} << kBits;

    constexpr MemGlobalAddress24() noexcept = default;

    explicit constexpr MemGlobalAddress24(std::size_t encoded)
        : encoded_(checked(encoded))
    {
    }

    static constexpr MemGlobalAddress24 FromFields(
        std::size_t hemisphere,
        std::size_t mem_slice,
        MemSliceByteAddress17 local)
    {
        if (hemisphere >= hw::kHemispheres) {
            throw std::out_of_range("MEM hemisphere does not fit global address");
        }
        if (mem_slice >= hw::kMemSliceColumns) {
            throw std::out_of_range("MEM slice does not fit global address");
        }
        return MemGlobalAddress24(
            (hemisphere << (kSliceBits + kLocalBits))
            | (mem_slice << kLocalBits)
            | local.encoded());
    }

    constexpr std::size_t encoded() const noexcept { return encoded_; }
    constexpr std::size_t hemisphere() const noexcept
    {
        return encoded_ >> (kSliceBits + kLocalBits);
    }
    constexpr std::size_t mem_slice() const noexcept
    {
        return (encoded_ >> kLocalBits)
            & address_detail::mask_for_bits(kSliceBits);
    }
    constexpr MemSliceByteAddress17 slice_byte_address() const
    {
        return MemSliceByteAddress17(
            encoded_ & address_detail::mask_for_bits(kLocalBits));
    }

    friend constexpr bool operator==(MemGlobalAddress24, MemGlobalAddress24) = default;

private:
    static constexpr std::uint32_t checked(std::size_t encoded)
    {
        if (encoded >= kLimit) {
            throw std::out_of_range("MEM global address exceeds the configured field width");
        }
        return static_cast<std::uint32_t>(encoded);
    }

    std::uint32_t encoded_{0};
};

} // namespace ftlpu
