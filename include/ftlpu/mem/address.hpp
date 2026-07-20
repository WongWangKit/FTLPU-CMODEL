#pragma once

#include "ftlpu/core/hardware_params.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ftlpu {

class MemLocalWordAddress13;

// Byte address after a MEM slice has been routed. Layout:
//   [16] bank, [15:4] word, [3:0] byte within the 16-byte word.
class MemSliceByteAddress17 {
public:
    static constexpr std::size_t kBits = 17;
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
            (bank << 16) | (word << 4) | byte_offset);
    }

    constexpr std::size_t encoded() const noexcept { return encoded_; }
    constexpr std::size_t bank() const noexcept { return encoded_ >> 16; }
    constexpr std::size_t word() const noexcept { return (encoded_ >> 4) & 0x0fff; }
    constexpr std::size_t byte_offset() const noexcept { return encoded_ & 0x0f; }
    constexpr bool word_aligned() const noexcept { return byte_offset() == 0; }

    constexpr MemLocalWordAddress13 local_word_address() const;

    friend constexpr bool operator==(
        MemSliceByteAddress17,
        MemSliceByteAddress17) = default;

private:
    static constexpr std::uint32_t checked(std::size_t encoded)
    {
        if (encoded >= kLimit) {
            throw std::out_of_range("MEM slice byte address exceeds 17 bits");
        }
        return static_cast<std::uint32_t>(encoded);
    }

    std::uint32_t encoded_{0};
};

// Address carried by MEM Read/Write. The physical pipeline stage selects the
// tile-local block; bit 12 selects a bank and bits 11:0 select a word.
class MemLocalWordAddress13 {
public:
    static constexpr std::size_t kBits = 13;
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
        return MemLocalWordAddress13((bank << 12) | word);
    }

    constexpr std::size_t encoded() const noexcept { return encoded_; }
    constexpr std::size_t bank() const noexcept { return encoded_ >> 12; }
    constexpr std::size_t word() const noexcept { return encoded_ & 0x0fff; }

    constexpr MemLocalWordAddress13 next_word() const
    {
        return advance_words(1);
    }

    constexpr MemLocalWordAddress13 advance_words(std::size_t count) const
    {
        if (count > hw::kSramWordsPerBank - 1 - word()) {
            throw std::out_of_range("MEM word address advance crosses a bank boundary");
        }
        return FromFields(bank(), word() + count);
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
    static constexpr std::uint16_t checked(std::size_t encoded)
    {
        if (encoded >= kLimit) {
            throw std::out_of_range("MEM local word address exceeds 13 bits");
        }
        return static_cast<std::uint16_t>(encoded);
    }

    std::uint16_t encoded_{0};
};

constexpr MemLocalWordAddress13 MemSliceByteAddress17::local_word_address() const
{
    if (!word_aligned()) {
        throw std::invalid_argument("MEM Read/Write address must be 16-byte aligned");
    }
    return MemLocalWordAddress13::FromFields(bank(), word());
}

// Lower 24 bits of the software-visible byte address. Layout:
//   [23] hemisphere, [22:17] slice, [16:0] slice-local byte address.
class MemGlobalAddress24 {
public:
    static constexpr std::size_t kBits = 24;
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
            (hemisphere << 23) | (mem_slice << 17) | local.encoded());
    }

    constexpr std::size_t encoded() const noexcept { return encoded_; }
    constexpr std::size_t hemisphere() const noexcept { return encoded_ >> 23; }
    constexpr std::size_t mem_slice() const noexcept { return (encoded_ >> 17) & 0x3f; }
    constexpr MemSliceByteAddress17 slice_byte_address() const
    {
        return MemSliceByteAddress17(encoded_ & 0x1ffff);
    }

    friend constexpr bool operator==(MemGlobalAddress24, MemGlobalAddress24) = default;

private:
    static constexpr std::uint32_t checked(std::size_t encoded)
    {
        if (encoded >= kLimit) {
            throw std::out_of_range("MEM global address exceeds 24 bits");
        }
        return static_cast<std::uint32_t>(encoded);
    }

    std::uint32_t encoded_{0};
};

} // namespace ftlpu
