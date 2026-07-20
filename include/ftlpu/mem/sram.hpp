#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/stream.hpp"
#include "ftlpu/mem/address.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ftlpu {

// One of the two physical banks in a tile-local SRAM block.
class SramBank {
public:
    static constexpr std::size_t kWords = hw::kSramWordsPerBank;
    static constexpr std::size_t kBytesPerWord = hw::kSramWordBytes;
    static constexpr std::size_t kCapacityBytes = hw::kSramBankBytes;

    SramBank()
        : bytes_(kCapacityBytes, 0)
    {
    }

    void clear()
    {
        std::fill(bytes_.begin(), bytes_.end(), 0);
    }

    std::uint8_t byte(std::size_t word, std::size_t byte_offset) const
    {
        return bytes_.at(flat_index(word, byte_offset));
    }

    void set_byte(std::size_t word, std::size_t byte_offset, std::uint8_t value)
    {
        bytes_.at(flat_index(word, byte_offset)) = value;
    }

    StreamPayloadSegment16 read_word(std::size_t word) const
    {
        check_word(word);
        StreamPayloadSegment16 result{};
        for (std::size_t byte = 0; byte < kBytesPerWord; ++byte) {
            result[byte] = this->byte(word, byte);
        }
        return result;
    }

    void write_word(std::size_t word, const StreamPayloadSegment16& values)
    {
        check_word(word);
        for (std::size_t byte = 0; byte < kBytesPerWord; ++byte) {
            set_byte(word, byte, values[byte]);
        }
    }

private:
    static void check_word(std::size_t word)
    {
        if (word >= kWords) {
            throw std::out_of_range("SRAM word is outside the 4096-word bank");
        }
    }

    static std::size_t flat_index(std::size_t word, std::size_t byte_offset)
    {
        check_word(word);
        if (byte_offset >= kBytesPerWord) {
            throw std::out_of_range("SRAM byte offset is outside the 16-byte word");
        }
        return word * kBytesPerWord + byte_offset;
    }

    std::vector<std::uint8_t> bytes_{};
};

// SRAM physically attached to one MEM tile: two 4096 x 16-byte banks.
class SramTileBlock {
public:
    void clear()
    {
        for (auto& bank : banks_) {
            bank.clear();
        }
    }

    std::uint8_t byte(MemSliceByteAddress17 address) const
    {
        return bank(address.bank()).byte(address.word(), address.byte_offset());
    }

    void set_byte(MemSliceByteAddress17 address, std::uint8_t value)
    {
        bank(address.bank()).set_byte(address.word(), address.byte_offset(), value);
    }

    StreamPayloadSegment16 read_word(MemLocalWordAddress13 address) const
    {
        return bank(address.bank()).read_word(address.word());
    }

    void write_word(
        MemLocalWordAddress13 address,
        const StreamPayloadSegment16& values)
    {
        bank(address.bank()).write_word(address.word(), values);
    }

    SramBank& bank(std::size_t bank)
    {
        return banks_.at(bank);
    }

    const SramBank& bank(std::size_t bank) const
    {
        return banks_.at(bank);
    }

private:
    std::array<SramBank, hw::kSramBanksPerTileBlock> banks_{};
};

class SramSlice {
public:
    void clear()
    {
        for (auto& block : tile_blocks_) {
            block.clear();
        }
    }

    SramTileBlock& tile_block(std::size_t tile)
    {
        return tile_blocks_.at(tile);
    }

    const SramTileBlock& tile_block(std::size_t tile) const
    {
        return tile_blocks_.at(tile);
    }

private:
    std::array<SramTileBlock, hw::kSramTileBlocksPerSlice> tile_blocks_{};
};

class SramArray {
public:
    SramArray()
        : slices_(hw::kMemSliceColumns)
    {
    }

    void clear()
    {
        for (auto& slice : slices_) {
            slice.clear();
        }
    }

    SramSlice& slice(std::size_t mem_slice)
    {
        return slices_.at(mem_slice);
    }

    const SramSlice& slice(std::size_t mem_slice) const
    {
        return slices_.at(mem_slice);
    }

private:
    std::vector<SramSlice> slices_{};
};

} // namespace ftlpu
