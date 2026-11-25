#pragma once

#include "ftlpu/core/hardware_params.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ftlpu {

enum class C2cDmaDirection : std::uint8_t {
    Ddr4ToC2c,
    C2cToDdr4,
};

// Indirect read of a fixed 512-byte table row. The index is a little-endian
// u32 in DDR. A zero row limit selects ordinary absolute-address DMA.
struct C2cRowIndex {
    std::uint64_t address{};
    std::uint32_t row_limit{};
    void validate(C2cDmaDirection direction, std::uint64_t base,
                  std::size_t count, std::size_t stride) const {
        if (!row_limit && !address) return;
        if (!row_limit || direction != C2cDmaDirection::Ddr4ToC2c ||
            address % 4 || address / 4 >= (1ULL << 40) || row_limit >= (1ULL << 31) ||
            stride != hw::kPhysicalVectorBytes || count == 0 || count > 512 / stride ||
            base % stride || base > UINT64_MAX - (std::uint64_t(row_limit - 1) * 512 + count * stride - 1))
            throw std::invalid_argument("invalid indexed DMA row descriptor");
    }
};

struct C2cDmaInstruction {
    C2cDmaDirection direction{C2cDmaDirection::Ddr4ToC2c};
    std::uint64_t ddr4_address{0};
    std::size_t vector_count{1};
    std::size_t address_stride_bytes{hw::kPhysicalVectorBytes};
    std::uint64_t vector_tag_base{0};
    std::size_t stream_index{0};
    // CModel diagnostic vector tags are deliberately distinct from the
    // hardware synchronization tag carried by the fixed DMA ICU packet.
    std::uint32_t sync_tag{0};
    C2cRowIndex row_index{};

    static C2cDmaInstruction Load(
        std::uint64_t ddr4_address,
        std::size_t vector_count = 1,
        std::size_t address_stride_bytes = hw::kPhysicalVectorBytes,
        std::uint64_t vector_tag_base = 0,
        std::size_t stream_index = 0)
    {
        auto instruction = C2cDmaInstruction {
            C2cDmaDirection::Ddr4ToC2c,
            ddr4_address,
            vector_count,
            address_stride_bytes,
            vector_tag_base,
            stream_index,
        };
        instruction.validate();
        return instruction;
    }

    static C2cDmaInstruction Store(
        std::uint64_t ddr4_address,
        std::size_t vector_count = 1,
        std::size_t address_stride_bytes = hw::kPhysicalVectorBytes,
        std::size_t stream_index = 0)
    {
        auto instruction = C2cDmaInstruction {
            C2cDmaDirection::C2cToDdr4,
            ddr4_address,
            vector_count,
            address_stride_bytes,
            0,
            stream_index,
        };
        instruction.validate();
        return instruction;
    }

    std::uint64_t vector_address(std::size_t vector_index) const
    {
        if (vector_index >= vector_count) {
            throw std::out_of_range(
                "C2C DMA vector index is outside the instruction");
        }
        if (address_stride_bytes != 0
            && vector_index
                > (std::numeric_limits<std::uint64_t>::max()
                    - ddr4_address) / address_stride_bytes) {
            throw std::overflow_error("C2C DMA DDR4 address overflow");
        }
        return ddr4_address + vector_index * address_stride_bytes;
    }

    void validate() const
    {
        row_index.validate(direction, ddr4_address, vector_count, address_stride_bytes);
        if (vector_count == 0) {
            throw std::invalid_argument(
                "C2C DMA vector_count must be non-zero");
        }
        if (stream_index >= hw::kC2cStreamsPerDirection) {
            throw std::out_of_range(
                "C2C DMA lane index is outside the configured lane range");
        }
        (void)vector_address(vector_count - 1);
    }
};

} // namespace ftlpu
