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

struct C2cDmaInstruction {
    C2cDmaDirection direction{C2cDmaDirection::Ddr4ToC2c};
    std::uint64_t ddr4_address{0};
    std::size_t vector_count{1};
    std::size_t address_stride_bytes{hw::kPhysicalVectorBytes};
    std::uint64_t vector_tag_base{0};

    static C2cDmaInstruction Load(
        std::uint64_t ddr4_address,
        std::size_t vector_count = 1,
        std::size_t address_stride_bytes = hw::kPhysicalVectorBytes,
        std::uint64_t vector_tag_base = 0)
    {
        auto instruction = C2cDmaInstruction {
            C2cDmaDirection::Ddr4ToC2c,
            ddr4_address,
            vector_count,
            address_stride_bytes,
            vector_tag_base,
        };
        instruction.validate();
        return instruction;
    }

    static C2cDmaInstruction Store(
        std::uint64_t ddr4_address,
        std::size_t vector_count = 1,
        std::size_t address_stride_bytes = hw::kPhysicalVectorBytes)
    {
        auto instruction = C2cDmaInstruction {
            C2cDmaDirection::C2cToDdr4,
            ddr4_address,
            vector_count,
            address_stride_bytes,
            0,
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
        if (vector_count == 0) {
            throw std::invalid_argument(
                "C2C DMA vector_count must be non-zero");
        }
        (void)vector_address(vector_count - 1);
    }
};

} // namespace ftlpu
