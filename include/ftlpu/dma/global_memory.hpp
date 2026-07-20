#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/mem/address.hpp"
#include "ftlpu/mem/mem_array.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>

namespace ftlpu {

// Routes the lower 24 bits of a software-visible MEM address to one of the
// modeled hemisphere MEM arrays and one of its 44 slice columns.
class GlobalMemoryAddressSpace {
public:
    struct DecodedAddress {
        std::size_t hemisphere{0};
        std::size_t mem_slice{0};
        MemLocalWordAddress13 local_word{};
    };

    void bind_hemisphere(std::size_t hemisphere, MemArrayModel& memory)
    {
        check_hemisphere(hemisphere);
        hemispheres_[hemisphere] = &memory;
    }

    bool hemisphere_bound(std::size_t hemisphere) const
    {
        check_hemisphere(hemisphere);
        return hemispheres_[hemisphere] != nullptr;
    }

    DecodedAddress decode(MemGlobalAddress24 address) const
    {
        const auto hemisphere = address.hemisphere();
        const auto mem_slice = address.mem_slice();
        check_hemisphere(hemisphere);
        if (hemispheres_[hemisphere] == nullptr) {
            throw std::logic_error("DMA target hemisphere is not bound to a MEM array");
        }
        if (mem_slice >= hw::kMemSliceColumns) {
            throw std::out_of_range("DMA global address selects an invalid MEM slice");
        }
        return DecodedAddress {
            hemisphere,
            mem_slice,
            address.slice_byte_address().local_word_address(),
        };
    }

    void write_vector(
        MemGlobalAddress24 address,
        const StreamPayloadVector320& values)
    {
        const auto decoded = decode(address);
        hemispheres_[decoded.hemisphere]->write_sram_vector(
            decoded.mem_slice, decoded.local_word, values);
    }

    StreamPayloadVector320 read_vector(MemGlobalAddress24 address) const
    {
        const auto decoded = decode(address);
        return hemispheres_[decoded.hemisphere]->read_sram_vector(
            decoded.mem_slice, decoded.local_word);
    }

private:
    static void check_hemisphere(std::size_t hemisphere)
    {
        if (hemisphere >= hw::kHemispheres) {
            throw std::out_of_range("DMA hemisphere index is outside the two hemispheres");
        }
    }

    std::array<MemArrayModel*, hw::kHemispheres> hemispheres_{};
};

} // namespace ftlpu
