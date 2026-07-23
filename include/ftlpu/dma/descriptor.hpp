#pragma once

#include "ftlpu/dma/host_memory.hpp"
#include "ftlpu/mem/address.hpp"

#include <cstddef>
#include <cstdint>

namespace ftlpu {

class DmaTransferId {
public:
    constexpr DmaTransferId() noexcept = default;

    explicit constexpr DmaTransferId(std::uint64_t value) noexcept
        : value_(value)
    {
    }

    constexpr std::uint64_t value() const noexcept { return value_; }
    constexpr bool valid() const noexcept { return value_ != 0; }

    friend constexpr bool operator==(DmaTransferId, DmaTransferId) = default;

private:
    std::uint64_t value_{0};
};

enum class DmaDirection : std::uint8_t {
    HostToMemory,
    MemoryToHost,
};

enum class DmaPurpose : std::uint8_t {
    General,
    Model,
    InputTensor,
    OutputTensor,
    Program,
};

struct DmaDescriptor {
    DmaDirection direction{DmaDirection::HostToMemory};
    DmaPurpose purpose{DmaPurpose::General};
    HostBufferId host_buffer{};
    std::size_t host_offset_bytes{0};
    MemGlobalAddress24 memory_address{};
    std::size_t vector_count{1};
};

} // namespace ftlpu
