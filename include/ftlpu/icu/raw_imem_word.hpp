#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ftlpu {

template <std::size_t Bits>
struct IcuRawImemWord {
    static_assert(Bits >= 32 && Bits % 32 == 0);

    static constexpr std::size_t bit_count = Bits;
    static constexpr std::size_t lane_count = Bits / 32;
    std::array<std::uint32_t, lane_count> lanes{};

    bool bit(std::size_t index) const
    {
        if (index >= Bits)
            throw std::out_of_range(
                "ICU raw i-MEM bit index is outside the word");
        return ((lanes[index / 32] >> (index % 32)) & 1u) != 0;
    }
};

using IcuRawImemWord96 = IcuRawImemWord<96>;
using IcuRawImemWord128 = IcuRawImemWord<128>;

} // namespace ftlpu
