#include "ftlpu/mem/sram.hpp"

#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace {

template <typename Fn>
bool throws(Fn&& fn)
{
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

} // namespace

int main()
{
    ftlpu::SramSlice slice;
    const auto address = ftlpu::MemLocalWordAddress13::FromFields(1, 4095);
    ftlpu::StreamPayloadVector320 expected{};
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            expected[tile][lane] =
                static_cast<std::uint8_t>((tile * ftlpu::hw::kLanesPerTile + lane) & 0xff);
        }
    }

    slice.write_vector(address, expected);
    assert(slice.read_vector(address) == expected);
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        assert(slice.tile_block(tile).read_word(address) == expected[tile]);
    }

    const auto other_bank = ftlpu::MemLocalWordAddress13::FromFields(0, 4095);
    assert(slice.read_vector(other_bank) == ftlpu::StreamPayloadVector320 {});

    ftlpu::SramBank bank;
    assert(throws([&] { (void)bank.read_word(ftlpu::hw::kSramWordsPerBank); }));
    assert(throws([&] {
        bank.write_word(
            ftlpu::hw::kSramWordsPerBank,
            ftlpu::StreamPayloadSegment16 {});
    }));

    return 0;
}
