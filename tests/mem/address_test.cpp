#include "ftlpu/mem/address.hpp"

#include <cassert>
#include <stdexcept>
#include <type_traits>

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
    static_assert(!std::is_convertible_v<ftlpu::MemGlobalAddress24, std::size_t>);
    static_assert(!std::is_convertible_v<ftlpu::MemSliceByteAddress17, std::size_t>);
    static_assert(!std::is_convertible_v<ftlpu::MemLocalWordAddress13, std::size_t>);
    static_assert(!std::is_convertible_v<ftlpu::MemGlobalAddress24, ftlpu::MemSliceByteAddress17>);
    static_assert(!std::is_convertible_v<ftlpu::MemSliceByteAddress17, ftlpu::MemLocalWordAddress13>);

    const auto local_byte =
        ftlpu::MemSliceByteAddress17::FromFields(1, 4095, 15);
    assert(local_byte.encoded() == 0x1ffff);
    assert(local_byte.bank() == 1);
    assert(local_byte.word() == 4095);
    assert(local_byte.byte_offset() == 15);

    const auto local_word =
        ftlpu::MemLocalWordAddress13::FromFields(1, 4095);
    assert(local_word.encoded() == 0x1fff);
    assert(local_word.slice_byte_address(15) == local_byte);

    const auto aligned =
        ftlpu::MemSliceByteAddress17::FromFields(1, 4095, 0);
    assert(aligned.local_word_address() == local_word);
    assert(throws([&] { (void)local_byte.local_word_address(); }));

    constexpr auto bank0_word10 =
        ftlpu::MemLocalWordAddress13::FromFields(0, 10);
    static_assert(bank0_word10.next_word()
        == ftlpu::MemLocalWordAddress13::FromFields(0, 11));
    static_assert(bank0_word10.advance_words(100)
        == ftlpu::MemLocalWordAddress13::FromFields(0, 110));
    static_assert(bank0_word10.advance_words(0) == bank0_word10);
    assert(throws([] {
        (void)ftlpu::MemLocalWordAddress13::FromFields(0, 4095).next_word();
    }));
    assert(throws([] {
        (void)ftlpu::MemLocalWordAddress13::FromFields(1, 4095).next_word();
    }));
    assert(throws([] {
        (void)ftlpu::MemLocalWordAddress13::FromFields(0, 4090).advance_words(6);
    }));

    const auto global = ftlpu::MemGlobalAddress24::FromFields(1, 43, local_byte);
    assert(global.hemisphere() == 1);
    assert(global.mem_slice() == 43);
    assert(global.slice_byte_address() == local_byte);
    assert(global.encoded() == ((std::size_t {1} << 23) | (43u << 17) | 0x1ffffu));

    assert(throws([] { (void)ftlpu::MemGlobalAddress24(1u << 24); }));
    assert(throws([] { (void)ftlpu::MemSliceByteAddress17(1u << 17); }));
    assert(throws([] { (void)ftlpu::MemLocalWordAddress13(1u << 13); }));
    assert(throws([] {
        (void)ftlpu::MemGlobalAddress24::FromFields(
            0, 44, ftlpu::MemSliceByteAddress17(0));
    }));

    return 0;
}
