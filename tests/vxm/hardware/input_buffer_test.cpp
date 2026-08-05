#include "ftlpu/vxm/input_buffer.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

int main()
{
    using ftlpu::VxmInputBuffer;

    auto buffer = VxmInputBuffer{};
    buffer.configure(2);
    assert(buffer.collecting());
    assert(buffer.expected_count() == 2);
    assert(buffer.fill_count() == 0);

    auto group5 = VxmInputBuffer::GroupVector{};
    auto group1 = VxmInputBuffer::GroupVector{};
    for (std::size_t lane = 0;
         lane < VxmInputBuffer::kLaneCount;
         ++lane) {
        group5[lane] = {
            static_cast<std::uint8_t>(lane),
            std::uint8_t{0x35}};
        group1[lane] = {
            static_cast<std::uint8_t>(lane + 16),
            std::uint8_t{0x31}};
    }

    buffer.capture_group(5, group5);
    assert(buffer.collecting());
    assert(buffer.fill_count() == 1);
    assert(buffer.has_group(5));
    assert(!buffer.has_group(1));

    auto duplicate_rejected = false;
    try {
        buffer.capture_group(5, group5);
    } catch (const std::logic_error&) {
        duplicate_rejected = true;
    }
    assert(duplicate_rejected);

    buffer.capture_group(1, group1);
    assert(buffer.ready());
    assert(buffer.fill_count() == 2);
    const auto& bundle = buffer.bundle();
    for (std::size_t lane = 0;
         lane < VxmInputBuffer::kLaneCount;
         ++lane) {
        assert(bundle[lane][2] == group1[lane][0]);
        assert(bundle[lane][3] == group1[lane][1]);
        assert(bundle[lane][10] == group5[lane][0]);
        assert(bundle[lane][11] == group5[lane][1]);
    }

    buffer.release_after_issue();
    assert(buffer.empty());
    assert(buffer.fill_count() == 0);
    assert(buffer.expected_count() == 0);
    return 0;
}
