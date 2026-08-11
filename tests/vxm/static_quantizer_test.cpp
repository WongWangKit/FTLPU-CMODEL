#include "ftlpu/vxm/static_quantizer.hpp"

#include <cassert>
#include <cstdint>
#include <vector>

int main()
{
    using ftlpu::VxmStaticQuantizer;

    static_assert(VxmStaticQuantizer::kLatency == 1);
    auto quantizer = VxmStaticQuantizer{};

    auto completed = quantizer.tick({
        {4.0f, 0.5f, 0, 3},
    });
    assert(completed.empty());
    assert(!quantizer.idle());

    // A new beat is accepted while the prior beat retires: no ready signal,
    // no backpressure and initiation interval one.
    completed = quantizer.tick({
        {2.0f, 0.25f, 1, 9},
    });
    assert(completed.size() == 1);
    assert(completed[0].stream == 3);
    assert(completed[0].value == std::int8_t {8});

    completed = quantizer.tick();
    assert(completed.size() == 1);
    assert(completed[0].stream == 9);
    assert(completed[0].value == std::int8_t {9});
    assert(quantizer.idle());
    return 0;
}
