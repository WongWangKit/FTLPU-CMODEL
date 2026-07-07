#include "ftlpu/stream.hpp"

#include <cassert>
#include <cstdint>

int main()
{
    ftlpu::StreamRegister<std::uint32_t> reg;

    assert(!reg.output().has_value());

    reg.set_input(ftlpu::StreamWord<std::uint32_t> {10, false});
    reg.tick();
    assert(reg.output().has_value());
    assert(reg.output()->data == 10);
    assert(!reg.output()->last);

    reg.tick();
    assert(!reg.output().has_value());

    reg.set_input(ftlpu::StreamWord<std::uint32_t> {20, true});
    reg.tick();
    assert(reg.output().has_value());
    assert(reg.output()->data == 20);
    assert(reg.output()->last);

    reg.reset();
    assert(!reg.output().has_value());

    return 0;
}
