#include "ftlpu/core/sram_1rw.hpp"

#include <cassert>
#include <cstdint>

namespace {

using Sram = ftlpu::timing::Sram1Rw<std::uint32_t, 4>;

struct Harness {
    Sram dut{};
    ftlpu::timing::Clock clock{};
    ftlpu::timing::SystemPhase system{};

    Harness()
    {
        clock.register_module(dut);
        system.add_node("sram", [&] { dut.low(); }, [&] { dut.high(); });
    }

    void tick(const Sram::Inputs& inputs)
    {
        dut.drive(inputs);
        system.low();
        clock.complete_low();
        clock.validate_all();
        clock.commit_all();
        system.high();
        clock.complete_high();
        clock.negedge_all();
    }
};

} // namespace

int main()
{
    Harness h;
    h.tick(Sram::Inputs{.rst_ni = false});
    assert(!h.dut.rvalid());
    assert(h.dut.rdata() == 0);

    h.tick(Sram::Inputs{.cen_i = true, .wen_i = true,
                        .address_i = 1, .wdata_i = 0x12345678u});
    assert(!h.dut.rvalid());
    assert(h.dut.inspect(1) == 0x12345678u);

    h.tick(Sram::Inputs{.cen_i = true, .wen_i = false, .address_i = 1});
    assert(h.dut.rvalid());
    assert(h.dut.rdata() == 0x12345678u);

    // Idle clears only valid; the registered read data holds.
    h.tick(Sram::Inputs{});
    assert(!h.dut.rvalid());
    assert(h.dut.rdata() == 0x12345678u);

    h.tick(Sram::Inputs{.cen_i = true, .wen_i = true,
                        .address_i = 2, .wdata_i = 0xa5a5a5a5u});
    h.tick(Sram::Inputs{.cen_i = true, .wen_i = false, .address_i = 1});
    assert(h.dut.rvalid() && h.dut.rdata() == 0x12345678u);
    h.tick(Sram::Inputs{.cen_i = true, .wen_i = false, .address_i = 2});
    assert(h.dut.rvalid() && h.dut.rdata() == 0xa5a5a5a5u);

    // wen owns the one port.  A write never creates a read response.
    h.tick(Sram::Inputs{.cen_i = true, .wen_i = true,
                        .address_i = 2, .wdata_i = 0xfeedbeefu});
    assert(!h.dut.rvalid());
    h.tick(Sram::Inputs{.cen_i = true, .wen_i = false, .address_i = 2});
    assert(h.dut.rvalid() && h.dut.rdata() == 0xfeedbeefu);

    // Reset clears the output registers, but the array survives.
    h.tick(Sram::Inputs{.rst_ni = false});
    assert(!h.dut.rvalid() && h.dut.rdata() == 0);
    assert(h.dut.inspect(2) == 0xfeedbeefu);
    h.tick(Sram::Inputs{.cen_i = true, .wen_i = false, .address_i = 2});
    assert(h.dut.rvalid() && h.dut.rdata() == 0xfeedbeefu);
}
