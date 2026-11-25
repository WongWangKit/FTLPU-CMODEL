#include "ftlpu/core/sram_fifo.hpp"

#include <array>
#include <cassert>
#include <cstdint>

namespace {

using Fifo = ftlpu::timing::SramFifo<std::uint32_t, 3>;

struct Harness {
    Fifo dut{};
    ftlpu::timing::Clock clock{};
    ftlpu::timing::SystemPhase system{};

    Harness()
    {
        clock.register_module(dut);
        system.add_node("sram_fifo", [&] { dut.low(); }, [&] { dut.high(); });
    }

    Fifo::Outputs tick(const Fifo::Inputs& inputs)
    {
        dut.drive(inputs);
        system.low();
        const auto sampled = dut.outputs();
        clock.complete_low();
        clock.validate_all();
        clock.commit_all();
        system.high();
        clock.complete_high();
        clock.negedge_all();
        return sampled;
    }

    void wait_valid()
    {
        for (unsigned i = 0; i != 5 && !dut.outputs().dequeue_valid_o; ++i)
            tick(Fifo::Inputs{});
        assert(dut.outputs().dequeue_valid_o);
    }
};

} // namespace

int main()
{
    Harness h;
    h.tick(Fifo::Inputs{.rst_ni = false});
    assert(h.dut.outputs().level_o == 0);
    assert(!h.dut.outputs().dequeue_valid_o);

    auto p = h.tick(Fifo::Inputs{.enqueue_valid_i = true,
                                 .enqueue_data_i = 11});
    assert(p.enqueue_fire);
    assert(h.dut.outputs().level_o == 1);

    // The SRAM request and return are separate registered steps.
    p = h.tick(Fifo::Inputs{.enqueue_valid_i = true,
                            .enqueue_data_i = 22});
    assert(p.dequeue_request);
    assert(!p.enqueue_ready_o);
    p = h.tick(Fifo::Inputs{.enqueue_valid_i = true,
                            .enqueue_data_i = 22});
    assert(p.enqueue_fire);
    assert(h.dut.outputs().dequeue_valid_o);
    assert(h.dut.outputs().dequeue_data_o == 11);

    // Backpressure holds output valid/data stable while more entries arrive.
    p = h.tick(Fifo::Inputs{.enqueue_valid_i = true,
                            .enqueue_data_i = 33});
    assert(p.enqueue_fire);
    assert(h.dut.outputs().level_o == 3);
    assert(h.dut.outputs().dequeue_valid_o);
    assert(h.dut.outputs().dequeue_data_o == 11);
    p = h.tick(Fifo::Inputs{.enqueue_valid_i = true,
                            .enqueue_data_i = 44});
    assert(!p.enqueue_ready_o && !p.enqueue_fire);
    assert(h.dut.outputs().dequeue_data_o == 11);

    // Drain in order.  Depth three also proves explicit non-power-of-two wrap.
    constexpr std::array expected{11u, 22u, 33u};
    for (const auto value : expected) {
        h.wait_valid();
        assert(h.dut.outputs().dequeue_data_o == value);
        p = h.tick(Fifo::Inputs{.dequeue_ready_i = true});
        assert(p.dequeue_fire);
    }
    for (unsigned i = 0; i != 4; ++i) h.tick(Fifo::Inputs{});
    assert(h.dut.outputs().level_o == 0);
    assert(!h.dut.outputs().dequeue_valid_o);

    // Fill and drain again to cross the wrapped write/read pointers.
    for (const auto value : std::array{44u, 55u, 66u}) {
        for (;;) {
            p = h.tick(Fifo::Inputs{.enqueue_valid_i = true,
                                    .enqueue_data_i = value});
            if (p.enqueue_fire) break;
        }
    }
    for (const auto value : std::array{44u, 55u, 66u}) {
        h.wait_valid();
        assert(h.dut.outputs().dequeue_data_o == value);
        h.tick(Fifo::Inputs{.dequeue_ready_i = true});
    }

    // Reset clears FIFO state/output but does not clear the physical SRAM.
    h.tick(Fifo::Inputs{.rst_ni = false});
    assert(h.dut.outputs().level_o == 0);
    assert(!h.dut.outputs().dequeue_valid_o);
}
