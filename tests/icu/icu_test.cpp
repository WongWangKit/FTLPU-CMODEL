#include "ftlpu/icu/icu.hpp"

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

struct TestInstruction {
    int opcode{0};
    std::size_t value{0};

    friend bool operator==(const TestInstruction&, const TestInstruction&) = default;
};

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
    using Control = ftlpu::IcuControlInstruction;
    using Icu = ftlpu::SliceIcu<TestInstruction>;

    // Functional and control instructions share one FIFO. Read/Sync/Write
    // cannot reorder because Sync remains at the queue head until notified.
    {
        Icu icu;
        icu.load({
            TestInstruction {1, 100},
            Control::Sync(),
            TestInstruction {2, 200},
        });

        assert((icu.tick() == TestInstruction {1, 100}));
        assert(!icu.tick().has_value());
        assert(icu.blocked_on_sync());
        assert(!icu.tick().has_value());
        assert(icu.blocked_on_sync());

        icu.notify();
        assert(!icu.tick().has_value()); // consumes Sync locally
        assert(!icu.blocked_on_sync());
        assert((icu.tick() == TestInstruction {2, 200}));
        assert(icu.done());
    }

    // NOP is processed locally and blocks later functional entries.
    {
        Icu icu;
        icu.load({
            TestInstruction {3, 0},
            Control::Nop(2),
            TestInstruction {4, 0},
        });
        assert(icu.tick()->opcode == 3);
        assert(!icu.tick().has_value());
        assert(!icu.tick().has_value());
        assert(icu.tick()->opcode == 4);
        assert(icu.done());
    }

    // Repeat also occupies the same ordered stream and reproduces the most
    // recent functional instruction at the requested interval.
    {
        Icu icu;
        icu.load({TestInstruction {5, 9}, Control::Repeat(2, 2)});
        assert(icu.tick()->opcode == 5);
        assert(!icu.tick().has_value());
        assert(icu.tick()->opcode == 5);
        assert(!icu.tick().has_value());
        assert(icu.tick()->opcode == 5);
        assert(icu.done());
    }

    // MEM Repeat preserves the existing address-stride capability.
    {
        ftlpu::SliceIcu<ftlpu::MemInstruction> icu;
        icu.load({
            ftlpu::MemInstruction::Read(10, ftlpu::StreamId::East(1)),
            Control::Repeat(2, 1, 4),
        });
        assert(icu.tick()->address == ftlpu::MemLocalWordAddress13(10));
        assert(icu.tick()->address == ftlpu::MemLocalWordAddress13(14));
        assert(icu.tick()->address == ftlpu::MemLocalWordAddress13(18));
    }

    // Fetch and Notify are ICU-local controls; neither is sent to the slice.
    {
        Icu icu;
        icu.load({Control::Fetch(), Control::Notify(), TestInstruction {6, 0}});
        assert(!icu.tick().has_value());
        assert(icu.fetch_count() == 1);
        assert(!icu.tick().has_value());
        assert(icu.take_notify());
        assert(!icu.take_notify());
        assert(icu.tick()->opcode == 6);
    }

    assert(throws([] {
        Icu icu;
        icu.load({Control::Repeat(1)});
        (void)icu.tick();
    }));
    assert(throws([] {
        Icu icu;
        icu.load({TestInstruction {7, 0}, Control::Repeat(1, 0)});
        (void)icu.tick();
        (void)icu.tick();
    }));

    // Each connected ICU has one IQ, while one MXM may connect multiple ICUs.
    // Its load and compute ICUs can therefore dispatch in the same cycle.
    {
        ftlpu::InstructionControlUnit icu;
        using Port = ftlpu::InstructionControlUnit::MxmIcuPort;
        icu.enqueue_mxm(0, ftlpu::MxmControlInstruction::IW(0));
        icu.enqueue_mxm_control(0, Port::Load, Control::Sync());
        icu.enqueue_mxm(0, ftlpu::MxmControlInstruction::IW(1));
        icu.enqueue_mxm(
            0,
            ftlpu::MxmControlInstruction::Compute(0, 3, 7));

        const auto load = icu.mxm_iq(0, Port::Load).tick();
        const auto compute = icu.mxm_iq(0, Port::Compute).tick();
        assert(load.has_value() && load->opcode == ftlpu::MxmControlOpcode::IW);
        assert(compute.has_value() && compute->opcode == ftlpu::MxmControlOpcode::Compute);

        assert(!icu.mxm_iq(0, Port::Load).tick().has_value());
        assert(icu.mxm_iq(0, Port::Load).blocked_on_sync());
        icu.notify_mxm(0, Port::Load);
        assert(!icu.mxm_iq(0, Port::Load).tick().has_value());
        const auto second_load = icu.mxm_iq(0, Port::Load).tick();
        assert(second_load.has_value());
        assert(second_load->weight_buffer == 1);
    }

    return 0;
}
