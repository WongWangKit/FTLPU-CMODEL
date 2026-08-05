#include "ftlpu/icu/icu.hpp"
#include "icu_timing_report.hpp"

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

using TestIcu = ftlpu::DistributedIcuQueue<
    TestInstruction,
    40,
    32,
    4,
    1>;
using Entry = TestIcu::Entry;
using Control = ftlpu::IcuControlInstruction;

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

Entry function(int opcode, std::size_t value = 0)
{
    return Entry {
        std::in_place_type<TestInstruction>,
        TestInstruction {opcode, value}};
}

Entry control(Control instruction)
{
    return Entry {
        std::in_place_type<Control>, instruction};
}

} // namespace

int main()
{
    // Width, local i-MEM depth, IQ depth and fetch latency are independent
    // compile-time hardware parameters.
    static_assert(TestIcu::instruction_bits == 40);
    static_assert(TestIcu::imem_depth == 32);
    static_assert(TestIcu::iq_depth == 4);
    static_assert(TestIcu::fetch_latency == 1);
    static_assert(
        ftlpu::InstructionControlUnit::MxmIcu::instruction_bits == 32);
    static_assert(
        ftlpu::InstructionControlUnit::VxmIcu::instruction_bits == 96);
    static_assert(
        ftlpu::InstructionControlUnit::MemIcu::imem_depth
        != ftlpu::InstructionControlUnit::VxmIcu::imem_depth);

    // Compiler start_cycle leaves enough time for local i-MEM reads to fill
    // the IQ. NOP and Repeat then determine exact issue cycles.
    {
        TestIcu icu;
        icu.load_imem(0, {
            function(1, 100),
            control(Control::Nop(2)),
            function(2, 200),
            control(Control::Repeat(2, 2)),
        });
        icu.configure({0, 4, 3});

        assert(!icu.tick().has_value()); // cycle 0: prefetch
        assert(!icu.tick().has_value()); // cycle 1: prefetch
        assert(!icu.tick().has_value()); // cycle 2: prefetch
        assert((icu.tick() == TestInstruction {1, 100})); // cycle 3
        assert(!icu.tick().has_value());                  // cycle 4: NOP 1
        assert(!icu.tick().has_value());                  // cycle 5: NOP 2
        assert((icu.tick() == TestInstruction {2, 200})); // cycle 6
        assert(!icu.tick().has_value());                  // cycle 7: gap
        assert((icu.tick() == TestInstruction {2, 200})); // cycle 8
        assert(!icu.tick().has_value());                  // cycle 9: gap
        assert((icu.tick() == TestInstruction {2, 200})); // cycle 10
        assert(icu.done());
        assert(icu.fetched_count() == 4);
        assert(icu.issued_count() == 4);
        assert(!icu.underflowed());
    }

    // A start cycle that precedes the one-cycle local fetch is a compiler
    // schedule error; hardware does not silently insert a data-dependent NOP.
    {
        TestIcu icu;
        icu.load_imem(0, {function(3)});
        icu.configure({0, 1, 0});
        assert(throws([&] { (void)icu.tick(); }));
        assert(icu.underflowed());
    }

    // Finite i-MEM and descriptor ranges fail fast.
    {
        using TinyIcu = ftlpu::DistributedIcuQueue<TestInstruction, 32, 2, 1>;
        TinyIcu icu;
        icu.write_imem(0, TestInstruction {1, 0});
        icu.write_imem(1, TestInstruction {2, 0});
        assert(throws([&] {
            icu.write_imem(2, TestInstruction {3, 0});
        }));
        assert(throws([&] {
            icu.configure({1, 2, 4});
        }));
    }

    // Sync remains ordered with functional instructions in the same IQ.
    {
        TestIcu icu;
        icu.load_imem(0, {
            function(4),
            control(Control::Sync()),
            function(5),
        });
        icu.configure({0, 3, 3});
        for (std::size_t cycle = 0; cycle < 3; ++cycle)
            assert(!icu.tick().has_value());
        assert(icu.tick()->opcode == 4);
        assert(!icu.tick().has_value());
        assert(icu.blocked_on_sync());
        icu.notify();
        assert(!icu.tick().has_value());
        assert(icu.tick()->opcode == 5);
    }

    // MEM Repeat keeps its compiler-encoded address stride.
    {
        using MemIcu = ftlpu::DistributedIcuQueue<
            ftlpu::MemInstruction, 96, 8, 4>;
        MemIcu icu;
        icu.append_program(
            ftlpu::MemInstruction::Read(10, ftlpu::StreamId::East(1)));
        icu.append_control(Control::Repeat(2, 1, 4));
        assert(icu.tick()->address == ftlpu::MemLocalWordAddress13(10));
        assert(icu.tick()->address == ftlpu::MemLocalWordAddress13(14));
        assert(icu.tick()->address == ftlpu::MemLocalWordAddress13(18));
    }

    // MXM load, compute, and activation dequantize have separate local
    // i-MEM/IQ endpoints and can issue in the same logical VLIW cycle.
    {
        ftlpu::InstructionControlUnit icu;
        using Port = ftlpu::InstructionControlUnit::MxmIcuPort;
        icu.enqueue_mxm(0, ftlpu::MxmControlInstruction::IW(0));
        icu.enqueue_mxm(
            0, ftlpu::MxmControlInstruction::Compute(0, 6));
        icu.enqueue_mxm(
            0, ftlpu::MxmControlInstruction::ActivationDequantize());

        const auto load = icu.mxm_iq(0, Port::Load).tick();
        const auto compute = icu.mxm_iq(0, Port::Compute).tick();
        const auto dequant = icu.mxm_iq(0, Port::Dequant).tick();
        assert(load.has_value());
        assert(compute.has_value());
        assert(dequant.has_value());
        assert(load->opcode == ftlpu::MxmControlOpcode::IW);
        assert(compute->opcode == ftlpu::MxmControlOpcode::Compute);
        assert(dequant->opcode
            == ftlpu::MxmControlOpcode::ActivationDequantize);
    }

    // SXM transpose and permute are independent physical paths and therefore
    // also have independent local ICU endpoints.
    {
        ftlpu::InstructionControlUnit icu;
        using Port = ftlpu::InstructionControlUnit::SxmIcuPort;
        const auto streams = ftlpu::SxmInstruction::StreamList{
            ftlpu::SxmStreamId{0}};
        icu.enqueue_sxm(
            ftlpu::Hemisphere::East,
            ftlpu::SxmInstruction::Transpose(streams, streams, 1));
        icu.enqueue_sxm(
            ftlpu::Hemisphere::East,
            ftlpu::SxmInstruction::Permute(
                streams,
                streams,
                ftlpu::Permute320::identity_map(),
                ftlpu::SxmWeightLayout::VectorColumns,
                1));

        const auto transpose = icu.sxm_iq(0, Port::Transpose).tick();
        const auto permute = icu.sxm_iq(0, Port::Permute).tick();
        assert(transpose.has_value());
        assert(permute.has_value());
        assert(transpose->opcode == ftlpu::SxmOpcode::Transpose);
        assert(permute->opcode == ftlpu::SxmOpcode::Permute);
    }

    const auto timing = icu_timing_test::run_timing_scenario();
    assert(timing.passed);
    icu_timing_test::write_reports(timing);

    return 0;
}
