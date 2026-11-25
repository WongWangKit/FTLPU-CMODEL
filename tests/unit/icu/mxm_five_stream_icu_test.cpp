#include "ftlpu/system/icu.hpp"
#include "ftlpu/icu/qctl_distributed_program_store.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

namespace {

using namespace ftlpu;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void check_qctl_five_stream_routes()
{
    IcuQctlDistributedProgramStore store;
    store.configure(32, IcuQctlDistributedProgramStore::kQueueCount,
        2, 64, 2);

    constexpr std::array<std::size_t, 10> queues{
        208, 212, 216, 220, 224, // East MXM0: Load/Dequant/Compute/Param/Backend.
        210, 214, 218, 222, 226, // West MXM2: Load/Dequant/Compute/Param/Backend.
    };

    for (std::size_t index = 0; index < queues.size(); ++index) {
        IcuQctlDistributedProgramStore::Inputs input{};
        input.partition_valid = true;
        input.partition_queue = queues[index];
        input.partition_base = index;
        input.partition_count = 1;
        input.partition_generation = index + 1;

        bool accepted = false;
        for (std::size_t cycle = 0; cycle < 32 && !accepted; ++cycle) {
            const auto& preview = store.begin_cycle(input);
            accepted = preview.partition_fire;
            store.complete_cycle();
        }
        require(accepted, "MXM QCTL partition was not accepted");
        for (std::size_t cycle = 0;
             cycle < 16 && !store.configured(queues[index]); ++cycle) {
            store.begin_cycle({});
            store.complete_cycle();
        }
        require(store.configured(queues[index]),
            "MXM QCTL partition did not reach its regional store");
    }

    for (std::size_t index = 0; index < queues.size(); ++index) {
        IcuQctlDistributedProgramStore::Inputs input{};
        input.load_valid = true;
        input.load_address = index;
        input.load_data = IcuQueueControl{
            std::uint32_t(index + 3), IcuControlOpcode::Sync,
            std::uint32_t(0x40 + index)};

        bool accepted = false;
        for (std::size_t cycle = 0; cycle < 32 && !accepted; ++cycle) {
            const auto& preview = store.begin_cycle(input);
            accepted = preview.load_fire;
            store.complete_cycle();
        }
        require(accepted, "MXM QCTL record was not accepted");
        for (std::size_t cycle = 0;
             cycle < 16 && !store.loaded(queues[index]); ++cycle) {
            store.begin_cycle({});
            store.complete_cycle();
        }
        const auto head = store.head(queues[index], 0, index + 1);
        require(store.loaded(queues[index]) && head.valid && !head.error,
            "MXM QCTL record did not reach the selected endpoint");
        require(head.record.cycle == index + 3
                && head.record.operand == 0x40 + index,
            "MXM QCTL record changed while crossing the relay chain");
    }
}

std::vector<IcuRawImemWord128> make_single_context_image(
    std::uint32_t start_cycle,
    std::uint64_t native_instruction,
    std::size_t native_bits,
    std::size_t operand_bits)
{
    std::vector<IcuRawImemWord128> image(2);
    image[0].lanes[0] = 0x15u; // MacroPacked v1, one descriptor.
    std::size_t offset = 128;
    auto push = [&](std::uint64_t value, std::size_t width) {
        for (std::size_t bit = 0; bit < width; ++bit, ++offset) {
            if (((value >> bit) & 1u) == 0) continue;
            image[offset / 128].lanes[(offset % 128) / 32]
                |= std::uint32_t{1} << (offset % 32);
        }
    };
    push(0, 3);                  // Empty Dictionary.
    push(start_cycle, 32);       // Absolute start cycle.
    push(0, operand_bits);       // Initial induction operand.
    push(0, 1);                  // One-template Run.
    push(0, 1);                  // Compact Template.
    push(native_instruction, native_bits);
    push(0, 2);                  // Single schedule shape.
    push(0, 2);                  // No induction target.
    return image;
}

template <typename Queue, typename Instruction>
void check_stream(Queue& queue,
    IcuMacroQueueKind kind,
    const std::vector<IcuRawImemWord128>& image,
    std::size_t expected_cycle,
    const Instruction& expected)
{
    queue.load_raw_macro(kind, image);
    for (std::size_t cycles = 0; !queue.raw_macro_ready(); ++cycles) {
        if (cycles >= 64) {
            std::cerr << "prime diagnostic kind=" << static_cast<unsigned>(kind)
                      << " fetch_complete=" << queue.macro_fetch_complete()
                      << " decoder_state="
                      << static_cast<unsigned>(queue.macro_decoder_state_code())
                      << " decoder_phase="
                      << static_cast<unsigned>(queue.macro_decoder_phase_code())
                      << " fifo=" << queue.context_fifo_occupancy()
                      << " active=" << queue.active_macro_count()
                      << " fault=" << queue.macro_decoder_fault() << '\n';
            require(false, "MXM Raw-Macro frontend did not prime");
        }
        queue.prefetch_raw_macro();
    }

    std::optional<Instruction> observed;
    std::optional<std::size_t> observed_cycle;
    for (std::size_t cycle = 0; cycle <= expected_cycle + 8; ++cycle) {
        const auto issue = queue.dispatch_macro_issue(true);
        const auto& pins = queue.macro_issue_port_signals();
        require(pins.issue_fire == issue.has_value(),
            "MXM issue Fire and returned instruction disagree");
        if (!issue) continue;
        require(!observed, "MXM stream issued more than one command");
        observed = issue;
        observed_cycle = cycle;
    }
    require(observed_cycle == expected_cycle,
        "MXM command issued on the wrong absolute cycle");
    require(observed == expected,
        "MXM command bits changed across the Raw-Macro frontend");
    require(!queue.macro_issue_port_signals().fault_o,
        "MXM Macro Scheduler reported a fault");
}

} // namespace

int main()
{
    try {
        InstructionControlUnit::MxmLoadIcu load;
        InstructionControlUnit::MxmComputeIcu compute;
        InstructionControlUnit::MxmDequantIcu dequant;
        InstructionControlUnit::MxmParamLoadIcu param;
        InstructionControlUnit::MxmBackendIcu backend;

        check_stream(load, IcuMacroQueueKind::MxmLoad,
            make_single_context_image(3, 0x2, 2, 1), 3,
            MxmLoadCommand{true, false});
        check_stream(compute, IcuMacroQueueKind::MxmCompute,
            make_single_context_image(4, 0x7, 7, 6), 4,
            MxmComputeCommand{3, true});
        check_stream(dequant, IcuMacroQueueKind::MxmDequant,
            make_single_context_image(3, 0x1, 1, 1), 3,
            MxmDequantCommand{true});
        check_stream(param, IcuMacroQueueKind::MxmParamLoad,
            make_single_context_image(6, 0x1, 1, 1), 6,
            MxmParamLoadCommand{true});
        check_stream(backend, IcuMacroQueueKind::MxmBackend,
            make_single_context_image(7, 0x8, 13, 10), 7,
            MxmBackendCommand{false, 2, MxmBackendOperation::DirectOut});

        check_qctl_five_stream_routes();

        std::cout << "MXM_FIVE_STREAM_ICU_PASS streams=5 qctl_routes=10"
                     " qctl_queues=228\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mxm_five_stream_icu_test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
