#include "ftlpu/system/program_load_command_controller.hpp"

#include <iostream>
#include <stdexcept>

using namespace ftlpu;

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

struct CycleResult {
    ProgramLoadCommandHardwareInterfaceView view{};
};

CycleResult cycle(ProgramLoadCommandController& controller,
    ProgramLoadHostAdapter* adapter, ProgramLoadCommandInputs inputs = {})
{
    if (adapter) adapter->drive(inputs);
    controller.drive(inputs);
    controller.low();
    const auto view = controller.hardware_interface_view();
    if (adapter) adapter->observe(view);
    controller.validate_posedge();
    controller.posedge();
    return {view};
}

void command_ingress_and_controller_sequence()
{
    ProgramLoadCommandController controller(64);
    ProgramImage image{};
    image.source_address = 0x1234;
    image.kind = IcuMacroQueueKind::Mem;
    image.queue_index = 0;
    image.byte_count = 36;
    image.mode = IcuQueueMode::Macro;
    const std::uint16_t destination = 0x000;
    const auto commands = make_program_load_command_list({{image, destination}});
    require(commands.size() == 4,
        "host adapter did not emit LOAD/WAIT/COMMIT/END");

    ProgramLoadHostAdapter adapter;
    adapter.submit(commands);
    const auto reset_cycle = cycle(controller, &adapter);
    require(reset_cycle.view.program_cmd_write_valid_i == false
            && adapter.state() == ProgramLoadHostAdapter::State::WriteCommands,
        "host adapter did not dedicate one cycle to Program Reset");
    for (std::size_t address = 0; address < commands.size(); ++address) {
        const auto result = cycle(controller, &adapter);
        require(result.view.program_cmd_write_valid_i
                && result.view.program_cmd_write_ready_o
                && result.view.program_cmd_write_fire,
            "command RAM write did not fire");
        require(result.view.program_cmd_write_address_i == address,
            "host adapter command address skipped");
        require(result.view.program_cmd_write_opcode_i
                == static_cast<std::uint8_t>(commands[address].opcode),
            "host adapter command payload mismatch");
        require(controller.command_at(address) == commands[address],
            "command RAM did not commit written record");
    }
    const auto start = cycle(controller, &adapter);
    require(start.view.program_load_start_valid_i
            && start.view.program_load_start_ready_o
            && start.view.program_load_start_fire
            && start.view.program_load_start_command_count_i == commands.size(),
        "load start did not fire after command writes");
    require(adapter.complete() && controller.busy(),
        "load start did not transfer ownership to controller");

    cycle(controller, nullptr); // Fetch PC 0 into the active descriptor.
    require(controller.active_valid() && controller.command_pc() == 0,
        "controller did not fetch PC 0 into the active descriptor");

    ProgramLoadCommandInputs blocked{};
    blocked.ctrl_load_txn_ready_i = false;
    const auto held0 = cycle(controller, nullptr, blocked);
    require(held0.view.ctrl_load_txn_valid_o
            && !held0.view.ctrl_load_txn_fire,
        "LOAD_IMEM did not hold valid under backpressure");
    const auto held_payload = held0.view.active_command;
    const auto held1 = cycle(controller, nullptr, blocked);
    require(held1.view.ctrl_load_txn_valid_o
            && held1.view.active_command == held_payload
            && controller.command_pc() == 0,
        "active descriptor changed while ready was low");

    ProgramLoadCommandInputs accept{};
    accept.ctrl_load_txn_ready_i = true;
    const auto load = cycle(controller, nullptr, accept);
    require(load.view.ctrl_load_txn_fire
            && load.view.ctrl_load_txn_source_address_o == image.source_address
            && load.view.ctrl_load_txn_byte_count_o == image.byte_count
            && load.view.ctrl_load_txn_expected_word_count_o == 3
            && load.view.ctrl_load_txn_dma_id_o == 0,
        "LOAD_IMEM action payload/fire mismatch");
    require(controller.command_pc() == 1 && controller.next_dma_id() == 1,
        "LOAD_IMEM retirement did not advance PC/DMA ID");

    cycle(controller, nullptr); // Fetch WAIT_DMA.
    ProgramLoadCommandInputs wait{};
    wait.load_pipeline_idle_i = false;
    cycle(controller, nullptr, wait);
    require(controller.command_pc() == 1,
        "WAIT_DMA advanced before pipeline idle");
    wait.load_pipeline_idle_i = true;
    cycle(controller, nullptr, wait);
    require(controller.command_pc() == 2,
        "WAIT_DMA did not retire on pipeline idle");

    cycle(controller, nullptr); // Fetch COMMIT_QUEUE.
    ProgramLoadCommandInputs prime{};
    prime.ctrl_prime_ready_i = false;
    const auto prime_hold = cycle(controller, nullptr, prime);
    require(prime_hold.view.ctrl_prime_valid_o
            && !prime_hold.view.ctrl_prime_fire
            && controller.command_pc() == 2,
        "COMMIT_QUEUE did not hold under backpressure");
    prime.ctrl_prime_ready_i = true;
    const auto prime_fire = cycle(controller, nullptr, prime);
    require(prime_fire.view.ctrl_prime_fire
            && prime_fire.view.ctrl_prime_destination_o == destination
            && controller.command_pc() == 3,
        "COMMIT_QUEUE did not retire on ready");

    cycle(controller, nullptr); // Fetch END.
    cycle(controller, nullptr); // END retires without an external action.
    require(controller.done() && !controller.busy() && !controller.fault(),
        "END did not produce clean done state");
}

void config_fifo_contract()
{
    ProgramLoadConfigFifo fifo(2);
    const ProgramLoadConfigDescriptor a{0x1000, 36,
        ProgramLoadScope::East, 0x000, ProgramLoadFormat::Word96,
        0, true, true, 3, 7};
    const ProgramLoadConfigDescriptor b{0x2000, 32,
        ProgramLoadScope::West, 0x901, ProgramLoadFormat::Word128,
        4, true, true, 2, 8};
    const ProgramLoadConfigDescriptor c{0x3000, 48,
        ProgramLoadScope::East, 0x002, ProgramLoadFormat::Word96,
        8, true, true, 4, 9};

    const auto step = [&](ProgramLoadConfigFifoInputs inputs) {
        fifo.drive(std::move(inputs));
        fifo.low();
        const auto view = fifo.hardware_interface_view();
        fifo.validate_posedge();
        fifo.posedge();
        return view;
    };

    step({false, true, a, false});
    step({false, true, b, false});
    require(fifo.full() && fifo.front() == a,
        "Config FIFO did not preserve enqueue order");

    ProgramLoadConfigFifoInputs replacement{};
    replacement.enqueue_valid_i = true;
    replacement.enqueue_data_i = c;
    replacement.dequeue_ready_i = true;
    const auto replace = step(replacement);
    require(replace.full_o && replace.dequeue_valid_o
            && replace.dequeue_data_o == a
            && replace.dequeue_fire && replace.enqueue_ready_o
            && replace.enqueue_fire && fifo.occupancy() == 2
            && fifo.front() == b,
        "full Config FIFO replacement contract mismatch");

    ProgramLoadConfigFifoInputs hold{};
    hold.dequeue_ready_i = false;
    const auto held0 = step(hold);
    const auto held1 = step(hold);
    require(held0.dequeue_valid_o && held1.dequeue_valid_o
            && held0.dequeue_data_o == b && held1.dequeue_data_o == b,
        "Config FIFO head changed under backpressure");

    ProgramLoadConfigFifoInputs reset{};
    reset.reset_i = true;
    step(reset);
    require(fifo.empty() && fifo.read_pointer() == 0
            && fifo.write_pointer() == 0,
        "Config FIFO reset did not clear level/pointers");
}

void illegal_start_and_reset()
{
    ProgramLoadCommandController controller(64);
    ProgramLoadCommandInputs start{};
    start.program_load_start_valid_i = true;
    start.program_load_start_command_count_i = 0;
    const auto illegal = cycle(controller, nullptr, start);
    require(illegal.view.program_load_start_fire && illegal.view.fault_o,
        "zero command count did not flag same-cycle command fault");
    require(controller.fault() && !controller.busy(),
        "illegal start was not sticky after commit");

    ProgramLoadCommandInputs reset{};
    reset.reset_i = true;
    cycle(controller, nullptr, reset);
    require(!controller.fault() && !controller.done() && !controller.busy(),
        "reset did not clear controller state");
}

} // namespace

int main()
{
    try {
        command_ingress_and_controller_sequence();
        config_fifo_contract();
        illegal_start_and_reset();
        std::cout << "program_load_command_controller_test passed\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "program_load_command_controller_test failed: "
                  << ex.what() << '\n';
        return 1;
    }
}
