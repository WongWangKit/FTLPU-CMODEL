#pragma once

#include "ftlpu/core/timing.hpp"
#include "ftlpu/system/program_image.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ftlpu {

// Revision-bound Program Load command encoding. These values mirror the
// canonical lpu_program_load_pkg opcode values; the command RAM capacity is a
// constructor argument so the controller is not coupled to a Config FIFO.
enum class ProgramLoadCommandOpcode : std::uint8_t {
    LoadImem = 0,
    WaitDma = 1,
    CommitQueue = 2,
    Next = 3,
    Jump = 4,
    End = 5,
};

enum class ProgramLoadFormat : std::uint8_t { Word96 = 0, Word128 = 1 };
enum class ProgramLoadEncodingMode : std::uint8_t {
    Macro = 0, Fu3d = 1, Reserved2 = 2, Reserved3 = 3,
};
enum class ProgramLoadScope : std::uint8_t { East = 0, West = 1, Both = 2 };
enum class ProgramLoadFaultCode : std::uint8_t {
    None = 0, Descriptor = 1, Route = 8, Command = 10,
};

struct ProgramLoadCommand {
    ProgramLoadCommandOpcode opcode{ProgramLoadCommandOpcode::LoadImem};
    std::uint64_t source_address{0};
    std::uint32_t byte_count{0};
    std::uint16_t destination{0};
    ProgramLoadFormat format{ProgramLoadFormat::Word96};
    ProgramLoadEncodingMode encoding_mode{ProgramLoadEncodingMode::Macro};
    std::uint32_t imem_base_address{0};
    bool image_first{false};
    bool image_last{false};
    std::uint16_t expected_word_count{0};
    std::uint8_t jump_address{0};
};

inline bool operator==(const ProgramLoadCommand& lhs,
    const ProgramLoadCommand& rhs) noexcept
{
    return lhs.opcode == rhs.opcode
        && lhs.source_address == rhs.source_address
        && lhs.byte_count == rhs.byte_count
        && lhs.destination == rhs.destination
        && lhs.format == rhs.format
        && lhs.encoding_mode == rhs.encoding_mode
        && lhs.imem_base_address == rhs.imem_base_address
        && lhs.image_first == rhs.image_first
        && lhs.image_last == rhs.image_last
        && lhs.expected_word_count == rhs.expected_word_count
        && lhs.jump_address == rhs.jump_address;
}

inline bool operator!=(const ProgramLoadCommand& lhs,
    const ProgramLoadCommand& rhs) noexcept
{
    return !(lhs == rhs);
}

struct ProgramLoadCommandInputs {
    bool reset_i{false};
    bool program_cmd_write_enable_i{true};
    bool program_load_start_enable_i{true};
    bool program_cmd_write_valid_i{false};
    std::uint8_t program_cmd_write_address_i{0};
    ProgramLoadCommand program_cmd_write_payload_i{};
    bool program_load_start_valid_i{false};
    std::uint16_t program_load_start_command_count_i{0};
    bool ctrl_load_txn_ready_i{false};
    bool ctrl_prime_ready_i{false};
    bool load_pipeline_idle_i{false};
};

// PRE_POSEDGE hardware-interface view. It exposes the flattened canonical
// fields from the real controller owner rather than reconstructing a shadow
// monitor after the edge.
struct ProgramLoadCommandHardwareInterfaceView {
    bool program_cmd_write_valid_i{false};
    bool program_cmd_write_ready_o{false};
    bool program_cmd_write_fire{false};
    std::uint8_t program_cmd_write_address_i{0};
    std::uint8_t program_cmd_write_opcode_i{0};
    std::uint64_t program_cmd_write_source_address_i{0};
    std::uint32_t program_cmd_write_byte_count_i{0};
    std::uint16_t program_cmd_write_destination_i{0};
    std::uint8_t program_cmd_write_format_i{0};
    std::uint8_t program_cmd_write_encoding_mode_i{0};
    std::uint32_t program_cmd_write_imem_base_address_i{0};
    bool program_cmd_write_image_first_i{false};
    bool program_cmd_write_image_last_i{false};
    std::uint16_t program_cmd_write_expected_word_count_i{0};
    std::uint8_t program_cmd_write_jump_address_i{0};
    bool program_load_start_valid_i{false};
    bool program_load_start_ready_o{false};
    bool program_load_start_fire{false};
    std::uint16_t program_load_start_command_count_i{0};

    bool ctrl_load_txn_valid_o{false};
    bool ctrl_load_txn_ready_i{false};
    bool ctrl_load_txn_fire{false};
    std::uint64_t ctrl_load_txn_source_address_o{0};
    std::uint32_t ctrl_load_txn_byte_count_o{0};
    ProgramLoadScope ctrl_load_txn_scope_o{ProgramLoadScope::East};
    std::uint16_t ctrl_load_txn_destination_o{0};
    ProgramLoadFormat ctrl_load_txn_format_o{ProgramLoadFormat::Word96};
    std::uint32_t ctrl_load_txn_imem_base_address_o{0};
    bool ctrl_load_txn_image_first_o{false};
    bool ctrl_load_txn_image_last_o{false};
    std::uint16_t ctrl_load_txn_expected_word_count_o{0};
    std::uint8_t ctrl_load_txn_dma_id_o{0};
    bool ctrl_prime_valid_o{false};
    bool ctrl_prime_ready_i{false};
    bool ctrl_prime_fire{false};
    ProgramLoadScope ctrl_prime_scope_o{ProgramLoadScope::East};
    std::uint16_t ctrl_prime_destination_o{0};

    bool load_pipeline_idle_i{false};
    std::uint8_t command_pc_o{0};
    bool active_valid_o{false};
    ProgramLoadCommand active_command{};
    bool busy_o{false};
    bool done_o{false};
    bool fault_o{false};
    ProgramLoadFaultCode fault_code_o{ProgramLoadFaultCode::None};
};

class ProgramLoadCommandController final : public timing::ClockedModule {
public:
    static constexpr std::size_t kRevisionCommandDepth = 64;
    static constexpr std::uint16_t kDestinationMask = 0x1fff;
    static constexpr std::uint32_t kImemAddressMask = 0x1ffff;
    static constexpr std::uint8_t kCommandAddressMask = 0x3f;

    explicit ProgramLoadCommandController(
        std::size_t command_depth = kRevisionCommandDepth)
        : current_(make_reset_state(command_depth))
    {
        if (command_depth == 0 || command_depth > kRevisionCommandDepth)
            throw std::invalid_argument(
                "Program Load command depth must be in [1,64]");
    }

    std::size_t command_depth() const noexcept
    {
        return current_.command_mem.size();
    }

    void drive(const ProgramLoadCommandInputs& inputs)
    {
        if (driven_)
            throw std::logic_error(
                "Program Load command inputs driven more than once in LOW");
        validate_widths(inputs);
        inputs_ = inputs;
        driven_ = true;
    }

    const ProgramLoadCommandHardwareInterfaceView&
    hardware_interface_view() const noexcept
    {
        return view_;
    }

    bool ctrl_prime_valid_current() const noexcept
    {
        return current_.active_valid && !current_.fault
            && current_.active_record.opcode
                == ProgramLoadCommandOpcode::CommitQueue;
    }

    void low() override
    {
        if (!driven_)
            throw std::logic_error(
                "Program Load command controller LOW has no driven inputs");
        if (next_.has_value())
            throw std::logic_error(
                "Program Load command controller LOW executed twice");

        view_ = build_view(current_, inputs_);
        auto next = current_;
        if (inputs_.reset_i) {
            const auto memory = std::move(next.command_mem);
            next = make_reset_state(memory.size());
            next.command_mem = memory; // RTL reset does not clear command RAM.
            next_ = std::move(next);
            return;
        }

        if (view_.program_cmd_write_fire)
            next.command_mem.at(inputs_.program_cmd_write_address_i) =
                inputs_.program_cmd_write_payload_i;

        const bool start_illegal = view_.program_load_start_fire
            && (inputs_.program_load_start_command_count_i == 0
                || inputs_.program_load_start_command_count_i
                    > current_.command_mem.size());
        const bool fetch_command = current_.busy && !current_.active_valid
            && current_.command_pc < current_.command_count;
        const bool active_illegal = active_command_illegal(current_);
        const bool exhausted_without_end = current_.busy
            && !current_.active_valid
            && current_.command_pc >= current_.command_count;
        const bool active_complete = current_.active_valid && (
            (current_.active_record.opcode
                    == ProgramLoadCommandOpcode::LoadImem
                && view_.ctrl_load_txn_fire)
            || (current_.active_record.opcode
                    == ProgramLoadCommandOpcode::WaitDma
                && inputs_.load_pipeline_idle_i)
            || (current_.active_record.opcode
                    == ProgramLoadCommandOpcode::CommitQueue
                && view_.ctrl_prime_fire)
            || current_.active_record.opcode == ProgramLoadCommandOpcode::Next
            || (current_.active_record.opcode
                    == ProgramLoadCommandOpcode::Jump
                && !jump_target_illegal(current_))
            || current_.active_record.opcode == ProgramLoadCommandOpcode::End);

        if (view_.program_load_start_fire) {
            next.command_pc = 0;
            next.command_count = inputs_.program_load_start_command_count_i;
            next.active_valid = false;
            next.busy = !start_illegal;
            next.done = false;
            next.next_dma_id = 0;
        } else if (current_.busy) {
            if (fetch_command) {
                next.active_record =
                    current_.command_mem.at(current_.command_pc);
                next.active_valid = true;
            }
            if (active_illegal || exhausted_without_end) {
                next.active_valid = false;
                next.busy = false;
            } else if (active_complete) {
                next.active_valid = false;
                switch (current_.active_record.opcode) {
                case ProgramLoadCommandOpcode::Jump:
                    next.command_pc = current_.active_record.jump_address;
                    break;
                case ProgramLoadCommandOpcode::End:
                    next.busy = false;
                    next.done = true;
                    break;
                default:
                    ++next.command_pc;
                    break;
                }
                if (current_.active_record.opcode
                    == ProgramLoadCommandOpcode::LoadImem)
                    ++next.next_dma_id;
            }
        }

        if (!current_.fault
            && (start_illegal || active_illegal || exhausted_without_end)) {
            next.fault = true;
            next.fault_code = ProgramLoadFaultCode::Command;
        }
        next_ = std::move(next);
    }

    void validate_posedge() const override
    {
        if (!driven_ || !next_.has_value())
            throw std::logic_error(
                "Program Load command controller commit requires one LOW");
    }

    void posedge() override
    {
        validate_posedge();
        current_ = std::move(*next_);
        next_.reset();
        inputs_ = {};
        driven_ = false;
    }

    void high() override {}

    void reset()
    {
        if (next_.has_value())
            throw std::logic_error(
                "Program Load command reset cannot interrupt LOW");
        const auto memory = std::move(current_.command_mem);
        current_ = make_reset_state(memory.size());
        current_.command_mem = memory;
        inputs_ = {};
        view_ = {};
        driven_ = false;
    }

    bool busy() const noexcept { return current_.busy; }
    bool done() const noexcept { return current_.done; }
    bool fault() const noexcept { return current_.fault; }
    std::uint8_t command_pc() const noexcept { return current_.command_pc; }
    bool active_valid() const noexcept { return current_.active_valid; }
    std::uint8_t next_dma_id() const noexcept { return current_.next_dma_id; }

    const ProgramLoadCommand& command_at(std::size_t address) const
    {
        return current_.command_mem.at(address);
    }

private:
    struct State {
        std::vector<ProgramLoadCommand> command_mem{};
        ProgramLoadCommand active_record{};
        std::uint8_t command_pc{0};
        std::uint16_t command_count{0};
        bool active_valid{false};
        bool busy{false};
        bool done{false};
        std::uint8_t next_dma_id{0};
        bool fault{false};
        ProgramLoadFaultCode fault_code{ProgramLoadFaultCode::None};
    };

    static State make_reset_state(std::size_t depth)
    {
        State state{};
        state.command_mem.resize(depth);
        return state;
    }

    static ProgramLoadScope scope_from_destination(
        std::uint16_t destination) noexcept
    {
        return (destination & 0x1000) != 0
            ? ProgramLoadScope::West : ProgramLoadScope::East;
    }

    static bool jump_target_illegal(const State& state) noexcept
    {
        return state.active_valid
            && state.active_record.opcode == ProgramLoadCommandOpcode::Jump
            && state.active_record.jump_address >= state.command_count;
    }

    static bool active_command_illegal(const State& state) noexcept
    {
        return state.active_valid
            && (static_cast<std::uint8_t>(state.active_record.opcode)
                    > static_cast<std::uint8_t>(ProgramLoadCommandOpcode::End)
                || jump_target_illegal(state));
    }

    static void validate_widths(const ProgramLoadCommandInputs& inputs)
    {
        const auto& command = inputs.program_cmd_write_payload_i;
        if (inputs.program_cmd_write_address_i > kCommandAddressMask
            || static_cast<std::uint8_t>(command.opcode) > 7
            || command.destination > kDestinationMask
            || static_cast<std::uint8_t>(command.format) > 3
            || static_cast<std::uint8_t>(command.encoding_mode) > 3
            || command.imem_base_address > kImemAddressMask
            || command.jump_address > kCommandAddressMask
            || inputs.program_load_start_command_count_i > 127)
            throw std::invalid_argument(
                "Program Load command input exceeds canonical field width");
    }

    static ProgramLoadCommandHardwareInterfaceView build_view(
        const State& state, const ProgramLoadCommandInputs& inputs)
    {
        ProgramLoadCommandHardwareInterfaceView view{};
        const auto& write = inputs.program_cmd_write_payload_i;
        view.program_cmd_write_valid_i = inputs.program_cmd_write_valid_i;
        view.program_cmd_write_ready_o = !state.busy
            && inputs.program_cmd_write_enable_i;
        view.program_cmd_write_fire = view.program_cmd_write_valid_i
            && view.program_cmd_write_ready_o;
        view.program_cmd_write_address_i = inputs.program_cmd_write_address_i;
        view.program_cmd_write_opcode_i =
            static_cast<std::uint8_t>(write.opcode);
        view.program_cmd_write_source_address_i = write.source_address;
        view.program_cmd_write_byte_count_i = write.byte_count;
        view.program_cmd_write_destination_i = write.destination;
        view.program_cmd_write_format_i =
            static_cast<std::uint8_t>(write.format);
        view.program_cmd_write_encoding_mode_i =
            static_cast<std::uint8_t>(write.encoding_mode);
        view.program_cmd_write_imem_base_address_i = write.imem_base_address;
        view.program_cmd_write_image_first_i = write.image_first;
        view.program_cmd_write_image_last_i = write.image_last;
        view.program_cmd_write_expected_word_count_i =
            write.expected_word_count;
        view.program_cmd_write_jump_address_i = write.jump_address;

        view.program_load_start_valid_i = inputs.program_load_start_valid_i;
        view.program_load_start_ready_o = !state.busy
            && inputs.program_load_start_enable_i;
        view.program_load_start_fire = view.program_load_start_valid_i
            && view.program_load_start_ready_o;
        view.program_load_start_command_count_i =
            inputs.program_load_start_command_count_i;

        view.ctrl_load_txn_valid_o = state.active_valid
            && state.active_record.opcode
                == ProgramLoadCommandOpcode::LoadImem
            && !state.fault;
        view.ctrl_load_txn_ready_i = inputs.ctrl_load_txn_ready_i;
        view.ctrl_load_txn_fire = view.ctrl_load_txn_valid_o
            && view.ctrl_load_txn_ready_i;
        view.ctrl_load_txn_source_address_o = state.active_record.source_address;
        view.ctrl_load_txn_byte_count_o = state.active_record.byte_count;
        view.ctrl_load_txn_scope_o =
            scope_from_destination(state.active_record.destination);
        view.ctrl_load_txn_destination_o = state.active_record.destination;
        view.ctrl_load_txn_format_o = state.active_record.format;
        view.ctrl_load_txn_imem_base_address_o =
            state.active_record.imem_base_address;
        view.ctrl_load_txn_image_first_o = state.active_record.image_first;
        view.ctrl_load_txn_image_last_o = state.active_record.image_last;
        view.ctrl_load_txn_expected_word_count_o =
            state.active_record.expected_word_count;
        view.ctrl_load_txn_dma_id_o = state.next_dma_id;

        view.ctrl_prime_valid_o = state.active_valid
            && state.active_record.opcode
                == ProgramLoadCommandOpcode::CommitQueue
            && !state.fault;
        view.ctrl_prime_ready_i = inputs.ctrl_prime_ready_i;
        view.ctrl_prime_fire = view.ctrl_prime_valid_o
            && view.ctrl_prime_ready_i;
        view.ctrl_prime_scope_o =
            scope_from_destination(state.active_record.destination);
        view.ctrl_prime_destination_o = state.active_record.destination;

        view.load_pipeline_idle_i = inputs.load_pipeline_idle_i;
        view.command_pc_o = state.command_pc;
        view.active_valid_o = state.active_valid;
        view.active_command = state.active_record;
        view.busy_o = state.busy;
        view.done_o = state.done;

        const bool start_illegal = view.program_load_start_fire
            && (inputs.program_load_start_command_count_i == 0
                || inputs.program_load_start_command_count_i
                    > state.command_mem.size());
        const bool active_illegal = active_command_illegal(state);
        const bool exhausted_without_end = state.busy
            && !state.active_valid && state.command_pc >= state.command_count;
        view.fault_o = state.fault || start_illegal || active_illegal
            || exhausted_without_end;
        view.fault_code_o = view.fault_o
            ? ProgramLoadFaultCode::Command : ProgramLoadFaultCode::None;
        return view;
    }

    State current_{};
    std::optional<State> next_{};
    ProgramLoadCommandInputs inputs_{};
    ProgramLoadCommandHardwareInterfaceView view_{};
    bool driven_{false};
};

struct ProgramLoadConfigDescriptor {
    std::uint64_t source_address{0};
    std::uint32_t byte_count{0};
    ProgramLoadScope scope{ProgramLoadScope::East};
    std::uint16_t destination{0};
    ProgramLoadFormat format{ProgramLoadFormat::Word96};
    std::uint32_t imem_base_address{0};
    bool image_first{false};
    bool image_last{false};
    std::uint16_t expected_word_count{0};
    std::uint8_t dma_id{0};
};

inline bool operator==(const ProgramLoadConfigDescriptor& lhs,
    const ProgramLoadConfigDescriptor& rhs) noexcept
{
    return lhs.source_address == rhs.source_address
        && lhs.byte_count == rhs.byte_count
        && lhs.scope == rhs.scope
        && lhs.destination == rhs.destination
        && lhs.format == rhs.format
        && lhs.imem_base_address == rhs.imem_base_address
        && lhs.image_first == rhs.image_first
        && lhs.image_last == rhs.image_last
        && lhs.expected_word_count == rhs.expected_word_count
        && lhs.dma_id == rhs.dma_id;
}

struct ProgramLoadConfigFifoInputs {
    bool reset_i{false};
    bool enqueue_valid_i{false};
    std::optional<ProgramLoadConfigDescriptor> enqueue_data_i{};
    bool dequeue_ready_i{false};
};

struct ProgramLoadConfigFifoHardwareInterfaceView {
    bool enqueue_valid_i{false};
    bool enqueue_ready_o{false};
    bool enqueue_fire{false};
    std::optional<ProgramLoadConfigDescriptor> enqueue_data_i{};
    bool dequeue_valid_o{false};
    bool dequeue_ready_i{false};
    bool dequeue_fire{false};
    std::optional<ProgramLoadConfigDescriptor> dequeue_data_o{};
    std::size_t level_o{0};
    bool empty_o{true};
    bool full_o{false};
};

// Exact register-FIFO contract used by rtl/common/lpu_fifo.sv for Program
// Load descriptors: combinational head, full replacement, no output prefetch.
class ProgramLoadConfigFifo final : public timing::ClockedModule {
public:
    static constexpr std::size_t kDepth = 16;
    static constexpr std::size_t kDescriptorWidth = 155;

    explicit ProgramLoadConfigFifo(std::size_t depth = kDepth)
        : depth_(depth), current_(make_state(depth))
    {
        if (depth_ == 0)
            throw std::invalid_argument(
                "Program Load Config FIFO depth must be non-zero");
    }

    void drive(ProgramLoadConfigFifoInputs inputs)
    {
        if (driven_ || next_.has_value())
            throw std::logic_error(
                "Program Load Config FIFO inputs driven outside LOW window");
        if (inputs.enqueue_valid_i != inputs.enqueue_data_i.has_value())
            throw std::invalid_argument(
                "Program Load Config FIFO valid/data mismatch");
        inputs_ = std::move(inputs);
        driven_ = true;
    }

    void drive_idle() { drive({}); }

    const ProgramLoadConfigFifoHardwareInterfaceView&
    hardware_interface_view() const noexcept
    {
        return view_;
    }

    std::size_t occupancy() const noexcept { return current_.level; }
    bool empty() const noexcept { return current_.level == 0; }
    bool full() const noexcept { return current_.level == depth_; }
    std::size_t read_pointer() const noexcept { return current_.read_pointer; }
    std::size_t write_pointer() const noexcept { return current_.write_pointer; }

    const ProgramLoadConfigDescriptor& front() const
    {
        if (empty())
            throw std::logic_error("Program Load Config FIFO front on empty");
        const auto& value = current_.storage.at(current_.read_pointer);
        if (!value)
            throw std::logic_error(
                "Program Load Config FIFO occupied slot has no descriptor");
        return *value;
    }

    void low() override
    {
        if (!driven_) drive_idle();
        if (next_.has_value())
            throw std::logic_error(
                "Program Load Config FIFO LOW executed twice");
        view_ = build_view(current_, inputs_, depth_);
        auto next = current_;
        if (inputs_.reset_i) {
            next.read_pointer = 0;
            next.write_pointer = 0;
            next.level = 0;
            next_ = std::move(next); // Physical memory intentionally survives.
            return;
        }
        if (view_.enqueue_fire) {
            next.storage.at(current_.write_pointer) =
                *inputs_.enqueue_data_i;
            next.write_pointer = increment(current_.write_pointer);
        }
        if (view_.dequeue_fire)
            next.read_pointer = increment(current_.read_pointer);
        if (view_.enqueue_fire && !view_.dequeue_fire) ++next.level;
        if (!view_.enqueue_fire && view_.dequeue_fire) --next.level;
        next_ = std::move(next);
    }

    void validate_posedge() const override
    {
        if (!next_.has_value())
            throw std::logic_error(
                "Program Load Config FIFO commit requires one LOW");
        if (next_->level > depth_)
            throw std::logic_error(
                "Program Load Config FIFO Next level exceeds depth");
    }

    void posedge() override
    {
        validate_posedge();
        current_ = std::move(*next_);
        next_.reset();
        inputs_ = {};
        driven_ = false;
    }

    void high() override {}

    void reset()
    {
        if (next_.has_value())
            throw std::logic_error(
                "Program Load Config FIFO reset cannot interrupt LOW");
        current_.read_pointer = 0;
        current_.write_pointer = 0;
        current_.level = 0;
        inputs_ = {};
        view_ = {};
        driven_ = false;
    }

private:
    struct State {
        std::vector<std::optional<ProgramLoadConfigDescriptor>> storage{};
        std::size_t read_pointer{0};
        std::size_t write_pointer{0};
        std::size_t level{0};
    };

    static State make_state(std::size_t depth)
    {
        State state{};
        state.storage.resize(depth);
        return state;
    }

    std::size_t increment(std::size_t pointer) const noexcept
    {
        return pointer + 1 == depth_ ? 0 : pointer + 1;
    }

    static ProgramLoadConfigFifoHardwareInterfaceView build_view(
        const State& state, const ProgramLoadConfigFifoInputs& inputs,
        std::size_t depth)
    {
        ProgramLoadConfigFifoHardwareInterfaceView view{};
        view.enqueue_valid_i = inputs.enqueue_valid_i;
        view.enqueue_data_i = inputs.enqueue_data_i;
        view.dequeue_valid_o = state.level != 0;
        view.dequeue_ready_i = inputs.dequeue_ready_i;
        view.dequeue_fire = view.dequeue_valid_o && view.dequeue_ready_i;
        view.enqueue_ready_o = state.level < depth || view.dequeue_fire;
        view.enqueue_fire = view.enqueue_valid_i && view.enqueue_ready_o;
        view.level_o = state.level;
        view.empty_o = state.level == 0;
        view.full_o = state.level == depth;
        if (view.dequeue_valid_o)
            view.dequeue_data_o = state.storage.at(state.read_pointer);
        return view;
    }

    std::size_t depth_{0};
    State current_{};
    std::optional<State> next_{};
    ProgramLoadConfigFifoInputs inputs_{};
    ProgramLoadConfigFifoHardwareInterfaceView view_{};
    bool driven_{false};
};

// Host-side adapter: submit() may populate this software plan without
// advancing hardware time, but the real command RAM changes only on observed
// program_cmd_write_fire cycles. Payload remains stable while ready is low.
class ProgramLoadHostAdapter {
public:
    enum class State { Idle, Reset, WriteCommands, Start, Complete };

    void submit(std::vector<ProgramLoadCommand> commands)
    {
        if (state_ == State::WriteCommands || state_ == State::Start)
            throw std::logic_error("Program Load host adapter is busy");
        if (commands.empty()
            || commands.size()
                > ProgramLoadCommandController::kRevisionCommandDepth)
            throw std::invalid_argument(
                "Program Load host command list must contain 1..64 commands");
        commands_ = std::move(commands);
        write_offset_ = 0;
        state_ = State::Reset;
    }

    void drive(ProgramLoadCommandInputs& inputs) const
    {
        inputs.program_cmd_write_valid_i = false;
        inputs.program_load_start_valid_i = false;
        inputs.reset_i = state_ == State::Reset;
        if (state_ == State::WriteCommands) {
            inputs.program_cmd_write_valid_i = true;
            inputs.program_cmd_write_address_i =
                static_cast<std::uint8_t>(write_offset_);
            inputs.program_cmd_write_payload_i = commands_.at(write_offset_);
        } else if (state_ == State::Start) {
            inputs.program_load_start_valid_i = true;
            inputs.program_load_start_command_count_i =
                static_cast<std::uint16_t>(commands_.size());
        }
    }

    void observe(const ProgramLoadCommandHardwareInterfaceView& view)
    {
        if (state_ == State::Reset) {
            state_ = State::WriteCommands;
        } else if (state_ == State::WriteCommands
            && view.program_cmd_write_fire) {
            if (++write_offset_ == commands_.size()) state_ = State::Start;
        } else if (state_ == State::Start
            && view.program_load_start_fire) {
            state_ = State::Complete;
        }
    }

    void reset() noexcept
    {
        commands_.clear();
        write_offset_ = 0;
        state_ = State::Idle;
    }

    State state() const noexcept { return state_; }
    bool complete() const noexcept { return state_ == State::Complete; }
    std::size_t write_offset() const noexcept { return write_offset_; }
    const std::vector<ProgramLoadCommand>& commands() const noexcept
    {
        return commands_;
    }

private:
    std::vector<ProgramLoadCommand> commands_{};
    std::size_t write_offset_{0};
    State state_{State::Idle};
};

inline ProgramLoadCommand make_load_imem_command(
    const ProgramImage& image, std::uint16_t destination,
    std::uint32_t imem_base_address = 0,
    bool image_first = true, bool image_last = true)
{
    if (image.storage != ProgramStorage::Imem
        || image.mode != IcuQueueMode::Macro)
        throw std::invalid_argument(
            "canonical Program Load command requires Macro local i-MEM image");
    if (image.byte_count == 0
        || image.byte_count > std::numeric_limits<std::uint32_t>::max()
        || image.word_count() > std::numeric_limits<std::uint16_t>::max())
        throw std::invalid_argument(
            "ProgramImage cannot be represented by canonical command fields");
    ProgramLoadCommand command{};
    command.opcode = ProgramLoadCommandOpcode::LoadImem;
    command.source_address = image.source_address;
    command.byte_count = static_cast<std::uint32_t>(image.byte_count);
    command.destination = destination;
    command.format = image.word_bytes() == 12
        ? ProgramLoadFormat::Word96 : ProgramLoadFormat::Word128;
    command.encoding_mode = ProgramLoadEncodingMode::Macro;
    command.imem_base_address = imem_base_address;
    command.image_first = image_first;
    command.image_last = image_last;
    command.expected_word_count =
        static_cast<std::uint16_t>(image.word_count());
    return command;
}

inline std::vector<ProgramLoadCommand> make_program_load_command_list(
    const std::vector<std::pair<ProgramImage, std::uint16_t>>& images)
{
    std::vector<ProgramLoadCommand> commands;
    commands.reserve(images.size() * 3 + 1);
    for (const auto& [image, destination] : images) {
        commands.push_back(make_load_imem_command(image, destination));
        auto wait = commands.back();
        wait.opcode = ProgramLoadCommandOpcode::WaitDma;
        wait.source_address = 0;
        wait.byte_count = 0;
        commands.push_back(wait);
        auto commit = wait;
        commit.opcode = ProgramLoadCommandOpcode::CommitQueue;
        commands.push_back(commit);
    }
    ProgramLoadCommand end{};
    end.opcode = ProgramLoadCommandOpcode::End;
    commands.push_back(end);
    if (commands.size()
        > ProgramLoadCommandController::kRevisionCommandDepth)
        throw std::invalid_argument(
            "Program Load command list exceeds revision capacity");
    return commands;
}

} // namespace ftlpu
