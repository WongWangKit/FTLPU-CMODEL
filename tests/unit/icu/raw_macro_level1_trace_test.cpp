#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/core/timing.hpp"
#include "ftlpu/system/icu.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Queue = ftlpu::InstructionControlUnit::MemIcu;
using RawWord = ftlpu::IcuRawImemWord96;

// The RTL Level1 fixture instantiates IMEM_DEPTH=16, so its internal
// lpu_mem_imem_fetch address port is four bits even though the production
// lpu_mem_local_imem_frontend defaults to the full 17-bit address space.
constexpr std::size_t kLevel1ImemAddressWidth = 4;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

std::string json_escape(std::string_view text)
{
    std::string result;
    result.reserve(text.size());
    for (const char value : text) {
        switch (value) {
        case '\\': result += "\\\\"; break;
        case '"': result += "\\\""; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default: result += value; break;
        }
    }
    return result;
}

std::string binary(std::uint64_t value, std::size_t width)
{
    if (width == 0 || width > 64)
        throw std::invalid_argument("integer trace width must be in [1,64]");
    std::string result(width, '0');
    for (std::size_t bit = 0; bit < width; ++bit)
        result[width - 1 - bit] = ((value >> bit) & 1u) ? '1' : '0';
    return result;
}

template <std::size_t N>
std::string binary(const std::array<std::uint32_t, N>& lanes)
{
    std::string result(N * 32, '0');
    for (std::size_t bit = 0; bit < N * 32; ++bit)
        result[N * 32 - 1 - bit]
            = ((lanes[bit / 32] >> (bit % 32)) & 1u) ? '1' : '0';
    return result;
}

class TraceWriter {
public:
    explicit TraceWriter(const std::filesystem::path& path)
    {
        if (path.has_parent_path())
            std::filesystem::create_directories(path.parent_path());
        output_.open(path, std::ios::out | std::ios::trunc);
        if (!output_) throw std::runtime_error("cannot create Level1 trace");
    }

    void emit(std::string_view case_id, std::uint64_t global_cycle,
        std::int64_t logical_cycle, std::uint64_t transport_tick,
        std::string_view phase, std::string_view sample_point,
        std::string_view module, std::string_view instance,
        std::string_view signal, std::size_t width, std::string value,
        std::string_view event_type, std::string_view mapping_status,
        std::string_view source_ref)
    {
        if (value.size() != width
            || value.find_first_not_of("01") != std::string::npos)
            throw std::logic_error("trace value is not exact-width binary");
        if (event_type != "SIGNAL_SAMPLE"
            && event_type != "STATE_SAMPLE"
            && event_type != "ARCH_EVENT")
            throw std::logic_error("non-canonical Level1 trace event type");
        if (event_type == "ARCH_EVENT"
            && signal != "CASE_START"
            && signal != "RAW_WORD_ACCEPTED"
            && signal != "PRIME_FIRE"
            && signal != "START"
            && signal != "MEM_ISSUE"
            && signal != "CASE_END")
            throw std::logic_error("non-canonical Level1 architecture event");
        output_
            << "{\"schema\":\"ftlpu.raw_macro.level1.trace.v1\""
            << ",\"run_id\":\"cmodel_raw_macro_level1_v1\""
            << ",\"case_id\":\"" << json_escape(case_id) << "\""
            << ",\"source\":\"CMODEL\""
            << ",\"global_cycle\":" << global_cycle
            << ",\"logical_cycle\":" << logical_cycle
            << ",\"transport_tick\":" << transport_tick
            << ",\"phase\":\"" << json_escape(phase) << "\""
            << ",\"sample_point\":\"" << json_escape(sample_point) << "\""
            << ",\"event_sequence\":" << event_sequence_++
            << ",\"module\":\"" << json_escape(module) << "\""
            << ",\"instance\":\"" << json_escape(instance) << "\""
            << ",\"signal\":\"" << json_escape(signal) << "\""
            << ",\"width\":" << width
            << ",\"value\":\"" << value << "\""
            << ",\"event_type\":\"" << json_escape(event_type) << "\""
            << ",\"mapping_status\":\"" << json_escape(mapping_status) << "\""
            << ",\"source_ref\":\"" << json_escape(source_ref) << "\"}\n";
        if (!output_) throw std::runtime_error("failed to write Level1 trace");
    }

private:
    std::ofstream output_{};
    std::uint64_t event_sequence_{};
};

const std::vector<RawWord> kTwoIssueImage{
    {{{0x00000025u, 0x00000000u, 0x00000000u}}},
    {{{0x02000021u, 0x00000180u, 0x00303200u}}},
    {{{0x00000400u, 0x00000000u, 0x00000000u}}},
};

std::vector<RawWord> make_repeat4_image()
{
    std::vector<RawWord> result(1);
    // valid=1, mode=Macro(01), command_count=1. Control bit 1 is reserved.
    result[0].lanes[0] = 0x15u;
    std::size_t offset = 96;
    const auto bits = [&](std::uint64_t value, unsigned width) {
        for (unsigned bit = 0; bit < width; ++bit, ++offset) {
            if (result.size() <= offset / 96) result.emplace_back();
            result[offset / 96].lanes[(offset % 96) / 32]
                |= std::uint32_t((value >> bit) & 1u) << (offset % 32);
        }
    };
    bits(0, 3);    // Dictionary count.
    bits(0, 32);   // start_cycle = 0.
    bits(100, 13); // Initial MEM row address.
    bits(0, 1);    // One-template run.
    bits(0, 1);    // Compact template.
    bits(1, 2);    // Inner1D schedule.
    bits(0, 1);    // Read.
    bits(0, 6);    // East stream 0.
    bits(0, 1);    // Preserve is irrelevant for Read.
    bits(3, 11);   // inner_count = 4.
    bits(0, 8);    // inner_interval = 1.
    bits(1, 10);   // address stride = 1.
    return result;
}

struct ExpectedIssue {
    std::size_t cycle{};
    std::size_t address{};

    friend bool operator==(const ExpectedIssue&, const ExpectedIssue&) = default;
};

struct CaseSpec {
    std::string id{};
    std::vector<RawWord> image{};
    std::vector<ExpectedIssue> expected{};
};

enum class Stage : std::uint8_t {
    InitialIdle0,
    InitialIdle1,
    Load,
    LoadGap,
    PrimeRequest,
    Prime,
    StartWait,
    StartRun,
    Run,
    Done,
};

std::string_view stage_name(Stage stage)
{
    switch (stage) {
    case Stage::InitialIdle0: return "INITIAL_IDLE_0";
    case Stage::InitialIdle1: return "INITIAL_IDLE_1";
    case Stage::Load: return "LOAD";
    case Stage::LoadGap: return "LOAD_GAP";
    case Stage::PrimeRequest: return "PRIME_REQUEST";
    case Stage::Prime: return "PRIME";
    case Stage::StartWait: return "START_WAIT";
    case Stage::StartRun: return "START_RUN";
    case Stage::Run: return "RUN";
    case Stage::Done: return "DONE";
    }
    return "INVALID";
}

struct RunState {
    Queue queue{};
    Stage stage{Stage::InitialIdle0};
    std::size_t load_index{};
    std::uint64_t transport_tick{};
    std::vector<ExpectedIssue> issues{};
};

struct CycleObservation {
    Stage executed_stage{Stage::InitialIdle0};
    std::uint64_t transport_tick{};
    std::int64_t logical_cycle{-1};
    bool load_accept{};
    bool prime_fire{};
    bool prime_ready{};
    bool start{};
    bool done{};
    std::optional<ftlpu::MemInstruction> issue{};
};

class RawMacroLevel1Module final : public ftlpu::timing::ClockedModule {
public:
    using PosedgeObserver = std::function<void()>;

    explicit RawMacroLevel1Module(const CaseSpec& spec)
        : spec_(spec)
    {}

    void set_reset(bool reset) noexcept { reset_ = reset; }
    void set_posedge_observer(PosedgeObserver observer)
    {
        posedge_observer_ = std::move(observer);
    }

    void low() override
    {
        state_.prepare();
        auto& next = state_.next_value();
        observation_ = {};
        if (reset_) {
            next = {};
            return;
        }

        observation_.executed_stage = state_.current().stage;
        observation_.transport_tick = state_.current().transport_tick;
        next.transport_tick = state_.current().transport_tick + 1;
        const auto execute_run_cycle = [&] {
            observation_.logical_cycle
                = static_cast<std::int64_t>(next.queue.cycle());
            observation_.issue = next.queue.dispatch_mem_issue(true);
            if (observation_.issue) {
                next.issues.push_back(ExpectedIssue{
                    static_cast<std::size_t>(observation_.logical_cycle),
                    observation_.issue->address});
            }
            if (next.queue.done()) {
                next.stage = Stage::Done;
            }
        };

        switch (state_.current().stage) {
        case Stage::InitialIdle0:
            next.stage = Stage::InitialIdle1;
            break;
        case Stage::InitialIdle1:
            next.stage = Stage::Load;
            break;
        case Stage::Load: {
            const auto address = state_.current().load_index;
            if (address >= spec_.image.size())
                throw std::logic_error("Level1 load index exceeded image");
            std::array<std::uint32_t, 4> carrier{};
            std::copy(spec_.image[address].lanes.begin(),
                spec_.image[address].lanes.end(), carrier.begin());
            next.queue.load_mem_program_word(address, carrier, 0x7,
                address + 1 == spec_.image.size());
            observation_.load_accept = true;
            next.load_index = address + 1;
            next.stage = Stage::LoadGap;
            break;
        }
        case Stage::LoadGap:
            next.stage = state_.current().load_index == spec_.image.size()
                ? Stage::PrimeRequest : Stage::Load;
            break;
        case Stage::PrimeRequest:
            next.queue.prime_mem_program(
                ftlpu::IcuMacroQueueKind::Mem, spec_.image.size());
            observation_.prime_fire = true;
            next.stage = Stage::Prime;
            break;
        case Stage::Prime:
            if (!next.queue.raw_macro_ready())
                next.queue.prefetch_raw_macro();
            if (next.queue.raw_macro_ready()) {
                observation_.prime_ready = true;
                next.stage = Stage::StartWait;
            }
            break;
        case Stage::StartWait:
            next.stage = Stage::StartRun;
            break;
        case Stage::StartRun:
            observation_.start = true;
            next.stage = Stage::Run;
            execute_run_cycle();
            break;
        case Stage::Run:
            execute_run_cycle();
            break;
        case Stage::Done:
            observation_.done = true;
            break;
        }
    }

    void validate_posedge() const override { state_.validate_commit(); }

    void posedge() override
    {
        state_.commit();
        if (posedge_observer_) posedge_observer_();
    }

    void high() override
    {
        // HIGH is observation-only. No Next preparation or queue progression.
    }

    const RunState& current() const noexcept { return state_.current(); }
    const RunState& next() const noexcept { return state_.next(); }
    const CycleObservation& observation() const noexcept { return observation_; }

private:
    const CaseSpec& spec_;
    ftlpu::timing::ClockedState<RunState> state_{};
    CycleObservation observation_{};
    PosedgeObserver posedge_observer_{};
    bool reset_{true};
};

void emit_mapping_declarations(TraceWriter& writer, const CaseSpec& spec)
{
    const auto emit = [&](std::string_view signal, std::string_view status,
                          std::string_view ref) {
        writer.emit(spec.id, 0, -1, 0, "LOW", "PRE_POSEDGE",
            "lpu_mem_local_imem_frontend", "mem_queue_0", signal, 1, "0",
            "STATE_SAMPLE", status, ref);
    };
    emit("ddb_internal_slot", "UNMAPPED",
        "macro_bitstream_decoder.hpp:DDB internals are intentionally private");
    emit("active_context_slot_payload", "UNMAPPED",
        "distributed_queue.hpp:full Active Context slot payload is private");
    emit("fu_sram_effect", "UNMAPPED",
        "Level1 ends at the Scheduler to MEM issue boundary");
    emit("mem_instruction_queue", "NA",
        "MEM direct scheduler path has no Instruction Queue");
    emit("c_plus_2_lookahead", "NA",
        "MEM timed depth/lookahead are zero");
    emit("scheduler_output_register", "NA",
        "MEM Scheduler has no independent output register");
}

void emit_architecture_event(TraceWriter& writer, const CaseSpec& spec,
    const ftlpu::timing::Clock& clock, const RawMacroLevel1Module& module,
    std::string_view signal, std::size_t width, std::string value,
    std::string_view source_ref,
    std::string_view module_name = "raw_macro_level1",
    std::string_view instance_name = "raw_macro_level1",
    std::string_view mapping_status = "SEMANTIC")
{
    const auto& observation = module.observation();
    writer.emit(spec.id, clock.cycle(), observation.logical_cycle,
        observation.transport_tick, "LOW", "PRE_POSEDGE",
        module_name, instance_name, signal, width,
        std::move(value), "ARCH_EVENT", mapping_status, source_ref);
}

void emit_low_snapshot(TraceWriter& writer, const CaseSpec& spec,
    const ftlpu::timing::Clock& clock, const RawMacroLevel1Module& module)
{
    const auto& observation = module.observation();
    const auto& queue = module.next().queue;
    const auto emit = [&](std::string_view module_name,
                          std::string_view signal, std::size_t width,
                          std::string value, std::string_view event_type,
                          std::string_view mapping, std::string_view ref) {
        writer.emit(spec.id, clock.cycle(), observation.logical_cycle,
            observation.transport_tick, "LOW", "PRE_POSEDGE", module_name,
            "mem_queue_0", signal, width, std::move(value), event_type,
            mapping, ref);
    };

    emit("raw_macro_level1", "stage", 4,
        binary(static_cast<std::uint8_t>(module.next().stage), 4),
        "STATE_SAMPLE", "SEMANTIC", stage_name(module.next().stage));
    emit("cmodel_fetch_diagnostic", "fetch_descriptor_started", 1,
        binary(queue.macro_fetch_descriptor_started(), 1), "STATE_SAMPLE",
        "UNMAPPED", "IcuMacroV1Decoder internal diagnostic");
    emit("cmodel_fetch_diagnostic", "fetch_descriptor_active", 1,
        binary(queue.macro_fetch_active(), 1), "STATE_SAMPLE", "UNMAPPED",
        "IcuMacroV1Decoder internal diagnostic");
    emit("cmodel_fetch_diagnostic", "fetch_complete", 1,
        binary(queue.macro_fetch_complete(), 1), "STATE_SAMPLE", "UNMAPPED",
        "IcuMacroV1Decoder internal diagnostic");
    emit("cmodel_fetch_diagnostic", "next_fetch_address", 17,
        binary(queue.macro_next_fetch_address(), 17), "STATE_SAMPLE",
        "UNMAPPED", "IcuMacroV1Decoder internal diagnostic");
    emit("cmodel_fetch_diagnostic", "pending_fetch_level", 3,
        binary(queue.macro_pending_fetch_level(), 3), "STATE_SAMPLE",
        "UNMAPPED", "IcuMacroV1Decoder internal diagnostic");
    emit("cmodel_fetch_diagnostic", "pending_front_address", 17,
        binary(queue.macro_pending_fetch_front_address().value_or(0), 17),
        "STATE_SAMPLE", "UNMAPPED",
        "IcuMacroV1Decoder internal diagnostic; valid when level is nonzero");
    emit("cmodel_fetch_diagnostic", "pending_front_remaining", 8,
        binary(queue.macro_pending_fetch_front_remaining_cycles().value_or(0),
            8), "STATE_SAMPLE", "UNMAPPED",
        "IcuMacroV1Decoder internal diagnostic; valid when level is nonzero");
    emit("cmodel_fetch_diagnostic", "response_fifo_level", 3,
        binary(queue.macro_response_fifo_level(), 3), "STATE_SAMPLE",
        "UNMAPPED", "IcuMacroV1Decoder internal diagnostic");
    emit("cmodel_fetch_diagnostic", "response_fifo_front_address", 17,
        binary(queue.macro_response_fifo_front_address().value_or(0), 17),
        "STATE_SAMPLE", "UNMAPPED",
        "IcuMacroV1Decoder internal diagnostic; valid when level is nonzero");
    emit("cmodel_fetch_diagnostic", "word_fifo_front_address", 17,
        binary(queue.macro_word_fifo_front_address().value_or(0), 17),
        "STATE_SAMPLE", "UNMAPPED",
        "IcuMacroV1Decoder internal diagnostic; valid when level is nonzero");
    emit("cmodel_fetch_diagnostic", "outstanding_payload_fetches", 32,
        binary(queue.macro_outstanding_payload_fetches(), 32),
        "STATE_SAMPLE", "UNMAPPED",
        "IcuMacroV1Decoder internal diagnostic");
    emit("cmodel_fetch_diagnostic", "control_latched", 1,
        binary(queue.macro_control_latched(), 1), "STATE_SAMPLE", "UNMAPPED",
        "IcuMacroV1Decoder internal diagnostic");
    emit("lpu_mem_macro_decoder", "state_q", 4,
        binary(queue.macro_decoder_cycle_state_code(), 4), "STATE_SAMPLE", "DIRECT",
        "macro_bitstream_decoder.hpp:MemState");
    emit("lpu_mem_macro_decoder", "parser_phase", 2,
        binary(queue.macro_decoder_cycle_phase_code(), 2), "STATE_SAMPLE", "PARTIAL",
        "macro_bitstream_decoder.hpp:MEM subphase registers");
    emit("lpu_mem_macro_decoder", "reservoir_bits_q", 9,
        binary(queue.macro_decoder_cycle_reservoir_bits(), 9), "STATE_SAMPLE", "DIRECT",
        "macro_bitstream_decoder.hpp:mem_reservoir_bits_");
    emit("lpu_mem_macro_decoder", "fault_o", 1,
        binary(queue.macro_decoder_fault(), 1), "STATE_SAMPLE", "DIRECT",
        "macro_bitstream_decoder.hpp:mem_fault_");
    emit("lpu_mem_macro_decoder", "fault_code_o", 4,
        binary(queue.macro_decoder_fault_code(), 4), "STATE_SAMPLE", "DIRECT",
        "macro_bitstream_decoder.hpp:mem_fault_code_");

    if (observation.executed_stage == Stage::Load) {
        const auto& pins = queue.mem_local_imem_load_port_signals();
        emit("lpu_mem_local_imem_frontend", "load_valid_i", 1,
            binary(pins.load_valid_i, 1), "SIGNAL_SAMPLE", "DIRECT",
            "distributed_queue.hpp:load_mem_program_word");
        emit("lpu_mem_local_imem_frontend", "load_ready_o", 1,
            binary(pins.load_ready_o, 1), "SIGNAL_SAMPLE", "DIRECT",
            "distributed_queue.hpp:load_mem_program_word");
        emit("lpu_mem_local_imem_frontend", "load_imem_address_i", 17,
            binary(pins.load_imem_address_i, 17), "SIGNAL_SAMPLE", "DIRECT",
            "distributed_queue.hpp:MemLocalImemLoadPortSignals");
        emit("lpu_mem_local_imem_frontend", "load_data_i", 128,
            binary(pins.load_data_i), "SIGNAL_SAMPLE", "DIRECT",
            "distributed_queue.hpp:MemLocalImemLoadPortSignals");
        emit("lpu_mem_local_imem_frontend", "load_strb_i", 4,
            binary(pins.load_strb_i, 4), "SIGNAL_SAMPLE", "DIRECT",
            "distributed_queue.hpp:MemLocalImemLoadPortSignals");
        emit("lpu_mem_local_imem_frontend", "load_last_i", 1,
            binary(pins.load_last_i, 1), "SIGNAL_SAMPLE", "DIRECT",
            "distributed_queue.hpp:MemLocalImemLoadPortSignals");
        emit("lpu_mem_local_imem_frontend", "load_fire", 1,
            binary(pins.load_fire, 1), "SIGNAL_SAMPLE", "DIRECT",
            "distributed_queue.hpp:MemLocalImemLoadPortSignals");
        const auto word_index = module.next().load_index - 1;
        emit("raw_macro_level1", "stimulus_word_data", 96,
            binary(spec.image[word_index].lanes), "SIGNAL_SAMPLE", "SEMANTIC",
            "test stimulus: lane0 carries bits[31:0], payload is LSB-first");
        const std::array<std::uint32_t, 3> accepted_word{
            pins.load_data_i[0], pins.load_data_i[1], pins.load_data_i[2]};
        emit_architecture_event(writer, spec, clock, module,
            "RAW_WORD_ACCEPTED", 96, binary(accepted_word),
            "distributed_queue.hpp:load_mem_program_word accepted low 96 bits",
            "lpu_mem_local_imem_frontend", "mem_queue_0", "DIRECT");
    }

    if (observation.executed_stage == Stage::PrimeRequest) {
        const auto& pins = queue.mem_local_imem_load_port_signals();
        emit("lpu_mem_local_imem_frontend", "prime_valid_i", 1,
            binary(pins.prime_valid_i, 1), "SIGNAL_SAMPLE", "DIRECT",
            "distributed_queue.hpp:prime_mem_program");
        emit("lpu_mem_local_imem_frontend", "prime_ready_o", 1,
            binary(pins.prime_ready_o, 1), "SIGNAL_SAMPLE", "DIRECT",
            "distributed_queue.hpp:prime_mem_program");
        emit("lpu_mem_local_imem_frontend", "prime_fire", 1,
            binary(pins.prime_fire, 1), "SIGNAL_SAMPLE", "DIRECT",
            "distributed_queue.hpp:prime_mem_program");
        emit_architecture_event(writer, spec, clock, module, "PRIME_FIRE", 1,
            "1", "prime_valid_i && prime_ready_o",
            "lpu_mem_local_imem_frontend", "mem_queue_0", "SEMANTIC");
    }

    if (observation.executed_stage == Stage::Prime
        || observation.executed_stage == Stage::Run) {
        const auto& imem = queue.imem_read_port_signals();
        emit("lpu_mem_imem_fetch", "imem_read_req_valid_o", 1,
            binary(imem.imem_read_req_valid_o, 1), "SIGNAL_SAMPLE", "DIRECT",
            "macro_bitstream_decoder.hpp:ImemReadPortSignals");
        emit("lpu_mem_imem_fetch", "imem_read_req_ready_i", 1,
            binary(imem.imem_read_req_ready_i, 1), "SIGNAL_SAMPLE", "DIRECT",
            "macro_bitstream_decoder.hpp:ImemReadPortSignals");
        emit("lpu_mem_imem_fetch", "imem_read_req_address_o",
            kLevel1ImemAddressWidth,
            binary(imem.imem_read_req_address_o, kLevel1ImemAddressWidth),
            "SIGNAL_SAMPLE", "DIRECT",
            "Level1 IMEM_DEPTH=16 instance; production mapping remains 17-bit");
        emit("lpu_mem_imem_fetch", "imem_read_req_last_o", 1,
            binary(imem.imem_read_req_last_o, 1), "SIGNAL_SAMPLE", "DIRECT",
            "macro_bitstream_decoder.hpp:ImemReadPortSignals");
        emit("lpu_mem_imem_fetch", "imem_read_req_fire", 1,
            binary(imem.imem_read_req_fire, 1), "SIGNAL_SAMPLE", "DIRECT",
            "macro_bitstream_decoder.hpp:ImemReadPortSignals");
        emit("lpu_mem_local_imem", "imem_read_rsp_valid_i", 1,
            binary(imem.imem_read_rsp_valid_i, 1), "SIGNAL_SAMPLE", "DIRECT",
            "macro_bitstream_decoder.hpp:ImemReadPortSignals");
        emit("lpu_mem_local_imem", "imem_read_rsp_ready_o", 1,
            binary(imem.imem_read_rsp_ready_o, 1), "SIGNAL_SAMPLE", "DIRECT",
            "macro_bitstream_decoder.hpp:ImemReadPortSignals");
        std::array<std::uint32_t, 3> response_data{};
        if (imem.imem_read_rsp_data_i)
            response_data = imem.imem_read_rsp_data_i->lanes;
        emit("lpu_mem_local_imem", "imem_read_rsp_data_i", 96,
            binary(response_data), "SIGNAL_SAMPLE", "DIRECT",
            "macro_bitstream_decoder.hpp:ImemReadPortSignals");
        emit("lpu_mem_local_imem", "imem_read_rsp_last_i", 1,
            binary(imem.imem_read_rsp_last_i, 1), "SIGNAL_SAMPLE", "DIRECT",
            "macro_bitstream_decoder.hpp:ImemReadPortSignals");
        emit("lpu_mem_local_imem", "imem_read_rsp_fire", 1,
            binary(imem.imem_read_rsp_fire, 1), "SIGNAL_SAMPLE", "DIRECT",
            "macro_bitstream_decoder.hpp:ImemReadPortSignals");

        const auto& fifo = queue.context_fifo_port_signals();
        emit("lpu_mem_context_fifo", "context_fifo_enqueue_valid", 1,
            binary(fifo.enqueue_valid_i, 1), "SIGNAL_SAMPLE", "DIRECT",
            "distributed_queue.hpp:MacroContextFifoPortSignals");
        emit("lpu_mem_context_fifo", "context_fifo_enqueue_ready", 1,
            binary(fifo.enqueue_ready_o, 1), "SIGNAL_SAMPLE", "DIRECT",
            "distributed_queue.hpp:MacroContextFifoPortSignals");
        emit("lpu_mem_context_fifo", "context_fifo_enqueue_fire", 1,
            binary(fifo.enqueue_fire, 1), "SIGNAL_SAMPLE", "DIRECT",
            "distributed_queue.hpp:MacroContextFifoPortSignals");
        emit("lpu_mem_context_fifo", "context_fifo_dequeue_valid", 1,
            binary(fifo.dequeue_valid_o, 1), "SIGNAL_SAMPLE", "DIRECT",
            "distributed_queue.hpp:MacroContextFifoPortSignals");
        emit("lpu_mem_context_fifo", "context_fifo_dequeue_ready", 1,
            binary(fifo.dequeue_ready_i, 1), "SIGNAL_SAMPLE", "DIRECT",
            "distributed_queue.hpp:MacroContextFifoPortSignals");
        emit("lpu_mem_context_fifo", "context_fifo_dequeue_fire", 1,
            binary(fifo.dequeue_fire, 1), "SIGNAL_SAMPLE", "DIRECT",
            "distributed_queue.hpp:MacroContextFifoPortSignals");
        if (const auto& decoded = queue.decoded_context_event()) {
            const auto& schedule = decoded->schedule;
            emit("lpu_mem_macro_decoder", "context_record_index_o", 32,
                binary(decoded->record_index, 32), "SIGNAL_SAMPLE", "DERIVED",
                "macro_bitstream_decoder.hpp:DecodedIcuMacroContext");
            emit("lpu_mem_macro_decoder", "context_start_cycle_o", 32,
                binary(schedule.start_cycle, 32), "SIGNAL_SAMPLE", "SEMANTIC",
                "macro_bitstream_decoder.hpp:DecodedIcuMacroContext");
            emit("lpu_mem_macro_decoder", "context_inner_count_o", 32,
                binary(schedule.inner_count, 32), "SIGNAL_SAMPLE", "SEMANTIC",
                "macro_bitstream_decoder.hpp:IcuMacroSchedule");
            emit("lpu_mem_macro_decoder", "context_inner_interval_o", 32,
                binary(schedule.inner_interval, 32), "SIGNAL_SAMPLE", "SEMANTIC",
                "macro_bitstream_decoder.hpp:IcuMacroSchedule");
            emit("lpu_mem_macro_decoder", "context_inner_stride_o", 32,
                binary(static_cast<std::uint32_t>(schedule.inner_stride), 32),
                "SIGNAL_SAMPLE", "SEMANTIC",
                "macro_bitstream_decoder.hpp:IcuMacroSchedule");
            emit("lpu_mem_macro_decoder", "context_outer_count_o", 32,
                binary(schedule.outer_count, 32), "SIGNAL_SAMPLE", "SEMANTIC",
                "macro_bitstream_decoder.hpp:IcuMacroSchedule");
            emit("lpu_mem_macro_decoder", "context_outer_interval_o", 32,
                binary(schedule.outer_interval, 32), "SIGNAL_SAMPLE", "SEMANTIC",
                "macro_bitstream_decoder.hpp:IcuMacroSchedule");
            emit("lpu_mem_macro_decoder", "context_outer_stride_o", 32,
                binary(static_cast<std::uint32_t>(schedule.outer_stride), 32),
                "SIGNAL_SAMPLE", "SEMANTIC",
                "macro_bitstream_decoder.hpp:IcuMacroSchedule");
            emit("lpu_mem_macro_decoder", "context_induction_target_o", 2,
                binary(static_cast<std::uint8_t>(schedule.induction_target), 2),
                "SIGNAL_SAMPLE", "SEMANTIC",
                "macro_bitstream_decoder.hpp:IcuMacroSchedule");
            emit("lpu_mem_macro_decoder", "context_instruction_o", 32,
                binary(ftlpu::isa::encode_mem_instruction(
                    decoded->instruction), 32),
                "SIGNAL_SAMPLE", "DERIVED",
                "instruction_codec.hpp:encode_mem_instruction");
            emit("lpu_mem_macro_decoder", "context_address_o", 13,
                binary(decoded->instruction.address, 13),
                "SIGNAL_SAMPLE", "SEMANTIC",
                "MemInstruction::address semantic field");
        }
    }

    if (observation.executed_stage == Stage::StartRun
        || observation.executed_stage == Stage::Run) {
        // RTL Monitor emits START before a same-cycle Cycle-0 MEM_ISSUE.
        if (observation.start)
            emit_architecture_event(writer, spec, clock, module, "START", 1,
                "1", "first RUN cycle follows this committed transition",
                "lpu_mem_macro_scheduler", "mem_queue_0", "SEMANTIC");
        const auto& issue = queue.mem_issue_port_signals();
        emit("lpu_mem_macro_scheduler", "mem_issue_valid_o", 1,
            binary(issue.mem_issue_valid_o, 1), "SIGNAL_SAMPLE", "DIRECT",
            "distributed_queue.hpp:MemIssuePortSignals");
        emit("lpu_mem_macro_scheduler", "mem_issue_ready_i", 1,
            binary(issue.mem_issue_ready_i, 1), "SIGNAL_SAMPLE", "DIRECT",
            "distributed_queue.hpp:MemIssuePortSignals");
        const auto encoded = issue.mem_issue_instruction_o
            ? ftlpu::isa::encode_mem_instruction(
                *issue.mem_issue_instruction_o) : 0;
        emit("lpu_mem_macro_scheduler", "mem_issue_instruction_o", 64,
            binary(encoded, 64), "SIGNAL_SAMPLE", "DERIVED",
            "instruction_codec.hpp:encode_mem_instruction");
        emit("lpu_mem_macro_scheduler", "mem_issue_cycle_o", 32,
            binary(issue.mem_issue_cycle_o, 32), "SIGNAL_SAMPLE", "DIRECT",
            "distributed_queue.hpp:MemIssuePortSignals");
        emit("lpu_mem_macro_scheduler", "mem_issue_fire", 1,
            binary(issue.mem_issue_fire, 1), "SIGNAL_SAMPLE", "DIRECT",
            "distributed_queue.hpp:MemIssuePortSignals");
        emit("lpu_mem_macro_scheduler", "conflict_o", 1,
            binary(issue.conflict_o, 1), "SIGNAL_SAMPLE", "DIRECT",
            "distributed_queue.hpp:MemIssuePortSignals");
        emit("lpu_mem_macro_scheduler", "fault_o", 1,
            binary(issue.fault_o, 1), "SIGNAL_SAMPLE", "DIRECT",
            "distributed_queue.hpp:MemIssuePortSignals");
        emit("lpu_mem_macro_scheduler", "fault_code_o", 4,
            binary(issue.fault_code_o, 4), "SIGNAL_SAMPLE", "DIRECT",
            "distributed_queue.hpp:MemIssuePortSignals");
        if (observation.issue) {
            emit("lpu_mem_macro_scheduler", "issued_opcode", 3,
                binary(static_cast<std::uint8_t>(observation.issue->opcode), 3),
                "SIGNAL_SAMPLE", "SEMANTIC", "MemInstruction::opcode");
            emit("lpu_mem_macro_scheduler", "issued_address", 13,
                binary(observation.issue->address, 13),
                "SIGNAL_SAMPLE", "SEMANTIC", "MemInstruction::address");
            emit_architecture_event(writer, spec, clock, module,
                "MEM_ISSUE", 64,
                binary(ftlpu::isa::encode_mem_instruction(
                    *observation.issue), 64),
                "Scheduler mem_issue_fire",
                "lpu_mem_macro_scheduler", "mem_queue_0", "SEMANTIC");
        }
    }

    if (observation.prime_ready)
        emit("lpu_mem_local_imem_frontend", "frontend_ready_o", 1, "1",
            "STATE_SAMPLE", "SEMANTIC", "raw_macro_ready()");
    if (observation.done)
        emit_architecture_event(writer, spec, clock, module, "CASE_END", 1,
            "1", "queue.done()");
}

void emit_post_comb_snapshot(TraceWriter& writer, const CaseSpec& spec,
    const ftlpu::timing::Clock& clock, const RawMacroLevel1Module& module)
{
    const auto& observation = module.observation();
    const auto& queue = module.current().queue;
    const auto emit = [&](std::string_view module_name,
                          std::string_view signal, std::size_t width,
                          std::string value, std::string_view mapping,
                          std::string_view ref) {
        writer.emit(spec.id, clock.cycle(), observation.logical_cycle,
            observation.transport_tick, "POSEDGE", "POST_COMB", module_name,
            "mem_queue_0", signal, width, std::move(value), "STATE_SAMPLE",
            mapping, ref);
    };
    emit("lpu_mem_macro_frontend", "word_fifo_level_o", 3,
        binary(queue.word_fifo_level(), 3), "PARTIAL",
        "macro_bitstream_decoder.hpp:word_fifo_level() settled post-edge view");
    emit("lpu_mem_macro_frontend", "context_fifo_level_o", 3,
        binary(queue.context_fifo_occupancy(), 3), "DIRECT",
        "distributed_queue.hpp:context_fifo_occupancy() settled post-edge view");
    emit("lpu_mem_macro_scheduler", "active_count_o", 3,
        binary(queue.active_macro_count(), 3), "PARTIAL",
        "distributed_queue.hpp:active_macro_count() settled post-edge view");
}

void emit_phase_marker(TraceWriter& writer, const CaseSpec& spec,
    const ftlpu::timing::Clock& clock, const RawMacroLevel1Module& module,
    std::string_view phase, std::string_view sample_point)
{
    const auto& observation = module.observation();
    writer.emit(spec.id, clock.cycle(), observation.logical_cycle,
        observation.transport_tick, phase, sample_point, "timing::Clock",
        "global_clock", "stage", 4,
        binary(static_cast<std::uint8_t>(module.current().stage), 4),
        "STATE_SAMPLE", "SEMANTIC", stage_name(module.current().stage));
}

void run_case(TraceWriter& writer, const CaseSpec& spec)
{
    RawMacroLevel1Module module(spec);
    ftlpu::timing::Clock clock;
    ftlpu::timing::SystemPhase system_phase;
    clock.register_module(module);
    system_phase.add_node("raw-macro-level1",
        [&] { module.low(); }, [&] { module.high(); });
    // Five complete synchronous reset cycles. Trace time restarts only after
    // the fifth reset edge, while Clock retains its registered participant.
    for (std::size_t cycle = 0; cycle < 5; ++cycle) {
        system_phase.low();
        clock.complete_low();
        clock.validate_all();
        clock.commit_all();
        system_phase.high();
        clock.complete_high();
        clock.negedge_all();
    }
    module.set_reset(false);
    clock.reset_time();
    system_phase.reset();
    module.set_posedge_observer([&] {
        emit_phase_marker(writer, spec, clock, module,
            "POSEDGE", "POST_POSEDGE");
        emit_post_comb_snapshot(writer, spec, clock, module);
    });

    emit_mapping_declarations(writer, spec);
    writer.emit(spec.id, clock.cycle(), -1, 0, "LOW", "PRE_POSEDGE",
        "raw_macro_level1", "raw_macro_level1", "CASE_START", 1,
        "1", "ARCH_EVENT", "SEMANTIC", "case begins after reset release");
    writer.emit(spec.id, clock.cycle(), -1, 0, "LOW", "PRE_POSEDGE",
        "raw_macro_level1", "raw_macro_level1", "reset_cycles", 32,
        binary(5, 32), "STATE_SAMPLE", "SEMANTIC",
        "five complete Clock/SystemPhase reset cycles");

    for (std::size_t guard = 0; guard < 256; ++guard) {
        system_phase.low();
        emit_low_snapshot(writer, spec, clock, module);
        clock.complete_low();
        clock.validate_all();
        clock.commit_all();
        system_phase.high();
        emit_phase_marker(writer, spec, clock, module,
            "HIGH", "POST_COMB");
        clock.complete_high();
        clock.negedge_all();
        if (module.observation().done) break;
        if (guard == 255)
            throw std::runtime_error("Level1 raw Macro case did not finish");
    }

    require(module.current().queue.done(),
        "Level1 raw Macro queue did not drain");
    require(!module.current().queue.mem_issue_port_signals().fault_o,
        "Level1 raw Macro Scheduler reported a fault");
    require(module.current().queue.active_macro_count() == 0,
        "Level1 raw Macro Active Context did not drain");
    require(module.current().queue.context_fifo_port_signals().level_o == 0,
        "Level1 raw Macro Context FIFO did not drain");
    require(module.current().issues == spec.expected,
        "Level1 raw Macro issue cycle/address sequence mismatched");
}

} // namespace

int main(int argc, char** argv)
try {
    const std::filesystem::path output = argc > 1
        ? std::filesystem::path(argv[1])
        : std::filesystem::path("out/raw_macro_level1/cmodel_trace.jsonl");
    TraceWriter writer(output);
    const std::vector<CaseSpec> cases{
        {"raw_two_issue", kTwoIssueImage, {{3, 100}, {7, 101}}},
        {"raw_repeat4", make_repeat4_image(),
            {{0, 100}, {1, 101}, {2, 102}, {3, 103}}},
    };
    for (const auto& spec : cases) run_case(writer, spec);
    std::cout << "raw Macro Level1 Clock/SystemPhase trace passed: "
              << output.string() << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "raw Macro Level1 trace test failed: "
              << error.what() << '\n';
    return 1;
}
