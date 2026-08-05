#pragma once

#include "ftlpu/vxm/compiler/codegen.hpp"
#include "ftlpu/vxm/compiler/external_data.hpp"
#include "ftlpu/vxm/slice.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ftlpu::vxm::compiler {

struct VxmCModelConfigEvent {
    std::size_t cycle{0};
    std::size_t phase_id{0};
    std::size_t stage{0};
    VxmCompactInstruction packet{};
};

struct VxmCModelDepthEvent {
    std::size_t cycle{0};
    std::size_t phase_id{0};
    std::size_t superlane{0};
    VxmChainDepth depth{VxmChainDepth::Eight};
    bool feedback_transition{false};
};

struct VxmCModelScalarEvent {
    std::size_t cycle{0};
    std::size_t phase_id{0};
    std::size_t superlane{0};
    ValueId value{kInvalidValue};
    std::size_t source_stream{0};
    std::size_t destination_stage{0};
    std::size_t element_index{0};
};

struct VxmCModelInputEvent {
    std::size_t cycle{0};
    std::size_t phase_id{0};
    std::size_t superlane{0};
    ValueId value{kInvalidValue};
    std::size_t stream_base{0};
    std::size_t element_index{0};
    bool held{false};
};

struct VxmCModelCycleRecord {
    std::size_t cycle{0};
    std::vector<VxmExternalStreamRequest> requests{};
    std::vector<VxmCModelConfigEvent> configs{};
    std::vector<VxmCModelDepthEvent> depth_changes{};
    std::vector<VxmCModelScalarEvent> scalar_loads{};
    std::vector<VxmCModelInputEvent> inputs{};
    std::vector<VxmExternalOutputEvent> outputs{};
    std::array<VxmChainDepth, VxmSlice::kTileCount> depths{};
    std::array<
        std::array<VxmLaneAluTraceState, VxmLane::kAluCount>,
        VxmSlice::kTileCount> alu_states{};
};

struct VxmCModelRunResult {
    std::size_t cycles{0};
    std::size_t sram_read_latency{0};
    std::size_t sram_write_latency{0};
    std::vector<std::size_t> phase_shifts{};
    std::vector<std::size_t> phase_sram_waits{};
    std::vector<std::size_t> phase_scalar_load_waits{};
    std::vector<VxmExternalStreamRequest> requests{};
    std::vector<VxmExternalOutputEvent> outputs{};
    std::vector<VxmCModelCycleRecord> timeline{};
};

// Deterministic synchronous-SRAM timing used by the compiler/C Model
// boundary. Bandwidth and bank conflicts remain the responsibility of the
// future global MEM scheduler.
struct VxmSramTiming {
    std::size_t read_latency{1};
    std::size_t write_latency{1};
};

// Drives a complete VxmSlice from compiled, compact instructions. The data
// model is deliberately outside VXM: it stands in for statically scheduled
// MEM/MXM producers without modeling their internal implementation.
class VxmSliceCModelAdapter {
public:
    explicit VxmSliceCModelAdapter(VxmSramTiming timing = {})
        : timing_(timing)
    {}

    VxmCModelRunResult run(
        VxmSlice& slice, const VxmCompiledProgram& program,
        VxmExternalDataModel& external, std::ostream* trace = nullptr) const
    {
        program.schedule.validate();
        auto events = build_events(program);
        auto result = VxmCModelRunResult{};
        result.sram_read_latency = timing_.read_latency;
        result.sram_write_latency = timing_.write_latency;
        result.phase_shifts = events.phase_shifts;
        result.phase_sram_waits = events.phase_sram_waits;
        result.phase_scalar_load_waits =
            events.phase_scalar_load_waits;
        auto pending =
            std::map<std::size_t, std::vector<PendingResponse>>{};
        auto pending_scalars =
            std::map<std::size_t,
                     std::vector<PendingScalarResponse>>{};
        auto held = std::vector<HeldResponse>{};

        for (std::size_t cycle = 0; cycle <= events.last_cycle; ++cycle) {
            auto cycle_record = VxmCModelCycleRecord{};
            cycle_record.cycle = cycle;
            if (const auto found = events.requests.find(cycle);
                found != events.requests.end()) {
                for (const auto& request : found->second) {
                    auto response = PendingResponse{
                        request, external.read(request)};
                    pending[request.required_cycle].push_back(
                        std::move(response));
                    result.requests.push_back(request);
                    cycle_record.requests.push_back(request);
                }
            }
            if (const auto found = events.scalar_requests.find(cycle);
                found != events.scalar_requests.end()) {
                for (const auto& event : found->second) {
                    auto response = PendingScalarResponse{
                        event, external.read(event.request)};
                    pending_scalars[event.request.required_cycle]
                        .push_back(std::move(response));
                    result.requests.push_back(event.request);
                    cycle_record.requests.push_back(event.request);
                }
            }

            if (const auto found = events.configs.find(cycle);
                found != events.configs.end()) {
                for (const auto* command : found->second) {
                    slice.issue_south(
                        command->stage, command->packet);
                    cycle_record.configs.push_back({
                        cycle,
                        command->phase_id,
                        command->stage,
                        command->packet});
                }
            }

            if (const auto found = events.depths.find(cycle);
                found != events.depths.end()) {
                for (const auto& event : found->second) {
                    if (event.feedback_transition) {
                        slice.request_chain_depth_transition(
                            event.superlane, event.depth);
                    } else {
                        slice.set_chain_depth(
                            event.superlane, event.depth);
                    }
                    cycle_record.depth_changes.push_back({
                        cycle,
                        event.phase_id,
                        event.superlane,
                        event.depth,
                        event.feedback_transition});
                }
            }

            if (const auto found = pending_scalars.find(cycle);
                found != pending_scalars.end()) {
                for (const auto& response : found->second) {
                    const auto& event = response.event;
                    for (std::size_t lane = 0;
                         lane < VxmSuperlane::kLaneCount; ++lane) {
                        slice.superlane(event.superlane).lane(lane)
                            .load_local_scalar(
                                event.destination_stage,
                                response.values[lane]);
                    }
                    cycle_record.scalar_loads.push_back({
                        cycle,
                        event.phase_id,
                        event.superlane,
                        event.value,
                        event.source_stream,
                        event.destination_stage,
                        event.element_index});
                }
            }

            auto inputs =
                std::array<std::optional<VxmSlice::StreamMatrix>,
                           VxmSlice::kTileCount>{};
            if (const auto found = pending.find(cycle);
                found != pending.end()) {
                for (const auto& response : found->second) {
                    if (response.request.hold) {
                        held.push_back({
                            {response.request, response.values},
                            response.request.required_cycle
                                + response.request.reuse_count});
                    } else {
                        pack_response(inputs, response);
                        cycle_record.inputs.push_back({
                            cycle,
                            response.request.phase_id,
                            response.request.superlane,
                            response.request.value,
                            response.request.stream_base,
                            response.request.element_index,
                            false});
                    }
                }
            }
            for (const auto& response : held) {
                if (cycle >= response.request.required_cycle
                    && cycle < response.end_cycle) {
                    pack_values(
                        inputs, response.request.superlane,
                        response.request.stream_base, response.values);
                    cycle_record.inputs.push_back({
                        cycle,
                        response.request.phase_id,
                        response.request.superlane,
                        response.request.value,
                        response.request.stream_base,
                        response.request.element_index,
                        true});
                }
            }
            held.erase(
                std::remove_if(
                    held.begin(), held.end(),
                    [cycle](const auto& response) {
                        return cycle + 1 >= response.end_cycle;
                    }),
                held.end());

            for (std::size_t tile = 0;
                 tile < VxmSlice::kTileCount; ++tile) {
                if (inputs[tile]) {
                    slice.set_stream_inputs(tile, *inputs[tile]);
                }
            }

            if (trace) {
                *trace << "adapter cycle " << cycle
                       << " requests="
                       << count_at(events.requests, cycle)
                       << " configs=" << count_at(events.configs, cycle)
                       << " expected_outputs="
                       << count_at(events.outputs, cycle) << '\n';
            }
            slice.tick();

            if (const auto found = events.outputs.find(cycle);
                found != events.outputs.end()) {
                for (const auto& expected : found->second) {
                    const auto& produced =
                        slice.outputs_at(expected.superlane);
                    const auto output = std::find_if(
                        produced.begin(), produced.end(),
                        [&expected](const auto& candidate) {
                            return candidate.stream
                                == expected.stream_base;
                        });
                    if (output == produced.end()) {
                        throw std::logic_error(
                            "VXM adapter expected an output that C Model "
                            "did not produce");
                    }
                    auto values = VxmSuperlaneValues{};
                    for (std::size_t lane = 0;
                         lane < VxmSuperlane::kLaneCount; ++lane) {
                        const auto bytes = std::array<std::uint8_t, 2>{
                            output->byte_values[lane][0],
                            output->byte_values[lane][1]};
                        values[lane] = VxmLane::unpack_float16(bytes);
                    }
                    external.write(expected, values);
                    result.outputs.push_back(expected);
                    cycle_record.outputs.push_back(expected);
                }
            }
            for (std::size_t tile = 0;
                 tile < VxmSlice::kTileCount; ++tile) {
                cycle_record.depths[tile] =
                    slice.superlane(tile).lane(0).chain_depth();
                const auto& trace =
                    slice.superlane(tile).lane(0).last_trace();
                for (std::size_t stage = 0;
                     stage < VxmLane::kAluCount; ++stage) {
                    cycle_record.alu_states[tile][stage] =
                        trace[stage].state;
                }
            }
            result.timeline.push_back(std::move(cycle_record));
        }

        result.cycles = events.last_cycle + 1;
        return result;
    }

private:
    struct DepthEvent {
        std::size_t phase_id{0};
        std::size_t superlane{0};
        VxmChainDepth depth{VxmChainDepth::Eight};
        bool feedback_transition{false};
    };

    struct ScalarEvent {
        VxmExternalStreamRequest request{};
        std::size_t superlane{0};
        std::size_t phase_id{0};
        ValueId value{kInvalidValue};
        std::size_t source_stream{0};
        std::size_t destination_stage{0};
        std::size_t element_index{0};
    };

    struct PendingResponse {
        VxmExternalStreamRequest request{};
        VxmSuperlaneValues values{};
    };

    struct HeldResponse : PendingResponse {
        std::size_t end_cycle{0};
    };

    struct PendingScalarResponse {
        ScalarEvent event{};
        VxmSuperlaneValues values{};
    };

    struct ValueElementKey {
        ValueId value{kInvalidValue};
        std::size_t element_index{0};

        bool operator<(const ValueElementKey& rhs) const
        {
            return value < rhs.value
                || (value == rhs.value
                    && element_index < rhs.element_index);
        }
    };

    struct Events {
        std::map<std::size_t,
                 std::vector<const VxmConfigCommand*>> configs{};
        std::map<std::size_t,
                 std::vector<DepthEvent>> depths{};
        std::map<std::size_t,
                 std::vector<ScalarEvent>> scalar_requests{};
        std::map<std::size_t,
                 std::vector<VxmExternalStreamRequest>> requests{};
        std::map<std::size_t,
                 std::vector<VxmExternalOutputEvent>> outputs{};
        std::vector<std::size_t> phase_shifts{};
        std::vector<std::size_t> phase_sram_waits{};
        std::vector<std::size_t> phase_scalar_load_waits{};
        std::size_t last_cycle{0};
    };

    Events build_events(const VxmCompiledProgram& program) const
    {
        auto events = Events{};
        events.phase_shifts.resize(program.phases.size());
        events.phase_sram_waits.resize(program.phases.size());
        events.phase_scalar_load_waits.resize(program.phases.size());
        auto output_cycles =
            std::map<ValueElementKey, std::size_t>{};

        // Calculate only the delay that cannot be hidden. Each input element
        // is checked against the cycle at which its own producer becomes
        // visible to SRAM, rather than charging every phase a fixed delay.
        for (std::size_t index = 0;
             index < program.phases.size(); ++index) {
            const auto& phase = program.phases[index];
            auto shift = index == 0
                ? std::size_t{0}
                : events.phase_shifts[index - 1];
            const auto shift_before_scalar_load = shift;

            // The current Lane model has one local-scalar register per ALU
            // and does not permit it to be rewritten while that lane is
            // executing. Move only phases that actually load such a scalar;
            // ordinary configuration changes remain fully overlapped.
            if (index != 0 && !phase.local_scalar_loads.empty()) {
                const auto previous_end =
                    program.phases[index - 1].end_cycle
                    + events.phase_shifts[index - 1];
                for (const auto& load : phase.local_scalar_loads) {
                    const auto planned_load = load.load_cycle + shift;
                    if (planned_load < previous_end) {
                        shift += previous_end - planned_load;
                    }
                }
            }
            events.phase_scalar_load_waits[index] =
                shift - shift_before_scalar_load;
            const auto shift_before_sram = shift;
            const auto constrain_read =
                [&](ValueId value, std::size_t element_index,
                    std::size_t relative_required_cycle) {
                    auto earliest_issue = std::size_t{0};
                    const auto producer = output_cycles.find(
                        {value, element_index});
                    if (producer != output_cycles.end()) {
                        earliest_issue =
                            producer->second + timing_.write_latency;
                    }
                    const auto earliest_arrival =
                        earliest_issue + timing_.read_latency;
                    const auto planned_arrival =
                        relative_required_cycle + shift;
                    if (planned_arrival < earliest_arrival) {
                        shift += earliest_arrival - planned_arrival;
                    }
                };

            for (const auto& requirement :
                 phase.stream_requirements) {
                if (requirement.direction
                    != VxmStreamDirection::Input) {
                    continue;
                }
                const auto request_count = requirement.hold
                    ? std::size_t{1}
                    : requirement.transfer_count;
                for (std::size_t item = 0;
                     item < request_count; ++item) {
                    constrain_read(
                        requirement.value,
                        requirement.element_base
                            + (requirement.hold ? 0 : item),
                        requirement.first_cycle
                            + item * requirement.period);
                }
            }
            for (const auto& load : phase.local_scalar_loads) {
                constrain_read(
                    load.value, load.element_index, load.load_cycle);
            }
            if (phase.feedback_from_previous && index != 0
                && shift != events.phase_shifts[index - 1]) {
                throw std::logic_error(
                    "VXM direct-feedback phase has an external operand "
                    "that is not ready at the feedback boundary");
            }
            events.phase_sram_waits[index] =
                shift - shift_before_sram;
            events.phase_shifts[index] = shift;

            for (const auto& requirement :
                 phase.stream_requirements) {
                if (requirement.direction
                    != VxmStreamDirection::Output) {
                    continue;
                }
                for (std::size_t item = 0;
                     item < requirement.transfer_count; ++item) {
                    output_cycles[{
                        requirement.value,
                        requirement.element_base + item}]
                        = requirement.first_cycle + shift
                        + item * requirement.period;
                }
            }
        }

        for (std::size_t phase_index = 0;
             phase_index < program.phases.size(); ++phase_index) {
            const auto& phase = program.phases[phase_index];
            const auto phase_shift =
                events.phase_shifts[phase_index];

            for (const auto& command : phase.config_commands) {
                const auto cycle =
                    command.arrival_cycle + phase_shift;
                events.configs[cycle].push_back(&command);
                events.last_cycle = std::max(events.last_cycle, cycle);
            }

            for (std::size_t tile = 0;
                 tile < VxmSlice::kTileCount; ++tile) {
                const auto tile_shift = phase_shift + tile;
                const auto previous_depth = phase_index == 0
                    ? phase.chain_depth
                    : program.phases[phase_index - 1].chain_depth;
                const auto changes_depth =
                    phase_index != 0
                    && previous_depth != phase.chain_depth;
                if (changes_depth
                    && phase.feedback_from_previous) {
                    if (phase.data_start_cycle == 0) {
                        throw std::logic_error(
                            "VXM feedback transition has no preceding "
                            "boundary cycle");
                    }
                    const auto depth_cycle =
                        phase.data_start_cycle + tile_shift - 1;
                    events.depths[depth_cycle].push_back({
                        phase.phase_id,
                        tile,
                        phase.chain_depth,
                        true});
                    events.last_cycle =
                        std::max(events.last_cycle, depth_cycle);
                } else if (phase_index == 0 || changes_depth) {
                    const auto depth_cycle =
                        phase.data_start_cycle + tile_shift;
                    events.depths[depth_cycle].push_back({
                        phase.phase_id,
                        tile,
                        phase.chain_depth,
                        false});
                    events.last_cycle =
                        std::max(events.last_cycle, depth_cycle);
                }

                for (const auto& load : phase.local_scalar_loads) {
                    const auto required_cycle =
                        load.load_cycle + tile_shift;
                    if (required_cycle < timing_.read_latency) {
                        throw std::logic_error(
                            "VXM SRAM scalar read would require a "
                            "negative issue cycle");
                    }
                    const auto issue_cycle =
                        required_cycle - timing_.read_latency;
                    auto request = VxmExternalStreamRequest{
                        issue_cycle,
                        required_cycle,
                        phase.phase_id,
                        tile,
                        load.value,
                        load.source_stream_base,
                        load.element_index,
                        true,
                        1};
                    events.scalar_requests[issue_cycle].push_back({
                        request,
                        tile,
                        phase.phase_id,
                        load.value,
                        load.source_stream_base,
                        load.destination_stage,
                        load.element_index});
                    events.last_cycle =
                        std::max(events.last_cycle, required_cycle);
                }

                for (const auto& requirement :
                     phase.stream_requirements) {
                    if (requirement.direction
                        == VxmStreamDirection::Input) {
                        const auto request_count = requirement.hold
                            ? std::size_t{1}
                            : requirement.transfer_count;
                        for (std::size_t item = 0;
                             item < request_count; ++item) {
                            const auto required_cycle =
                                requirement.first_cycle + tile_shift
                                + item * requirement.period;
                            if (required_cycle
                                < timing_.read_latency) {
                                throw std::logic_error(
                                    "VXM external request would require "
                                    "a negative issue cycle");
                            }
                            const auto issue_cycle =
                                required_cycle - timing_.read_latency;
                            events.requests[issue_cycle].push_back({
                                issue_cycle,
                                required_cycle,
                                phase.phase_id,
                                tile,
                                requirement.value,
                                requirement.stream_base,
                                requirement.element_base
                                    + (requirement.hold ? 0 : item),
                                requirement.hold,
                                requirement.reuse_count});
                        }
                    } else {
                        for (std::size_t item = 0;
                             item < requirement.transfer_count; ++item) {
                            const auto cycle =
                                requirement.first_cycle + tile_shift
                                + item * requirement.period;
                            events.outputs[cycle].push_back({
                                cycle,
                                cycle + timing_.write_latency,
                                phase.phase_id,
                                tile,
                                requirement.value,
                                requirement.stream_base,
                                requirement.element_base + item});
                            events.last_cycle =
                                std::max(events.last_cycle, cycle);
                        }
                    }
                }
            }
        }
        return events;
    }

    static void pack_response(
        std::array<std::optional<VxmSlice::StreamMatrix>,
                   VxmSlice::kTileCount>& inputs,
        const PendingResponse& response)
    {
        pack_values(
            inputs, response.request.superlane,
            response.request.stream_base, response.values);
    }

    static void pack_values(
        std::array<std::optional<VxmSlice::StreamMatrix>,
                   VxmSlice::kTileCount>& inputs,
        std::size_t superlane, std::size_t stream_base,
        const VxmSuperlaneValues& values)
    {
        if (superlane >= VxmSlice::kTileCount
            || stream_base + VxmLane::kStreamGroupBytes
                > VxmLane::kInputStreams) {
            throw std::out_of_range(
                "VXM adapter Stream placement is outside the Slice");
        }
        if (!inputs[superlane]) {
            inputs[superlane] = VxmSlice::StreamMatrix{};
        }
        for (std::size_t lane = 0;
             lane < VxmSuperlane::kLaneCount; ++lane) {
            const auto bytes = VxmLane::pack_float16(values[lane]);
            for (std::size_t byte = 0;
                 byte < VxmLane::kStreamGroupBytes; ++byte) {
                (*inputs[superlane])[lane][stream_base + byte] =
                    bytes[byte];
            }
        }
    }

    template<typename Map>
    static std::size_t count_at(
        const Map& events, std::size_t cycle)
    {
        const auto found = events.find(cycle);
        return found == events.end() ? 0 : found->second.size();
    }

    VxmSramTiming timing_{};
};

} // namespace ftlpu::vxm::compiler
