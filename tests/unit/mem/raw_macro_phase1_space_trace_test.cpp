#include "ftlpu/system/tsp_slice_system.hpp"
#include "raw_macro_phase1_space/shared_input.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace ftlpu;
using namespace ftlpu::test::raw_macro_phase1_space;

void require(bool condition, std::string_view message)
{
    if (!condition) throw std::runtime_error(std::string(message));
}

std::string json_escape(std::string_view text)
{
    std::ostringstream output;
    for (const char value : text) {
        if (value == '\\' || value == '"') output << '\\';
        output << value;
    }
    return output.str();
}

std::string binary(std::uint64_t value, std::size_t width)
{
    std::string result(width, '0');
    for (std::size_t bit = 0; bit < width; ++bit)
        if ((value >> bit) & 1u) result[width - bit - 1] = '1';
    return result;
}

std::string hex_word(std::uint64_t value)
{
    std::ostringstream output;
    output << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
    return output.str();
}

std::string hex_bytes(const StreamPayloadTileSegment& bytes)
{
    std::ostringstream output;
    output << "0x" << std::hex << std::setfill('0');
    for (const auto byte : bytes) output << std::setw(2) << +byte;
    return output.str();
}

std::uint64_t pack_bytes(const StreamPayloadTileSegment& bytes)
{
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index)
        value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8);
    return value;
}

std::string_view opcode_name(MemOpcode opcode)
{
    switch (opcode) {
    case MemOpcode::Read: return "READ";
    case MemOpcode::Write: return "WRITE";
    case MemOpcode::Gather: return "GATHER";
    case MemOpcode::Scatter: return "SCATTER";
    }
    return "UNKNOWN";
}

std::string_view direction_name(StreamDirection direction)
{
    return direction == StreamDirection::East ? "E" : "W";
}

std::string_view hemisphere_name_upper(Hemisphere hemisphere)
{
    return hemisphere == Hemisphere::East ? "EAST" : "WEST";
}

std::string mem_instance(Hemisphere hemisphere, std::size_t mem_slice,
    std::size_t bank)
{
    return "mem_" + std::string(hemisphere == Hemisphere::East ? "e" : "w")
        + "_s" + std::to_string(mem_slice) + "_b" + std::to_string(bank);
}

std::string mem_hemisphere_instance(Hemisphere hemisphere)
{
    return hemisphere == Hemisphere::East ? "mem_e" : "mem_w";
}

struct LocatedCommand {
    Hemisphere hemisphere{Hemisphere::East};
    MemArrayModel::InstructionTrace command{};
};

struct ExpectedCounts {
    std::size_t issue_accept{};
    std::size_t tile_command{};
    std::size_t read_sample{};
    std::size_t write_commit{};
    std::size_t stream_consume{};
    std::size_t read_response{};
    std::size_t srf_inject{};
    std::size_t drain_cycle{};

    std::size_t architecture_events() const
    {
        return issue_accept + tile_command + read_sample + write_commit
            + stream_consume + read_response + srf_inject + 1;
    }
};

ExpectedCounts expected_counts(const CaseSpec& spec)
{
    ExpectedCounts result{};
    result.issue_accept = spec.operations.size();
    result.tile_command = spec.operations.size() * hw::kTileRows;
    for (const auto& operation : spec.operations) {
        if (operation.instruction.opcode == MemOpcode::Read) {
            result.read_sample += hw::kTileRows;
            result.read_response += hw::kTileRows;
            result.srf_inject += hw::kTileRows;
            result.drain_cycle = std::max(result.drain_cycle,
                operation.issue_cycle + hw::kTileRows);
        } else {
            result.write_commit += hw::kTileRows;
            result.stream_consume += hw::kTileRows;
            result.drain_cycle = std::max(result.drain_cycle,
                operation.issue_cycle + hw::kTileRows - 1);
        }
    }
    return result;
}

StreamPayloadTileSegment expected_read_bytes(
    const OperationSpec& operation, std::size_t tile)
{
    StreamPayloadTileSegment bytes{};
    for (std::size_t lane = 0; lane < bytes.size(); ++lane)
        bytes[lane] = read_preload_byte(operation, tile, lane);
    return bytes;
}

StreamPayloadTileSegment expected_write_bytes(std::size_t case_index,
    std::size_t operation_index, const OperationSpec& operation,
    std::size_t tile)
{
    StreamPayloadTileSegment bytes{};
    for (std::size_t lane = 0; lane < bytes.size(); ++lane)
        bytes[lane] = write_stimulus_byte(
            case_index, operation_index, operation, tile, lane);
    return bytes;
}

LocatedCommand command_from(const OperationSpec& operation, std::size_t tile)
{
    return {operation.hemisphere,
        {operation.mem_slice, operation.bank, tile, operation.instruction}};
}

class TraceWriter {
public:
    TraceWriter(const std::filesystem::path& path, std::string profile)
        : profile_(std::move(profile))
    {
        if (path.has_parent_path())
            std::filesystem::create_directories(path.parent_path());
        output_.open(path, std::ios::out | std::ios::trunc);
        if (!output_) throw std::runtime_error("cannot create Phase1 trace");
    }

    void emit(const CaseSpec& spec, std::size_t cycle,
        std::string_view phase, std::string_view sample_point,
        std::string_view module, std::string_view instance,
        std::string_view signal, std::size_t width, std::uint64_t value,
        std::string_view event_type, std::string_view mapping_status,
        std::string_view category, std::string_view scope,
        std::string_view source_ref,
        std::optional<LocatedCommand> located = std::nullopt,
        std::string_view data = "-",
        std::string_view hemisphere_scope = "GLOBAL")
    {
        int mem_slice = -1;
        int bank = -1;
        int tile = -1;
        int address = -1;
        int stream = -1;
        int preserve = 0;
        std::string_view opcode = "NONE";
        std::string_view direction = "NA";
        std::string_view hemisphere = hemisphere_scope;
        if (located) {
            const auto& command = located->command;
            hemisphere = hemisphere_name_upper(located->hemisphere);
            mem_slice = static_cast<int>(command.mem_slice);
            bank = static_cast<int>(command.bank);
            tile = static_cast<int>(command.tile);
            address = static_cast<int>(command.instruction.address);
            stream = static_cast<int>(command.instruction.stream);
            preserve = command.instruction.preserve_stream ? 1 : 0;
            opcode = opcode_name(command.instruction.opcode);
            direction = direction_name(command.instruction.stream_id().direction());
        }
        output_
            << "{\"schema\":\"ftlpu.raw_macro.level2.trace.v1\""
            << ",\"run_id\":\"cmodel_raw_macro_phase1_space_"
            << json_escape(profile_) << "\""
            << ",\"case_id\":\"" << json_escape(spec.id) << "\""
            << ",\"source\":\"CMODEL\""
            << ",\"global_cycle\":" << cycle
            << ",\"logical_cycle\":" << cycle
            << ",\"transport_tick\":" << cycle
            << ",\"phase\":\"" << phase << "\""
            << ",\"sample_point\":\"" << sample_point << "\""
            << ",\"event_sequence\":" << event_sequence_++
            << ",\"module\":\"" << module << "\""
            << ",\"instance\":\"" << instance << "\""
            << ",\"signal\":\"" << signal << "\""
            << ",\"width\":" << width
            << ",\"value\":\"" << binary(value, width) << "\""
            << ",\"event_type\":\"" << event_type << "\""
            << ",\"mapping_status\":\"" << mapping_status << "\""
            << ",\"source_ref\":\"" << json_escape(source_ref) << "\""
            << ",\"hemisphere\":\"" << hemisphere << "\""
            << ",\"mem_slice\":" << mem_slice
            << ",\"bank\":" << bank
            << ",\"tile\":" << tile
            << ",\"opcode\":\"" << opcode << "\""
            << ",\"address\":" << address
            << ",\"stream\":" << stream
            << ",\"direction\":\"" << direction << "\""
            << ",\"preserve\":" << preserve
            << ",\"data\":\"" << data << "\""
            << ",\"category\":\"" << category << "\""
            << ",\"scope\":\"" << scope << "\"}\n";
        if (!output_) throw std::runtime_error("failed to write Phase1 trace");
    }

private:
    std::ofstream output_{};
    std::string profile_{};
    std::size_t event_sequence_{0};
};

std::optional<std::size_t> operation_index_for(const CaseSpec& spec,
    Hemisphere hemisphere, const MemArrayModel::InstructionTrace& trace,
    std::size_t observation_cycle, std::size_t response_delay = 0)
{
    if (observation_cycle < trace.tile + response_delay) return std::nullopt;
    const auto issue_cycle = observation_cycle - trace.tile - response_delay;
    for (std::size_t index = 0; index < spec.operations.size(); ++index) {
        const auto& candidate = spec.operations[index];
        if (candidate.issue_cycle == issue_cycle
            && candidate.hemisphere == hemisphere
            && candidate.mem_slice == trace.mem_slice
            && candidate.bank == trace.bank
            && candidate.instruction.opcode == trace.instruction.opcode
            && candidate.instruction.address == trace.instruction.address
            && candidate.instruction.stream == trace.instruction.stream
            && candidate.instruction.preserve_stream
                == trace.instruction.preserve_stream)
            return index;
    }
    return std::nullopt;
}

void validate_profile(const ProfileSpec& profile)
{
    require(!profile.id.empty(), "Phase1 profile has no identifier");
    require(!profile.cases.empty(), "Phase1 profile has no cases");
    require(profile.rtl_active_mem_slices > 0,
        "Phase1 profile active slice count must be positive");
    require(profile.rtl_active_mem_slices
            <= profile.rtl_physical_mem_slices,
        "Phase1 active slice count exceeds physical RTL slices");
    require(profile.rtl_physical_mem_slices <= hw::kMemSliceColumns,
        "Phase1 physical RTL slice count exceeds frozen LPU geometry");
    require(profile.rtl_physical_mem_slices % hw::kMemSlicesPerGroup == 0,
        "Phase1 physical RTL slices must contain complete four-slice groups");
    require(profile.rtl_active_rows > 0
            && profile.rtl_active_rows <= hw::kSramDepthRows,
        "Phase1 active row depth is outside frozen LPU geometry");
}

void validate_case(const ProfileSpec& profile, const CaseSpec& spec)
{
    require(!spec.operations.empty(), "Phase1 case has no operations");
    std::set<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>>
        bank_cycles;
    for (const auto& operation : spec.operations) {
        require(hemisphere_index(operation.hemisphere) < hw::kHemispheres,
            "Phase1 operation hemisphere is out of range");
        require(operation.mem_slice < profile.rtl_active_mem_slices,
            "Phase1 operation exceeds profile active slice count");
        require(operation.mem_slice < hw::kMemSliceColumns,
            "Phase1 operation slice is out of CModel range");
        require(operation.bank < hw::kMemBanksPerSlice,
            "Phase1 operation bank is out of range");
        require(operation.instruction.address < profile.rtl_active_rows,
            "Phase1 operation exceeds profile active row depth");
        require(operation.instruction.opcode == MemOpcode::Read
                || operation.instruction.opcode == MemOpcode::Write,
            "Phase1 covers native Read/Write only");
        require(bank_cycles.emplace(operation.issue_cycle,
                    hemisphere_index(operation.hemisphere),
                    operation.mem_slice, operation.bank).second,
            "Phase1 case issues more than one command to one Bank in a cycle");
    }
}

void write_manifest(const std::filesystem::path& path,
    const ProfileSpec& profile)
{
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create Phase1 manifest");
    output << "{\n  \"schema\": \"ftlpu.raw_macro.phase1.space.input.v2\",\n"
           << "  \"profile\": \"" << profile.id << "\",\n"
           << "  \"word_bits\": 64,\n"
           << "  \"rtl_contract\": {\"use_native_mem\":true,"
              "\"active_mem_slices\":" << profile.rtl_active_mem_slices
           << ",\"physical_mem_slices\":"
           << profile.rtl_physical_mem_slices
           << ",\"active_rows\":" << profile.rtl_active_rows
           << ",\"hemispheres\":" << hw::kHemispheres
           << ",\"banks_per_slice\":" << hw::kMemBanksPerSlice
           << ",\"tiles\":" << hw::kTileRows << "},\n"
           << "  \"cases\": [\n";
    for (std::size_t case_index = 0; case_index < profile.cases.size(); ++case_index) {
        const auto& spec = profile.cases[case_index];
        const auto expected = expected_counts(spec);
        output << "    {\"id\":\"" << json_escape(spec.id) << "\","
               << "\"expected\":{\"issue_accept\":" << expected.issue_accept
               << ",\"tile_command\":" << expected.tile_command
               << ",\"read_sample\":" << expected.read_sample
               << ",\"write_commit\":" << expected.write_commit
               << ",\"stream_consume\":" << expected.stream_consume
               << ",\"read_response\":" << expected.read_response
               << ",\"srf_inject\":" << expected.srf_inject
               << ",\"architecture_events\":" << expected.architecture_events()
               << ",\"drain_cycle\":" << expected.drain_cycle << "},"
               << "\"operations\":[\n";
        for (std::size_t op_index = 0; op_index < spec.operations.size(); ++op_index) {
            const auto& operation = spec.operations[op_index];
            output << "      {\"operation_index\":" << op_index
                   << ",\"issue_cycle\":" << operation.issue_cycle
                   << ",\"hemisphere\":\""
                   << hemisphere_name_upper(operation.hemisphere) << "\""
                   << ",\"mem_slice\":" << operation.mem_slice
                   << ",\"bank\":" << operation.bank
                   << ",\"queue_id\":" << queue_id(operation)
                   << ",\"machine_word\":\"" << hex_word(operation.machine_word())
                   << "\",\"opcode\":\"" << opcode_name(operation.instruction.opcode)
                   << "\",\"address\":" << operation.instruction.address
                   << ",\"stream\":" << operation.instruction.stream
                   << ",\"direction\":\""
                   << direction_name(operation.instruction.stream_id().direction())
                   << "\",\"preserve\":"
                   << (operation.instruction.preserve_stream ? "true" : "false")
                   << ",\"tile_data\":[";
            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
                const auto bytes = operation.instruction.opcode == MemOpcode::Read
                    ? expected_read_bytes(operation, tile)
                    : expected_write_bytes(case_index, op_index, operation, tile);
                if (tile != 0) output << ',';
                output << '"' << hex_bytes(bytes) << '"';
            }
            output << "]}" << (op_index + 1 == spec.operations.size()
                    ? "\n" : ",\n");
        }
        output << "    ]}" << (case_index + 1 == profile.cases.size()
                ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

void require_count(const std::map<std::string, std::size_t>& counts,
    std::string_view key, std::size_t expected)
{
    const auto found = counts.find(std::string(key));
    const auto actual = found == counts.end() ? 0 : found->second;
    if (actual != expected)
        throw std::runtime_error(std::string(key) + " count mismatch: expected "
            + std::to_string(expected) + ", got " + std::to_string(actual));
}

ExpectedCounts run_case(TraceWriter& writer, const ProfileSpec& profile,
    const CaseSpec& spec, std::size_t case_index)
{
    validate_case(profile, spec);
    TspSliceSystem system({}, SxmTransposeArchitecture::LocalBlocks,
        MemArrayModel::ReadTiming::RegisteredR1);
    for (std::size_t side = 0; side < hw::kHemispheres; ++side)
        require(system.mem_array(static_cast<Hemisphere>(side)).read_timing()
                == MemArrayModel::ReadTiming::RegisteredR1,
            "Phase1 requires RegisteredR1 MEM timing");

    for (const auto& operation : spec.operations) {
        if (operation.instruction.opcode != MemOpcode::Read) continue;
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
                system.initialize_mem_sram_lane_byte(operation.hemisphere,
                    operation.mem_slice, operation.bank, tile,
                    operation.instruction.address, lane,
                    read_preload_byte(operation, tile, lane));
    }

    const auto expected = expected_counts(spec);
    std::map<std::string, std::size_t> counts;
    std::size_t issued = 0;
    bool drained = false;
    const auto max_cycles = expected.drain_cycle + 5;
    for (std::size_t cycle = 0; cycle < max_cycles; ++cycle) {
        for (std::size_t op_index = 0; op_index < spec.operations.size(); ++op_index) {
            const auto& operation = spec.operations[op_index];
            if (operation.issue_cycle != cycle) continue;
            auto& mem = system.mem_array(operation.hemisphere);
            require(mem.mem_issue_ready(operation.mem_slice, operation.bank),
                "MEM issue_ready must be tied high");
            const auto word = operation.machine_word();
            const auto decoded = isa::decode_mem_instruction(word);
            require(isa::encode_mem_instruction(decoded) == word,
                "formal 64-bit MEM word failed round-trip");
            mem.enqueue_instruction(operation.mem_slice, operation.bank, decoded);
            const auto command = command_from(operation, 0);
            const auto instance = mem_instance(operation.hemisphere,
                operation.mem_slice, operation.bank);
            writer.emit(spec, cycle, "LOW", "PRE_POSEDGE",
                "lpu_mem_execution_frontend", instance,
                "execution_issue_ready", 1, 1, "SIGNAL_SAMPLE", "DIRECT",
                "CANONICAL", "ICU", "MemArrayModel::mem_issue_ready", command);
            writer.emit(spec, cycle, "LOW", "PRE_POSEDGE",
                "architecture", "architecture", "MEM_ISSUE_ACCEPT", 64, word,
                "ARCH_EVENT", "SEMANTIC", "EVENT", "ICU",
                "external issue lane + isa::decode_mem_instruction", command,
                hex_word(word));
            ++counts["issue_accept"];
            ++issued;
        }

        std::set<std::tuple<std::size_t, std::size_t, std::size_t,
            std::size_t>> write_cells;
        for (std::size_t op_index = 0; op_index < spec.operations.size(); ++op_index) {
            const auto& operation = spec.operations[op_index];
            if (operation.instruction.opcode != MemOpcode::Write
                || cycle < operation.issue_cycle
                || cycle >= operation.issue_cycle + hw::kTileRows)
                continue;
            const auto tile = cycle - operation.issue_cycle;
            const auto bytes = expected_write_bytes(
                case_index, op_index, operation, tile);
            const auto stream = operation.instruction.stream_id();
            const auto& mem = system.mem_array(operation.hemisphere);
            const auto column = mem.ports().input_column(
                operation.mem_slice, stream.direction());
            require(write_cells.emplace(hemisphere_index(operation.hemisphere),
                        column, tile, operation.instruction.stream).second,
                "Phase1 profile creates an undefined SRF write-input collision");
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
                system.stream_fabric(operation.hemisphere).initialize_cell(
                    column, tile, lane, stream,
                    StreamCell::Valid(bytes[lane], false, cycle));
        }

        TspSliceSystem::LogSinks sinks{};
        sinks.capture_mem_trace = true;
        system.tick(sinks);

        for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
            const auto hemisphere = static_cast<Hemisphere>(side);
            auto& mem = system.mem_array(hemisphere);
            for (const auto& trace : mem.executed_instructions()) {
                const auto op_index = operation_index_for(
                    spec, hemisphere, trace, cycle);
                require(op_index.has_value(), "executed an unknown Phase1 command");
                const auto& operation = spec.operations[*op_index];
                require(cycle == operation.issue_cycle + trace.tile,
                    "Phase1 command did not advance one Tile per edge");
                writer.emit(spec, cycle, "LOW", "PRE_POSEDGE",
                    "lpu_mem_command_pipeline",
                    mem_instance(hemisphere, trace.mem_slice, trace.bank),
                    "MEM_TILE_COMMAND", 64, operation.machine_word(),
                    "ARCH_EVENT", "SEMANTIC", "EVENT", "MEM",
                    "MemArrayModel::executed_instructions",
                    LocatedCommand {hemisphere, trace},
                    hex_word(operation.machine_word()));
                ++counts["tile_command"];
            }

            for (const auto& transfer : mem.executed_transfers()) {
                const auto opcode = transfer.kind
                        == MemArrayModel::MemTransfer::Kind::LoadSramToStream
                    ? MemOpcode::Read : MemOpcode::Write;
                const auto trace = MemArrayModel::InstructionTrace {
                    transfer.mem_slice, transfer.bank, transfer.tile,
                    opcode == MemOpcode::Read
                        ? MemInstruction::Read(transfer.address, transfer.stream)
                        : MemInstruction::Write(transfer.address, transfer.stream)};
                const auto op_index = operation_index_for(
                    spec, hemisphere, trace, cycle);
                require(op_index.has_value(), "unknown Phase1 MEM transfer");
                const auto& operation = spec.operations[*op_index];
                const auto located = LocatedCommand {hemisphere, trace};
                const auto instance = mem_instance(
                    hemisphere, transfer.mem_slice, transfer.bank);
                if (opcode == MemOpcode::Read) {
                    require(transfer.bytes
                            == expected_read_bytes(operation, transfer.tile),
                        "Phase1 SRAM read sample differs from spatial preload");
                    writer.emit(spec, cycle, "LOW", "PRE_POSEDGE",
                        "lpu_mem_sram_leaf", instance,
                        "MEM_SRAM_READ_SAMPLE", hw::kLanesPerTile * 8,
                        pack_bytes(transfer.bytes), "ARCH_EVENT", "SEMANTIC",
                        "EVENT", "SRAM", "MemArrayModel::executed_transfers",
                        located, hex_bytes(transfer.bytes));
                    ++counts["read_sample"];
                } else {
                    const auto expected_bytes = expected_write_bytes(
                        case_index, *op_index, operation, transfer.tile);
                    require(transfer.bytes == expected_bytes,
                        "Phase1 stream consume differs from spatial stimulus");
                    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
                        require(mem.sram_lane_byte(transfer.mem_slice,
                                    transfer.bank, transfer.tile,
                                    transfer.address, lane) == transfer.bytes[lane],
                            "Phase1 SRAM write was not committed at POSEDGE");
                    writer.emit(spec, cycle, "POSEDGE", "POST_POSEDGE",
                        "lpu_mem_sram_leaf", instance,
                        "MEM_SRAM_WRITE_COMMIT", hw::kLanesPerTile * 8,
                        pack_bytes(transfer.bytes), "ARCH_EVENT", "SEMANTIC",
                        "EVENT", "SRAM", "MemArrayModel::posedge", located,
                        hex_bytes(transfer.bytes));
                    ++counts["write_commit"];
                }
            }

            for (const auto& transfer : mem.consumed_transfers()) {
                const auto trace = MemArrayModel::InstructionTrace {
                    transfer.mem_slice, transfer.bank, transfer.tile,
                    MemInstruction::Write(transfer.address, transfer.stream)};
                const auto op_index = operation_index_for(
                    spec, hemisphere, trace, cycle);
                require(op_index.has_value(), "unknown Phase1 stream consume");
                writer.emit(spec, cycle, "LOW", "PRE_POSEDGE",
                    "lpu_mem_stream_input",
                    mem_instance(hemisphere, transfer.mem_slice, transfer.bank),
                    "MEM_STREAM_CONSUME", hw::kLanesPerTile * 8,
                    pack_bytes(transfer.bytes), "ARCH_EVENT", "SEMANTIC",
                    "EVENT", "STREAM", "MemArrayModel::consumed_transfers",
                    LocatedCommand {hemisphere, trace}, hex_bytes(transfer.bytes));
                ++counts["stream_consume"];
            }

            for (const auto& response : mem.read_responses()) {
                const auto trace = MemArrayModel::InstructionTrace {
                    response.mem_slice, response.bank, response.tile,
                    MemInstruction::Read(response.address, response.stream)};
                const auto op_index = operation_index_for(
                    spec, hemisphere, trace, cycle);
                require(op_index.has_value(), "unknown Phase1 registered response");
                require(response.bytes == expected_read_bytes(
                            spec.operations[*op_index], response.tile),
                    "Phase1 registered response data mismatch");
                writer.emit(spec, cycle, "POSEDGE", "POST_POSEDGE",
                    "lpu_mem_sram_leaf",
                    mem_instance(hemisphere, response.mem_slice, response.bank),
                    "MEM_READ_RESPONSE", hw::kLanesPerTile * 8,
                    pack_bytes(response.bytes), "ARCH_EVENT", "SEMANTIC",
                    "EVENT", "SRAM", "MemArrayModel::read_responses",
                    LocatedCommand {hemisphere, trace}, hex_bytes(response.bytes));
                ++counts["read_response"];
            }

            for (const auto& response : mem.staged_read_responses()) {
                const auto trace = MemArrayModel::InstructionTrace {
                    response.mem_slice, response.bank, response.tile,
                    MemInstruction::Read(response.address, response.stream)};
                const auto op_index = operation_index_for(
                    spec, hemisphere, trace, cycle, 1);
                require(op_index.has_value(), "unknown Phase1 SRF injection");
                require(response.bytes == expected_read_bytes(
                            spec.operations[*op_index], response.tile),
                    "Phase1 SRF injection data mismatch");
                writer.emit(spec, cycle, "POSEDGE", "POST_POSEDGE",
                    "lpu_stream_register_fabric",
                    hemisphere == Hemisphere::East ? "east_srf" : "west_srf",
                    "MEM_SRF_INJECT_COMMIT", hw::kLanesPerTile * 8,
                    pack_bytes(response.bytes), "ARCH_EVENT", "SEMANTIC",
                    "EVENT", "STREAM", "MemArrayModel::staged_read_responses",
                    LocatedCommand {hemisphere, trace}, hex_bytes(response.bytes));
                ++counts["srf_inject"];
            }

            const bool busy = !mem.idle();
            writer.emit(spec, cycle, "POSEDGE", "POST_COMB",
                "cmodel_mem_diagnostic", mem_hemisphere_instance(hemisphere),
                "mem_busy_derived", 1, busy ? 1 : 0,
                "STATE_SAMPLE", "DERIVED", "DIAGNOSTIC", "MEM",
                "!MemArrayModel::idle", std::nullopt, "-",
                hemisphere_name_upper(hemisphere));
        }

        bool all_idle = true;
        for (std::size_t side = 0; side < hw::kHemispheres; ++side)
            all_idle = all_idle
                && system.mem_array(static_cast<Hemisphere>(side)).idle();
        if (issued == spec.operations.size() && all_idle) {
            require(cycle == expected.drain_cycle,
                "Phase1 global Drain cycle differs from static contract");
            writer.emit(spec, cycle, "POSEDGE", "POST_COMB",
                "architecture", "architecture", "MEM_DRAINED", 1, 1,
                "ARCH_EVENT", "SEMANTIC", "EVENT", "MEM",
                "all hemisphere MemArrayModel::idle");
            drained = true;
            break;
        }
    }

    require(drained, "Phase1 case did not globally drain");
    require_count(counts, "issue_accept", expected.issue_accept);
    require_count(counts, "tile_command", expected.tile_command);
    require_count(counts, "read_sample", expected.read_sample);
    require_count(counts, "write_commit", expected.write_commit);
    require_count(counts, "stream_consume", expected.stream_consume);
    require_count(counts, "read_response", expected.read_response);
    require_count(counts, "srf_inject", expected.srf_inject);
    return expected;
}

void write_profile_summary(const std::filesystem::path& path,
    const ProfileSpec& profile,
    const std::vector<ExpectedCounts>& results)
{
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create Phase1 summary");
    output << "{\n  \"schema\":\"ftlpu.raw_macro.phase1.space.summary.v1\",\n"
           << "  \"profile\":\"" << profile.id << "\",\n"
           << "  \"status\":\"PASS\",\n  \"cases\":[\n";
    for (std::size_t index = 0; index < profile.cases.size(); ++index) {
        output << "    {\"case_id\":\"" << profile.cases[index].id
               << "\",\"architecture_events\":"
               << results[index].architecture_events()
               << ",\"drain_cycle\":" << results[index].drain_cycle << "}"
               << (index + 1 == profile.cases.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

void append_coverage(std::ostream& output, const ProfileSpec& profile,
    const CaseSpec& spec)
{
    struct Counts { std::size_t operations{}, reads{}, writes{}; };
    std::map<std::tuple<std::string, std::size_t, std::size_t>, Counts> counts;
    for (const auto& operation : spec.operations) {
        auto& count = counts[{std::string(hemisphere_name_upper(
            operation.hemisphere)), operation.mem_slice, operation.bank}];
        ++count.operations;
        if (operation.instruction.opcode == MemOpcode::Read) ++count.reads;
        else ++count.writes;
    }
    const auto expected = expected_counts(spec);
    for (const auto& [coordinate, count] : counts) {
        const auto& [hemisphere, mem_slice, bank] = coordinate;
        output << profile.id << ',' << spec.id << ',' << hemisphere << ','
               << mem_slice << ',' << bank << ',' << count.operations << ','
               << count.reads << ',' << count.writes << ','
               << count.operations * hw::kTileRows << ','
               << expected.drain_cycle << '\n';
    }
}

void write_master_manifest(const std::filesystem::path& path,
    const std::vector<ProfileSpec>& all_profiles)
{
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create Phase1 master manifest");
    output << "{\n  \"schema\":\"ftlpu.raw_macro.phase1.space.master.v1\",\n"
              "  \"status\":\"CMODEL_READY_RTL_PENDING\",\n"
              "  \"profiles\":[\n";
    for (std::size_t index = 0; index < all_profiles.size(); ++index) {
        const auto& profile = all_profiles[index];
        output << "    {\"id\":\"" << profile.id
               << "\",\"manifest\":\"" << profile.id
               << "/shared_input_manifest.json\",\"trace\":\""
               << profile.id << "/cmodel_trace.jsonl\"}"
               << (index + 1 == all_profiles.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

} // namespace

int main(int argc, char** argv)
try {
    const auto output_root = argc > 1
        ? std::filesystem::path(argv[1])
        : std::filesystem::current_path() / "raw_macro_phase1_space";
    std::filesystem::create_directories(output_root);
    const auto all_profiles = profiles();
    require(hw::kHemispheres == 2 && hw::kMemSliceColumns == 52
            && hw::kMemBanksPerSlice == 2 && hw::kTileRows == 4
            && hw::kSramDepthRows == 8192,
        "Phase1 formal geometry differs from the frozen LPU32 contract");

    std::ofstream coverage(output_root / "coordinate_coverage.csv",
        std::ios::out | std::ios::trunc);
    if (!coverage) throw std::runtime_error("cannot create coverage CSV");
    coverage << "profile,case_id,hemisphere,mem_slice,bank,operations,reads,"
                "writes,tile_events,drain_cycle\n";

    std::size_t profile_count = 0;
    std::size_t case_count = 0;
    for (const auto& profile : all_profiles) {
        validate_profile(profile);
        const auto profile_root = output_root / profile.id;
        std::filesystem::create_directories(profile_root);
        write_manifest(profile_root / "shared_input_manifest.json", profile);
        TraceWriter writer(profile_root / "cmodel_trace.jsonl", profile.id);
        std::vector<ExpectedCounts> results;
        for (std::size_t case_index = 0;
             case_index < profile.cases.size(); ++case_index) {
            const auto& spec = profile.cases[case_index];
            results.push_back(run_case(writer, profile, spec, case_index));
            append_coverage(coverage, profile, spec);
            ++case_count;
        }
        write_profile_summary(profile_root / "cmodel_summary.json",
            profile, results);
        ++profile_count;
    }
    write_master_manifest(output_root / "master_manifest.json", all_profiles);
    std::cout << "PHASE1_SPACE_PASS profiles=" << profile_count
              << " cases=" << case_count
              << " output=" << output_root.string() << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "PHASE1_SPACE_FAIL: " << error.what() << '\n';
    return 1;
}
