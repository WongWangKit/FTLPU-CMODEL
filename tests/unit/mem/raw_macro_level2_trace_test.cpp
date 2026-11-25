#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include "raw_macro_level2/shared_input.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ftlpu;
using namespace ftlpu::test::raw_macro_level2;

void require(bool condition, std::string_view message)
{
    if (!condition) throw std::runtime_error(std::string(message));
}

std::string json_escape(std::string_view text)
{
    std::string result;
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
        throw std::invalid_argument("trace binary width must be in [1,64]");
    std::string result(width, '0');
    for (std::size_t bit = 0; bit < width; ++bit)
        result[width - bit - 1] = ((value >> bit) & 1u) ? '1' : '0';
    return result;
}

std::uint64_t pack_bytes(const StreamPayloadTileSegment& bytes)
{
    static_assert(hw::kLanesPerTile <= sizeof(std::uint64_t));
    std::uint64_t result = 0;
    for (std::size_t lane = 0; lane < bytes.size(); ++lane)
        result |= std::uint64_t(bytes[lane]) << (lane * 8);
    return result;
}

std::string hex_word(std::uint64_t value, std::size_t digits = 16)
{
    std::ostringstream output;
    output << "0x" << std::hex << std::setw(static_cast<int>(digits))
           << std::setfill('0') << value;
    return output.str();
}

std::string hex_bytes(const StreamPayloadTileSegment& bytes)
{
    std::ostringstream output;
    output << "0x" << std::hex << std::setfill('0');
    for (const auto value : bytes)
        output << std::setw(2) << static_cast<unsigned>(value);
    return output.str();
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

class TraceWriter {
public:
    explicit TraceWriter(const std::filesystem::path& path)
    {
        if (path.has_parent_path())
            std::filesystem::create_directories(path.parent_path());
        output_.open(path, std::ios::out | std::ios::trunc);
        if (!output_) throw std::runtime_error("cannot create Level2 trace");
    }

    void emit(const CaseSpec& spec, std::size_t cycle,
        std::string_view phase, std::string_view sample_point,
        std::string_view module, std::string_view instance,
        std::string_view signal, std::size_t width, std::uint64_t value,
        std::string_view event_type, std::string_view mapping_status,
        std::string_view category, std::string_view scope,
        std::string_view source_ref,
        std::optional<MemArrayModel::InstructionTrace> command = std::nullopt,
        std::string_view data = "-")
    {
        int mem_slice = -1;
        int bank = -1;
        int tile = -1;
        int address = -1;
        int stream = -1;
        int preserve = 0;
        std::string_view opcode = "NONE";
        std::string_view direction = "NA";
        if (command) {
            mem_slice = static_cast<int>(command->mem_slice);
            bank = static_cast<int>(command->bank);
            tile = static_cast<int>(command->tile);
            address = static_cast<int>(command->instruction.address);
            stream = static_cast<int>(command->instruction.stream);
            preserve = command->instruction.preserve_stream ? 1 : 0;
            opcode = opcode_name(command->instruction.opcode);
            direction = direction_name(command->instruction.stream_id().direction());
        }
        output_
            << "{\"schema\":\"ftlpu.raw_macro.level2.trace.v1\""
            << ",\"run_id\":\"cmodel_raw_macro_level2_v1\""
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
            << ",\"hemisphere\":\"EAST\""
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
        if (!output_) throw std::runtime_error("failed to write Level2 trace");
    }

private:
    std::ofstream output_{};
    std::uint64_t event_sequence_{};
};

MemArrayModel::InstructionTrace command_from(
    const OperationSpec& operation, std::size_t tile)
{
    return {operation.mem_slice, operation.bank, tile, operation.instruction};
}

StreamPayloadTileSegment expected_read_bytes(
    const OperationSpec& operation, std::size_t tile)
{
    StreamPayloadTileSegment bytes{};
    for (std::size_t lane = 0; lane < bytes.size(); ++lane)
        bytes[lane] = read_preload_byte(operation.instruction.address, tile, lane);
    return bytes;
}

StreamPayloadTileSegment expected_write_bytes(
    std::size_t case_index, std::size_t operation_index, std::size_t tile)
{
    StreamPayloadTileSegment bytes{};
    for (std::size_t lane = 0; lane < bytes.size(); ++lane)
        bytes[lane] = write_stimulus_byte(case_index, operation_index, tile, lane);
    return bytes;
}

std::optional<std::size_t> operation_index_for(
    const CaseSpec& spec, const MemArrayModel::InstructionTrace& trace)
{
    for (std::size_t index = 0; index < spec.operations.size(); ++index) {
        const auto& candidate = spec.operations[index];
        if (candidate.mem_slice == trace.mem_slice
            && candidate.bank == trace.bank
            && candidate.instruction.opcode == trace.instruction.opcode
            && candidate.instruction.address == trace.instruction.address
            && candidate.instruction.stream == trace.instruction.stream)
            return index;
    }
    return std::nullopt;
}

void write_manifest(const std::filesystem::path& path,
    const std::vector<CaseSpec>& all_cases)
{
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create Level2 input manifest");
    output << "{\n  \"schema\": \"ftlpu.raw_macro.level2.input.v1\",\n"
              "  \"word_bits\": 64,\n  \"cases\": [\n";
    for (std::size_t case_index = 0; case_index < all_cases.size(); ++case_index) {
        const auto& spec = all_cases[case_index];
        output << "    {\"id\":\"" << json_escape(spec.id)
               << "\",\"operations\":[\n";
        for (std::size_t op_index = 0; op_index < spec.operations.size(); ++op_index) {
            const auto& operation = spec.operations[op_index];
            output << "      {\"issue_cycle\":" << operation.issue_cycle
                   << ",\"hemisphere\":\"EAST\",\"mem_slice\":"
                   << operation.mem_slice << ",\"bank\":" << operation.bank
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
                    : expected_write_bytes(case_index, op_index, tile);
                if (tile != 0) output << ',';
                output << '"' << hex_bytes(bytes) << '"';
            }
            output << "]}" << (op_index + 1 == spec.operations.size() ? "\n" : ",\n");
        }
        output << "    ]}" << (case_index + 1 == all_cases.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

void run_case(TraceWriter& writer, const CaseSpec& spec, std::size_t case_index)
{
    TspSliceSystem system({}, SxmTransposeArchitecture::LocalBlocks,
        MemArrayModel::ReadTiming::RegisteredR1);
    auto& mem = system.mem_array(Hemisphere::East);
    require(mem.read_timing() == MemArrayModel::ReadTiming::RegisteredR1,
        "Level2 requires RegisteredR1 MEM timing");

    for (const auto& operation : spec.operations) {
        if (operation.instruction.opcode != MemOpcode::Read) continue;
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
                system.initialize_mem_sram_lane_byte(Hemisphere::East,
                    operation.mem_slice, operation.bank, tile,
                    operation.instruction.address, lane,
                    read_preload_byte(operation.instruction.address, tile, lane));
    }

    std::size_t issue_count = 0;
    std::size_t command_count = 0;
    std::size_t read_count = 0;
    std::size_t write_count = 0;
    std::size_t response_count = 0;
    std::size_t inject_count = 0;
    bool drained = false;
    const auto last_issue = std::max_element(spec.operations.begin(),
        spec.operations.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.issue_cycle < rhs.issue_cycle;
        })->issue_cycle;
    const auto max_cycles = last_issue + hw::kTileRows + 4;

    for (std::size_t cycle = 0; cycle < max_cycles; ++cycle) {
        for (std::size_t op_index = 0; op_index < spec.operations.size(); ++op_index) {
            const auto& operation = spec.operations[op_index];
            if (operation.issue_cycle != cycle) continue;
            require(mem.mem_issue_ready(operation.mem_slice, operation.bank),
                "MEM issue_ready must be tied high");
            const auto word = operation.machine_word();
            const auto decoded = isa::decode_mem_instruction(word);
            require(isa::encode_mem_instruction(decoded) == word,
                "official MEM machine word failed round-trip");
            mem.enqueue_instruction(operation.mem_slice, operation.bank, decoded);
            const auto command = command_from(operation, 0);
            writer.emit(spec, cycle, "LOW", "PRE_POSEDGE",
                "lpu_mem_execution_frontend", "mem_e_s0_b0",
                "execution_issue_ready", 1, 1, "SIGNAL_SAMPLE", "DIRECT",
                "CANONICAL", "ICU",
                "MemArrayModel::mem_issue_ready", command);
            writer.emit(spec, cycle, "LOW", "PRE_POSEDGE",
                "architecture", "architecture", "MEM_ISSUE_ACCEPT",
                64, word, "ARCH_EVENT", "SEMANTIC", "EVENT", "ICU",
                "external issue injection + isa::decode_mem_instruction",
                command, hex_word(word));
            ++issue_count;
        }

        for (std::size_t op_index = 0; op_index < spec.operations.size(); ++op_index) {
            const auto& operation = spec.operations[op_index];
            if (operation.instruction.opcode != MemOpcode::Write
                || cycle < operation.issue_cycle
                || cycle >= operation.issue_cycle + hw::kTileRows)
                continue;
            const auto tile = cycle - operation.issue_cycle;
            const auto bytes = expected_write_bytes(case_index, op_index, tile);
            const auto stream = operation.instruction.stream_id();
            const auto column = mem.ports().input_column(
                operation.mem_slice, stream.direction());
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
                system.stream_fabric(Hemisphere::East).initialize_cell(
                    column, tile, lane, stream,
                    StreamCell::Valid(bytes[lane], false, cycle));
        }

        TspSliceSystem::LogSinks sinks{};
        sinks.capture_mem_trace = true;
        system.tick(sinks);

        for (const auto& trace : mem.executed_instructions()) {
            const auto op_index = operation_index_for(spec, trace);
            require(op_index.has_value(), "executed an unknown MEM instruction");
            const auto& operation = spec.operations[*op_index];
            require(cycle == operation.issue_cycle + trace.tile,
                "tile command did not advance exactly one tile per edge");
            writer.emit(spec, cycle, "LOW", "PRE_POSEDGE",
                "lpu_mem_command_pipeline", "mem_e_s0_b0",
                "MEM_TILE_COMMAND", 64, operation.machine_word(),
                "ARCH_EVENT", "SEMANTIC", "EVENT", "MEM",
                "MemArrayModel::executed_instructions", trace,
                hex_word(operation.machine_word()));
            ++command_count;
        }

        for (const auto& transfer : mem.executed_transfers()) {
            const auto command = MemArrayModel::InstructionTrace {
                transfer.mem_slice, transfer.bank, transfer.tile,
                transfer.kind == MemArrayModel::MemTransfer::Kind::LoadSramToStream
                    ? MemInstruction::Read(transfer.address, transfer.stream)
                    : MemInstruction::Write(transfer.address, transfer.stream)};
            if (transfer.kind == MemArrayModel::MemTransfer::Kind::LoadSramToStream) {
                require(transfer.bytes == expected_read_bytes(
                    OperationSpec {cycle - transfer.tile, transfer.mem_slice,
                        transfer.bank, command.instruction}, transfer.tile),
                    "SRAM read sample differs from preload");
                writer.emit(spec, cycle, "LOW", "PRE_POSEDGE",
                    "lpu_mem_sram_leaf", "mem_e_s0_b0",
                    "MEM_SRAM_READ_SAMPLE", hw::kLanesPerTile * 8,
                    pack_bytes(transfer.bytes), "ARCH_EVENT", "SEMANTIC",
                    "EVENT", "SRAM", "MemArrayModel::executed_transfers",
                    command, hex_bytes(transfer.bytes));
                ++read_count;
            } else {
                for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
                    require(mem.sram_lane_byte(transfer.mem_slice, transfer.bank,
                        transfer.tile, transfer.address, lane) == transfer.bytes[lane],
                        "SRAM write was not committed at POSEDGE");
                writer.emit(spec, cycle, "POSEDGE", "POST_POSEDGE",
                    "lpu_mem_sram_leaf", "mem_e_s0_b0",
                    "MEM_SRAM_WRITE_COMMIT", hw::kLanesPerTile * 8,
                    pack_bytes(transfer.bytes), "ARCH_EVENT", "SEMANTIC",
                    "EVENT", "SRAM", "MemArrayModel::posedge",
                    command, hex_bytes(transfer.bytes));
                ++write_count;
            }
        }

        for (const auto& transfer : mem.consumed_transfers()) {
            const auto command = MemArrayModel::InstructionTrace {
                transfer.mem_slice, transfer.bank, transfer.tile,
                MemInstruction::Write(transfer.address, transfer.stream)};
            writer.emit(spec, cycle, "LOW", "PRE_POSEDGE",
                "lpu_mem_stream_input", "mem_e_s0_b0",
                "MEM_STREAM_CONSUME", hw::kLanesPerTile * 8,
                pack_bytes(transfer.bytes), "ARCH_EVENT", "SEMANTIC",
                "EVENT", "STREAM", "MemArrayModel::consumed_transfers",
                command, hex_bytes(transfer.bytes));
        }

        for (const auto& response : mem.read_responses()) {
            const auto command = MemArrayModel::InstructionTrace {
                response.mem_slice, response.bank, response.tile,
                MemInstruction::Read(response.address, response.stream)};
            writer.emit(spec, cycle, "POSEDGE", "POST_POSEDGE",
                "lpu_mem_sram_leaf", "mem_e_s0_b0",
                "MEM_READ_RESPONSE", hw::kLanesPerTile * 8,
                pack_bytes(response.bytes), "ARCH_EVENT", "SEMANTIC",
                "EVENT", "SRAM", "MemArrayModel::read_responses",
                command, hex_bytes(response.bytes));
            ++response_count;
        }

        for (const auto& response : mem.staged_read_responses()) {
            const auto command = MemArrayModel::InstructionTrace {
                response.mem_slice, response.bank, response.tile,
                MemInstruction::Read(response.address, response.stream)};
            writer.emit(spec, cycle, "POSEDGE", "POST_POSEDGE",
                "lpu_stream_register_fabric", "east_srf",
                "MEM_SRF_INJECT_COMMIT", hw::kLanesPerTile * 8,
                pack_bytes(response.bytes), "ARCH_EVENT", "SEMANTIC",
                "EVENT", "STREAM", "MemArrayModel::staged_read_responses",
                command, hex_bytes(response.bytes));
            ++inject_count;
        }

        const bool busy = !mem.idle();
        writer.emit(spec, cycle, "POSEDGE", "POST_COMB",
            "cmodel_mem_diagnostic", "mem_e_s0_b0", "mem_busy_derived",
            1, busy ? 1 : 0, "STATE_SAMPLE", "DERIVED", "DIAGNOSTIC",
            "MEM", "!MemArrayModel::idle");

        if (issue_count == spec.operations.size() && !busy) {
            writer.emit(spec, cycle, "POSEDGE", "POST_COMB",
                "architecture", "architecture", "MEM_DRAINED", 1, 1,
                "ARCH_EVENT", "SEMANTIC", "EVENT", "MEM",
                "MemArrayModel::idle");
            drained = true;
            break;
        }
    }

    const auto reads = std::count_if(spec.operations.begin(), spec.operations.end(),
        [](const auto& op) { return op.instruction.opcode == MemOpcode::Read; });
    const auto writes = spec.operations.size() - static_cast<std::size_t>(reads);
    require(drained, "MEM case did not drain");
    require(command_count == spec.operations.size() * hw::kTileRows,
        "wrong tile-command count");
    require(read_count == static_cast<std::size_t>(reads) * hw::kTileRows,
        "wrong SRAM read-sample count");
    require(write_count == writes * hw::kTileRows,
        "wrong SRAM write-commit count");
    require(response_count == static_cast<std::size_t>(reads) * hw::kTileRows,
        "wrong registered read-response count");
    require(inject_count == static_cast<std::size_t>(reads) * hw::kTileRows,
        "wrong SRF injection count");
}

} // namespace

int main(int argc, char** argv)
try {
    const auto trace_path = argc > 1
        ? std::filesystem::path(argv[1])
        : std::filesystem::path("cmodel_trace.jsonl");
    const auto manifest_path = argc > 2
        ? std::filesystem::path(argv[2])
        : trace_path.parent_path() / "shared_input_manifest.json";
    const auto all_cases = cases();
    write_manifest(manifest_path, all_cases);
    TraceWriter writer(trace_path);
    for (std::size_t index = 0; index < all_cases.size(); ++index)
        run_case(writer, all_cases[index], index);
    std::cout << "RAW_MACRO_LEVEL2_CMODEL PASS cases=" << all_cases.size()
              << " trace=" << trace_path.string()
              << " manifest=" << manifest_path.string() << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "RAW_MACRO_LEVEL2_CMODEL FAIL: " << error.what() << '\n';
    return 1;
}
