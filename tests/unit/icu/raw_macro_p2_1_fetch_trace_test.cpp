#include "ftlpu/icu/mem_local_frontend_fetch.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using Model = ftlpu::MemLocalFrontendFetchModel<16, 4, 4>;
using RawWord = Model::RawWord;

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
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
    std::string result(width, '0');
    for (std::size_t bit = 0; bit < width; ++bit)
        result[width - 1 - bit] = ((value >> bit) & 1u) ? '1' : '0';
    return result;
}

std::string binary(const RawWord& word)
{
    std::string result(96, '0');
    for (std::size_t bit = 0; bit < 96; ++bit)
        result[95 - bit] = ((word.lanes[bit / 32] >> (bit % 32)) & 1u)
            ? '1' : '0';
    return result;
}

class TraceWriter {
public:
    explicit TraceWriter(const std::filesystem::path& path)
    {
        if (path.has_parent_path())
            std::filesystem::create_directories(path.parent_path());
        output_.open(path, std::ios::out | std::ios::trunc);
        if (!output_) throw std::runtime_error("cannot create P2.1 trace");
    }

    void emit(std::string_view case_id, std::size_t cycle,
        std::string_view phase, std::string_view sample_point,
        std::string_view module, std::string_view signal, std::size_t width,
        std::string value, std::string_view event_type,
        std::string_view mapping_status, std::string_view source_ref)
    {
        if (value.size() != width
            || value.find_first_not_of("01") != std::string::npos)
            throw std::logic_error("P2.1 trace value is not exact binary");
        output_
            << "{\"schema\":\"ftlpu.raw_macro.p2_1.trace.v1\""
            << ",\"run_id\":\"cmodel_raw_macro_p2_1_v1\""
            << ",\"case_id\":\"" << json_escape(case_id) << "\""
            << ",\"source\":\"CMODEL\""
            << ",\"global_cycle\":" << cycle
            << ",\"logical_cycle\":-1"
            << ",\"transport_tick\":" << cycle
            << ",\"phase\":\"" << phase << "\""
            << ",\"sample_point\":\"" << sample_point << "\""
            << ",\"event_sequence\":" << event_sequence_++
            << ",\"module\":\"" << module << "\""
            << ",\"instance\":\"mem_queue_0\""
            << ",\"signal\":\"" << signal << "\""
            << ",\"width\":" << width
            << ",\"value\":\"" << value << "\""
            << ",\"event_type\":\"" << event_type << "\""
            << ",\"mapping_status\":\"" << mapping_status << "\""
            << ",\"source_ref\":\"" << json_escape(source_ref)
            << "\"}\n";
    }

private:
    std::ofstream output_{};
    std::size_t event_sequence_{0};
};

struct CaseSpec {
    std::string id;
    std::size_t base{};
    std::size_t count{};
    std::string profile;
    std::string scope;
};

std::vector<std::string> split_csv(const std::string& line)
{
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) fields.push_back(field);
    return fields;
}

std::vector<CaseSpec> read_cases(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read P2.1 case matrix");
    std::vector<CaseSpec> result;
    std::string line;
    require(static_cast<bool>(std::getline(input, line)), "empty case matrix");
    require(line == "case_id,base,word_count,profile,scope",
        "unexpected P2.1 case matrix header");
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split_csv(line);
        require(fields.size() == 5, "bad P2.1 case matrix row");
        result.push_back(CaseSpec{fields[0],
            static_cast<std::size_t>(std::stoull(fields[1])),
            static_cast<std::size_t>(std::stoull(fields[2])),
            fields[3], fields[4]});
    }
    return result;
}

unsigned hex_nibble(char value)
{
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    throw std::runtime_error("invalid hex digit in P2.1 word fixture");
}

std::vector<RawWord> read_words(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot read P2.1 machine words");
    std::vector<RawWord> result;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line.front() == '#') continue;
        std::stringstream stream(line);
        std::size_t index = 0;
        std::string hex;
        stream >> index >> hex;
        require(index == result.size(), "non-contiguous machine word index");
        require(hex.size() == 24, "P2.1 machine word is not 12 bytes");
        RawWord word{};
        for (std::size_t byte = 0; byte < 12; ++byte) {
            const auto value = (hex_nibble(hex[byte * 2]) << 4)
                | hex_nibble(hex[byte * 2 + 1]);
            word.lanes[byte / 4]
                |= static_cast<std::uint32_t>(value) << ((byte % 4) * 8);
        }
        result.push_back(word);
    }
    require(result.size() >= 12, "P2.1 machine word fixture is too short");
    return result;
}

struct CycleRecord {
    std::size_t cycle{};
    Model::Ports ports{};
    std::size_t outstanding{};
    std::size_t response_fifo_level{};
    std::size_t word_fifo_level{};
    bool busy{};
    bool done{};
    bool fault{};
    Model::FaultCode fault_code{Model::FaultCode::None};
};

struct CaseResult {
    std::vector<CycleRecord> cycles;
    std::vector<std::size_t> request_addresses;
    std::vector<bool> request_last;
    std::vector<std::size_t> request_cycles;
    std::vector<RawWord> response_words;
    std::vector<bool> response_last;
    std::vector<std::size_t> response_cycles;
    std::vector<RawWord> decoder_words;
    std::vector<std::size_t> decoder_cycles;
    std::size_t max_outstanding{};
    bool reset_or_clear_seen{false};
};

void emit_pre(TraceWriter& writer, const CaseSpec& spec,
    std::size_t cycle, const Model::Ports& p)
{
    const auto emit = [&](std::string_view module, std::string_view signal,
                          std::size_t width, std::string value,
                          std::string_view ref) {
        writer.emit(spec.id, cycle, "LOW", "PRE_POSEDGE", module, signal,
            width, std::move(value), "SIGNAL_SAMPLE", "DIRECT", ref);
    };
    emit("lpu_mem_imem_fetch", "start_valid_i", 1,
        binary(p.start_valid_i, 1), "MemLocalFrontendFetchModel::Ports");
    emit("lpu_mem_imem_fetch", "start_ready_o", 1,
        binary(p.start_ready_o, 1), "MemLocalFrontendFetchModel::Ports");
    emit("lpu_mem_imem_fetch", "start_base_address_i", 4,
        binary(p.start_base_address_i, 4), "MemLocalFrontendFetchModel::Ports");
    emit("lpu_mem_imem_fetch", "start_word_count_i", 5,
        binary(p.start_word_count_i, 5), "MemLocalFrontendFetchModel::Ports");
    emit("lpu_mem_imem_fetch", "start_fire", 1,
        binary(p.start_fire, 1), "valid && ready");

    emit("lpu_mem_imem_fetch", "imem_read_req_valid_o", 1,
        binary(p.imem_read_req_valid_o, 1), "MemLocalFrontendFetchModel::Ports");
    emit("lpu_mem_imem_fetch", "imem_read_req_ready_i", 1,
        binary(p.imem_read_req_ready_i, 1), "MemLocalFrontendFetchModel::Ports");
    emit("lpu_mem_imem_fetch", "imem_read_req_address_o", 4,
        binary(p.imem_read_req_address_o, 4), "MemLocalFrontendFetchModel::Ports");
    emit("lpu_mem_imem_fetch", "imem_read_req_last_o", 1,
        binary(p.imem_read_req_last_o, 1), "MemLocalFrontendFetchModel::Ports");
    emit("lpu_mem_imem_fetch", "imem_read_req_fire", 1,
        binary(p.imem_read_req_fire, 1), "valid && ready");

    emit("lpu_mem_imem_fetch", "imem_read_rsp_valid_i", 1,
        binary(p.imem_read_rsp_valid_i, 1), "MemLocalFrontendFetchModel::Ports");
    emit("lpu_mem_imem_fetch", "imem_read_rsp_ready_o", 1,
        binary(p.imem_read_rsp_ready_o, 1), "MemLocalFrontendFetchModel::Ports");
    emit("lpu_mem_imem_fetch", "imem_read_rsp_data_i", 96,
        binary(p.imem_read_rsp_data_i), "MemLocalFrontendFetchModel::Ports");
    emit("lpu_mem_imem_fetch", "imem_read_rsp_last_i", 1,
        binary(p.imem_read_rsp_last_i, 1), "MemLocalFrontendFetchModel::Ports");
    emit("lpu_mem_imem_fetch", "imem_read_rsp_fire", 1,
        binary(p.imem_read_rsp_fire, 1), "valid && ready");

    emit("lpu_mem_imem_fetch", "word_valid_o", 1,
        binary(p.word_valid_o, 1), "MemLocalFrontendFetchModel::Ports");
    emit("lpu_mem_imem_fetch", "word_ready_i", 1,
        binary(p.word_ready_i, 1), "MemLocalFrontendFetchModel::Ports");
    emit("lpu_mem_imem_fetch", "word_o", 96,
        binary(p.word_o), "MemLocalFrontendFetchModel::Ports");
    emit("lpu_mem_imem_fetch", "word_last_o", 1,
        binary(p.word_last_o, 1), "MemLocalFrontendFetchModel::Ports");
    emit("lpu_mem_imem_fetch", "word_fire", 1,
        binary(p.word_fire, 1), "valid && ready");

    emit("lpu_mem_macro_decoder", "word_valid_i", 1,
        binary(p.decoder_word_valid_i, 1), "Macro Word FIFO dequeue boundary");
    emit("lpu_mem_macro_decoder", "word_ready_o", 1,
        binary(p.decoder_word_ready_o, 1), "Macro Word FIFO dequeue boundary");
    emit("lpu_mem_macro_decoder", "word_i", 96,
        binary(p.decoder_word_i), "Macro Word FIFO dequeue boundary");
    emit("lpu_mem_macro_decoder", "word_last_i", 1,
        binary(p.decoder_word_last_i, 1), "Macro Word FIFO dequeue boundary");
    emit("lpu_mem_macro_decoder", "decoder_word_fire", 1,
        binary(p.decoder_word_fire, 1), "valid && ready");
}

void emit_post(TraceWriter& writer, const CaseSpec& spec,
    std::size_t cycle, const Model& model)
{
    const auto emit = [&](std::string_view module, std::string_view signal,
                          std::size_t width, std::uint64_t value,
                          std::string_view ref) {
        writer.emit(spec.id, cycle, "POSEDGE", "POST_COMB", module, signal,
            width, binary(value, width), "STATE_SAMPLE", "DIRECT", ref);
    };
    emit("lpu_mem_local_imem", "request_stage_valid_q", 1,
        model.request_stage_valid(), "registered request stage");
    emit("lpu_mem_local_imem", "response_stage_valid_q", 1,
        model.response_stage_valid(), "registered response stage");
    emit("lpu_mem_local_imem", "outstanding_q", 3,
        model.outstanding(), "accepted request minus response fire");
    emit("lpu_mem_local_imem", "response_fifo_level", 3,
        model.response_fifo_level(), "response FIFO occupancy");
    emit("lpu_mem_macro_frontend", "word_fifo_level_o", 3,
        model.word_fifo_level(), "Macro Word FIFO occupancy");
    emit("lpu_mem_imem_fetch", "busy_o", 1,
        model.busy(), "Fetch registered state");
    emit("lpu_mem_imem_fetch", "done_o", 1,
        model.done(), "Fetch one-cycle completion pulse");
    emit("lpu_mem_imem_fetch", "fault_o", 1,
        model.fault(), "Fetch sticky fault");
    emit("lpu_mem_imem_fetch", "fault_code_o", 2,
        static_cast<std::uint8_t>(model.fault_code()), "Fetch sticky fault code");
    emit("lpu_mem_imem_fetch", "next_address_q", 4,
        model.next_address(), "Fetch registered address");
    emit("lpu_mem_imem_fetch", "requests_remaining_q", 5,
        model.requests_remaining(), "Fetch request counter");
    emit("lpu_mem_imem_fetch", "responses_remaining_q", 5,
        model.responses_remaining(), "Fetch response counter");
}

void emit_event(TraceWriter& writer, const CaseSpec& spec,
    std::size_t cycle, std::string_view signal)
{
    writer.emit(spec.id, cycle, "LOW", "PRE_POSEDGE", "architecture",
        signal, 1, "1", "ARCH_EVENT", "SEMANTIC",
        "P2.1 directed-test transaction");
}

CaseResult run_case(const CaseSpec& spec, const std::vector<RawWord>& words,
    TraceWriter& writer)
{
    Model model;
    for (std::size_t index = 0;
         index < spec.count && spec.base + index < 16; ++index)
        model.load_word(spec.base + index, words[index]);

    CaseResult result;
    bool replacement_armed = false;
    bool response_corrupted = false;
    bool done_seen = false;
    std::optional<std::size_t> clear_cycle;
    if (spec.id == "P2.1-F01") {
        writer.emit(spec.id, 0, "LOW", "PRE_POSEDGE",
            "cmodel_production_path", "distributed_queue_fetch_state", 1,
            "0", "STATE_SAMPLE", "UNOBSERVABLE",
            "production MacroBitstreamDecoder still merges Fetch/FIFO state; P2.1 component is not integrated");
        writer.emit(spec.id, 0, "LOW", "PRE_POSEDGE",
            "lpu_mem_macro_scheduler", "mem_instruction_queue", 1,
            "0", "STATE_SAMPLE", "NA",
            "Raw MEM Scheduler drives canonical MEM Issue directly; no decoded Instruction Queue");
    }
    emit_event(writer, spec, 0, "CASE_START");

    for (std::size_t cycle = 0; cycle < 96; ++cycle) {
        Model::Inputs inputs;
        inputs.start_valid_i = cycle == 0;
        inputs.start_base_address_i = spec.base;
        inputs.start_word_count_i = spec.count;

        if (spec.profile == "request_stall") {
            inputs.imem_request_accept_enable_i =
                cycle != 1 && cycle != 2 && cycle != 4;
        } else if (spec.profile == "response_gap") {
            inputs.imem_request_accept_enable_i = cycle != 2 && cycle != 4;
        } else if (spec.profile == "word_backpressure") {
            inputs.word_accept_enable_i = cycle >= 8;
        } else if (spec.profile == "credit_boundary") {
            inputs.word_accept_enable_i = cycle >= 10;
        } else if (spec.profile == "full_replacement") {
            if (!replacement_armed
                && model.word_fifo_level() == 4
                && model.response_fifo_level() != 0)
                replacement_armed = true;
            inputs.decoder_word_ready_i = replacement_armed;
        } else if (spec.profile == "response_protocol_fault") {
            if (!response_corrupted
                && model.response_fifo_level() != 0
                && model.responses_remaining() == 2) {
                inputs.response_last_override_i = true;
                response_corrupted = true;
            }
        } else if (spec.profile == "clear_during_fetch"
                   || spec.profile == "reset_during_fetch") {
            if (!clear_cycle.has_value() && model.outstanding() >= 2)
                clear_cycle = cycle;
            if (spec.profile == "clear_during_fetch")
                inputs.clear_i = clear_cycle == cycle;
            else
                inputs.rst_ni = clear_cycle != cycle;
        }

        model.cycle(inputs);
        const auto& ports = model.ports();
        emit_pre(writer, spec, cycle, ports);
        emit_post(writer, spec, cycle, model);
        if (ports.start_fire) emit_event(writer, spec, cycle, "START_FIRE");
        if (ports.imem_read_req_fire)
            emit_event(writer, spec, cycle, "REQUEST_FIRE");
        if (ports.imem_read_rsp_fire)
            emit_event(writer, spec, cycle, "RESPONSE_FIRE");
        if (ports.word_fire) emit_event(writer, spec, cycle, "WORD_FIRE");
        if (ports.decoder_word_fire)
            emit_event(writer, spec, cycle, "DECODER_WORD_FIRE");
        if (model.fault()) emit_event(writer, spec, cycle, "FAULT");

        result.cycles.push_back(CycleRecord{cycle, ports,
            model.outstanding(), model.response_fifo_level(),
            model.word_fifo_level(), model.busy(), model.done(),
            model.fault(), model.fault_code()});
        done_seen = done_seen || model.done();
        result.max_outstanding = std::max(
            result.max_outstanding, model.outstanding());
        if (ports.imem_read_req_fire) {
            result.request_addresses.push_back(ports.imem_read_req_address_o);
            result.request_last.push_back(ports.imem_read_req_last_o);
            result.request_cycles.push_back(cycle);
        }
        if (ports.imem_read_rsp_fire) {
            result.response_words.push_back(ports.imem_read_rsp_data_i);
            result.response_last.push_back(ports.imem_read_rsp_last_i);
            result.response_cycles.push_back(cycle);
        }
        if (ports.decoder_word_fire) {
            result.decoder_words.push_back(ports.decoder_word_i);
            result.decoder_cycles.push_back(cycle);
        }
        if (inputs.clear_i || !inputs.rst_ni) {
            result.reset_or_clear_seen = true;
            require(!model.busy() && !model.fault()
                    && model.outstanding() == 0
                    && model.response_fifo_level() == 0
                    && model.word_fifo_level() == 0,
                spec.id + ": clear did not discard valid frontend state");
        }

        const bool fault_case = spec.profile == "descriptor_fault"
            || spec.profile == "response_protocol_fault";
        if (fault_case && model.fault()) break;
        if ((spec.profile == "clear_during_fetch"
                || spec.profile == "reset_during_fetch")
            && clear_cycle.has_value()
            && cycle >= *clear_cycle + 3)
            break;
        if (done_seen && model.outstanding() == 0
            && model.response_fifo_level() == 0
            && model.word_fifo_level() == 0)
            break;
    }
    emit_event(writer, spec,
        result.cycles.empty() ? 0 : result.cycles.back().cycle, "CASE_END");
    return result;
}

void check_success_order(const CaseSpec& spec, const CaseResult& result,
    const std::vector<RawWord>& words)
{
    require(result.request_addresses.size() == spec.count,
        spec.id + ": wrong request count");
    require(result.response_words.size() == spec.count,
        spec.id + ": wrong response count");
    require(result.decoder_words.size() == spec.count,
        spec.id + ": wrong Decoder word count");
    for (std::size_t index = 0; index < spec.count; ++index) {
        require(result.request_addresses[index] == spec.base + index,
            spec.id + ": request address order mismatch");
        require(result.request_last[index] == (index + 1 == spec.count),
            spec.id + ": request Last mismatch");
        require(result.response_words[index].lanes == words[index].lanes,
            spec.id + ": response payload order mismatch");
        require(result.response_last[index] == (index + 1 == spec.count),
            spec.id + ": response Last mismatch");
        require(result.decoder_words[index].lanes == words[index].lanes,
            spec.id + ": Decoder payload order mismatch");
    }
    require(!result.cycles.empty() && !result.cycles.back().fault,
        spec.id + ": unexpected fetch fault");
}

void verify_case(const CaseSpec& spec, const CaseResult& result,
    const std::vector<RawWord>& words)
{
    if (spec.profile == "descriptor_fault") {
        require(!result.cycles.empty() && result.cycles.back().fault,
            spec.id + ": missing descriptor fault");
        require(result.cycles.back().fault_code == Model::FaultCode::Descriptor,
            spec.id + ": wrong descriptor fault code");
        require(result.request_addresses.empty(),
            spec.id + ": illegal descriptor issued a request");
        return;
    }
    if (spec.profile == "response_protocol_fault") {
        require(!result.cycles.empty() && result.cycles.back().fault,
            spec.id + ": missing response protocol fault");
        require(result.cycles.back().fault_code
                == Model::FaultCode::ResponseProtocol,
            spec.id + ": wrong response protocol fault code");
        require(!result.cycles.back().busy,
            spec.id + ": accepted response Last did not clear Fetch Busy");
        require(result.cycles.back().done,
            spec.id + ": accepted response Last did not pulse Fetch Done");
        return;
    }
    if (spec.profile == "clear_during_fetch"
        || spec.profile == "reset_during_fetch") {
        require(result.reset_or_clear_seen,
            spec.id + ": reset/clear was never applied");
        const auto after_clear = std::find_if(result.cycles.begin(),
            result.cycles.end(), [](const auto& row) {
                return !row.busy && row.outstanding == 0
                    && row.response_fifo_level == 0
                    && row.word_fifo_level == 0;
            });
        require(after_clear != result.cycles.end(),
            spec.id + ": no cleared state was observed");
        return;
    }

    check_success_order(spec, result, words);
    if (spec.profile == "always_ready" && spec.count == 4) {
        for (std::size_t index = 1; index < result.request_cycles.size(); ++index)
            require(result.request_cycles[index]
                    == result.request_cycles[index - 1] + 1,
                spec.id + ": request II is not one");
    }
    if (spec.profile == "request_stall") {
        bool held = false;
        for (std::size_t index = 1; index < result.cycles.size(); ++index) {
            const auto& before = result.cycles[index - 1].ports;
            const auto& now = result.cycles[index].ports;
            if (before.imem_read_req_valid_o
                && !before.imem_read_req_ready_i
                && now.imem_read_req_valid_o
                && !now.imem_read_req_ready_i) {
                require(before.imem_read_req_address_o
                        == now.imem_read_req_address_o
                        && before.imem_read_req_last_o
                            == now.imem_read_req_last_o,
                    spec.id + ": request changed while stalled");
                held = true;
            }
        }
        require(held, spec.id + ": request stall was not observed");
    }
    if (spec.profile == "response_gap") {
        bool gap = false;
        for (std::size_t index = 1; index < result.response_cycles.size(); ++index)
            gap = gap || result.response_cycles[index]
                > result.response_cycles[index - 1] + 1;
        require(gap, spec.id + ": response gap was not observed");
    }
    if (spec.profile == "word_backpressure") {
        bool held = false;
        for (std::size_t index = 1; index < result.cycles.size(); ++index) {
            const auto& before = result.cycles[index - 1].ports;
            const auto& now = result.cycles[index].ports;
            if (before.word_valid_o && !before.word_ready_i
                && now.word_valid_o && !now.word_ready_i) {
                require(before.word_o.lanes == now.word_o.lanes
                        && before.word_last_o == now.word_last_o,
                    spec.id + ": response payload changed under backpressure");
                held = true;
            }
        }
        require(held, spec.id + ": word backpressure was not observed");
    }
    if (spec.profile == "credit_boundary") {
        require(result.max_outstanding == 4,
            spec.id + ": response credit depth was not reached");
        const bool blocked = std::any_of(result.cycles.begin(),
            result.cycles.end(), [](const auto& row) {
                return row.ports.imem_read_req_valid_o
                    && !row.ports.imem_read_req_ready_i
                    && row.outstanding == 4;
            });
        require(blocked, spec.id + ": fifth request was not blocked");
    }
    if (spec.profile == "full_replacement") {
        const bool replacement = std::any_of(result.cycles.begin(),
            result.cycles.end(), [](const auto& row) {
                return row.ports.decoder_word_fire
                    && row.ports.word_fire
                    && row.word_fifo_level == 4;
            });
        require(replacement,
            spec.id + ": full Word FIFO replacement was not observed");
    }
}

} // namespace

int main(int argc, char** argv)
try {
    require(argc == 4,
        "usage: raw_macro_p2_1_fetch_trace_test <words> <cases> <trace>");
    const auto words = read_words(argv[1]);
    const auto cases = read_cases(argv[2]);
    TraceWriter writer(argv[3]);

    std::size_t executed = 0;
    std::size_t retained = 0;
    for (const auto& spec : cases) {
        if (spec.scope == "full_path_retained") {
            ++retained;
            continue;
        }
        const auto result = run_case(spec, words, writer);
        verify_case(spec, result, words);
        ++executed;
    }
    require(executed == 13, "P2.1 did not execute all F01-F11 variants");
    require(retained == 1, "P2.1 F12 retention record is missing");
    std::cout << "raw_macro_p2_1_fetch_trace_test passed: executed="
              << executed << " retained=" << retained << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "raw_macro_p2_1_fetch_trace_test failed: "
              << error.what() << '\n';
    return 1;
}
