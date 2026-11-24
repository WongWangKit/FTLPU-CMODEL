#include "ftlpu/icu/instruction_dma_protocol_core.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ftlpu;

struct Stimulus {
    std::size_t cycle{0};
    unsigned scenario{0};
    InstructionDmaProtocolInputs inputs{};
    std::string response_hex{};
};

std::vector<std::string> split(const std::string& line)
{
    std::vector<std::string> fields;
    std::size_t begin = 0;
    while (begin <= line.size()) {
        const auto comma = line.find(',', begin);
        fields.push_back(line.substr(begin,
            comma == std::string::npos ? std::string::npos : comma - begin));
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return fields;
}

unsigned hex_digit(char value)
{
    value = static_cast<char>(std::tolower(static_cast<unsigned char>(value)));
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    throw std::runtime_error("invalid hexadecimal digit in DMA stimulus");
}

InstructionDmaResponseBeat parse_response(const std::vector<std::string>& f)
{
    InstructionDmaResponseBeat response{};
    auto hex = f[9];
    if (hex.size() > 64)
        throw std::runtime_error("DMA response data exceeds 256 bits");
    hex.insert(hex.begin(), 64 - hex.size(), '0');
    for (std::size_t lane = 0; lane < 32; ++lane) {
        const auto position = 62 - lane * 2;
        response.data[lane] = static_cast<std::uint8_t>(
            (hex_digit(hex[position]) << 4) | hex_digit(hex[position + 1]));
    }
    response.keep = static_cast<std::uint32_t>(std::stoul(f[10], nullptr, 16));
    response.last = std::stoul(f[11]) != 0;
    response.id = static_cast<std::uint8_t>(std::stoul(f[12]));
    response.error = std::stoul(f[13]) != 0;
    return response;
}

Stimulus parse_stimulus(const std::string& line)
{
    const auto f = split(line);
    if (f.size() != 15)
        throw std::runtime_error("DMA stimulus row does not have 15 fields");
    Stimulus stimulus{};
    stimulus.cycle = std::stoull(f[0]);
    stimulus.scenario = static_cast<unsigned>(std::stoul(f[1]));
    auto& input = stimulus.inputs;
    input.rst_ni = std::stoul(f[2]) != 0;
    input.ctrl_dma_cmd_valid_i = std::stoul(f[3]) != 0;
    input.ctrl_dma_cmd_address_i = std::stoull(f[4]);
    input.ctrl_dma_cmd_length_i =
        static_cast<std::uint32_t>(std::stoul(f[5]));
    input.ctrl_dma_cmd_id_i = static_cast<std::uint8_t>(std::stoul(f[6]));
    input.ddr_read_req_ready_i = std::stoul(f[7]) != 0;
    input.ddr_read_rsp_valid_i = std::stoul(f[8]) != 0;
    stimulus.response_hex = f[9];
    const auto response = parse_response(f);
    input.ddr_read_rsp_data_i = response.data;
    input.ddr_read_rsp_keep_i = response.keep;
    input.ddr_read_rsp_last_i = response.last;
    input.ddr_read_rsp_id_i = response.id;
    input.ddr_read_rsp_error_i = response.error;
    input.reasm_rsp_ready_i = std::stoul(f[14]) != 0;
    return stimulus;
}

std::string response_hex(const InstructionDmaResponseBeat& response)
{
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (std::size_t lane = 32; lane != 0; --lane)
        output << std::setw(2) << static_cast<unsigned>(response.data[lane - 1]);
    return output.str();
}

std::string normalized_hex(std::string value, std::size_t digits)
{
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value.size() < digits) value.insert(value.begin(), digits - value.size(), '0');
    return value;
}

} // namespace

int main(int argc, char** argv) try
{
    if (argc != 3) {
        std::cerr << "usage: instruction_dma_golden_trace <stimulus.csv> <output.csv>\n";
        return 2;
    }

    std::ifstream stimulusFile(argv[1]);
    std::ofstream output(argv[2]);
    if (!stimulusFile || !output)
        throw std::runtime_error("cannot open DMA stimulus or trace file");

    std::string line;
    std::getline(stimulusFile, line); // Header.
    // Preserve the established cross-model CSV schema while sampling the
    // canonical RTL-named C++ interface below.  This keeps the existing
    // 108-cycle RTL trace directly comparable without reintroducing legacy
    // member names into InstructionDmaProtocolCore.
    output << "cycle,scenario,reset_n,cmd_valid,cmd_address,cmd_length,cmd_id,"
              "req_ready,rsp_valid,rsp_data,rsp_keep,rsp_last,rsp_id,rsp_error,"
              "reasm_ready,cmd_ready_pre,req_valid_pre,req_address_pre,"
              "req_length_pre,req_id_pre,cmd_fire,outstanding_enq_fire,"
              "outstanding_deq_fire,rsp_ready_pre,rsp_input_fire,resp_valid_pre,"
              "resp_data_pre,resp_keep_pre,resp_last_pre,resp_id_pre,"
              "resp_error_pre,reasm_fire,outstanding_pre,rx_pre,"
              "fault_comb_pre,fault_code_comb_pre,outstanding_post,rx_post,"
              "fault_state_post,fault_code_post,idle_post,fault_wire_post_nba\n";

    InstructionDmaProtocolCore core;
    std::size_t rowCount = 0;
    while (std::getline(stimulusFile, line)) {
        if (line.empty()) continue;
        const auto stimulus = parse_stimulus(line);
        const auto& input = stimulus.inputs;
        core.drive(input);

        const auto outstandingPre = core.outstanding();
        const auto rxPre = core.response_fifo_occupancy();
        const auto commandFire = core.ctrl_dma_cmd_fire();
        const auto responseInputFire = core.ddr_read_rsp_fire();
        const auto outstandingDequeue = responseInputFire
            && input.ddr_read_rsp_last_i && outstandingPre != 0;
        const auto responseValid = core.reasm_rsp_valid_o();
        InstructionDmaResponseBeat response{};
        if (responseValid) response = core.reasm_rsp_o();

        output << stimulus.cycle << ',' << stimulus.scenario << ','
               << input.rst_ni << ',' << input.ctrl_dma_cmd_valid_i << ','
               << input.ctrl_dma_cmd_address_i << ','
               << input.ctrl_dma_cmd_length_i << ','
               << static_cast<unsigned>(input.ctrl_dma_cmd_id_i) << ','
               << input.ddr_read_req_ready_i << ','
               << input.ddr_read_rsp_valid_i << ','
               << normalized_hex(stimulus.response_hex, 64) << ','
               << normalized_hex(line.empty() ? "0" : split(line)[10], 8) << ','
               << input.ddr_read_rsp_last_i << ','
               << static_cast<unsigned>(input.ddr_read_rsp_id_i) << ','
               << input.ddr_read_rsp_error_i << ','
               << input.reasm_rsp_ready_i << ','
               << core.ctrl_dma_cmd_ready_o() << ','
               << core.ddr_read_req_valid_o() << ','
               << core.ddr_read_req_address_o() << ','
               << core.ddr_read_req_length_o() << ','
               << static_cast<unsigned>(core.ddr_read_req_id_o()) << ','
               << commandFire << ',' << commandFire << ','
               << outstandingDequeue << ',' << core.ddr_read_rsp_ready_o() << ','
               << responseInputFire << ',' << responseValid << ','
               << (responseValid ? response_hex(response) : std::string(64, '0'))
               << ',' << std::hex << std::setw(8) << std::setfill('0')
               << (responseValid ? response.keep : 0U) << std::dec << ','
               << (responseValid && response.last) << ','
               << (responseValid ? static_cast<unsigned>(response.id) : 0U) << ','
               << (responseValid && response.error) << ',' << core.reasm_rsp_fire()
               << ',' << outstandingPre << ',' << rxPre << ',' << core.fault()
               << ',' << static_cast<unsigned>(core.fault_code());

        core.low();
        core.validate_posedge();
        core.posedge();
        core.high();

        output << ',' << core.outstanding() << ','
               << core.response_fifo_occupancy() << ',' << core.fault() << ','
               << static_cast<unsigned>(core.fault_code()) << ',' << core.idle()
               << ',' << core.fault() << '\n';
        ++rowCount;
    }

    std::cout << "wrote " << rowCount
              << " CModel instruction DMA trace rows\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "instruction_dma_golden_trace failed: " << error.what() << '\n';
    return 1;
}
