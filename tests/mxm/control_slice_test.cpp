#include "ftlpu/mxm/control_slice.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

bool require(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << message << '\n';
        return false;
    }
    return true;
}

ftlpu::MxmControlSlice::WeightInput row_input(std::size_t tile, std::size_t supercell_column)
{
    ftlpu::MxmControlSlice::WeightInput input{};
    const auto base = static_cast<std::uint8_t>((tile * 20 + supercell_column) & 0xff);
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        for (std::size_t stream = 0; stream < ftlpu::hw::kMxmLoadStreamsPerCycle; ++stream) {
            input[lane][stream] = ftlpu::MxmArray::Supercell::InputWord {
                static_cast<std::int8_t>(base + lane + stream),
                stream + 1 == ftlpu::hw::kMxmLoadStreamsPerCycle,
            };
        }
    }
    return input;
}

std::int8_t expected_weight(std::size_t tile, std::size_t supercell_column, std::size_t lane, std::size_t stream)
{
    const auto base = static_cast<std::uint8_t>((tile * 20 + supercell_column) & 0xff);
    return static_cast<std::int8_t>(base + lane + stream);
}

} // namespace

int main()
{
    auto array = std::make_unique<ftlpu::MxmArray>();
    ftlpu::MxmControlSlice control(*array);
    std::ostringstream log;
    constexpr std::size_t kColumn = 7;

    control.issue_south(ftlpu::MxmControlInstruction::IW(kColumn));

    for (std::size_t cycle = 0; cycle < ftlpu::hw::kMxmSupercellsPerPlane; ++cycle) {
        control.set_weight_input(cycle, row_input(cycle, kColumn));
        control.tick(log);
    }

    for (std::size_t tile = 0; tile < ftlpu::hw::kMxmSupercellsPerPlane; ++tile) {
        if (!require(array->weight(tile, kColumn, 0, 0) == expected_weight(tile, kColumn, 0, 0), "weight(0,0) mismatch")) {
            return 1;
        }
        if (!require(array->weight(tile, kColumn, 7, 9) == expected_weight(tile, kColumn, 7, 9), "weight(7,9) mismatch")) {
            return 1;
        }
        if (!require(array->weight(tile, kColumn, 15, 15) == expected_weight(tile, kColumn, 15, 15), "weight(15,15) mismatch")) {
            return 1;
        }
    }

    bool caught = false;
    try {
        control.issue_south(ftlpu::MxmControlInstruction::IW(20));
    } catch (const std::out_of_range&) {
        caught = true;
    }
    if (!require(caught, "expected bad column to throw")) {
        return 1;
    }

    auto empty_array = std::make_unique<ftlpu::MxmArray>();
    ftlpu::MxmControlSlice missing_input_control(*empty_array);
    missing_input_control.issue_south(ftlpu::MxmControlInstruction::IW(0));
    caught = false;
    try {
        missing_input_control.tick(log);
    } catch (const std::logic_error&) {
        caught = true;
    }
    if (!require(caught, "expected missing local input to throw")) {
        return 1;
    }

    const auto text = log.str();
    if (!require(text.find("mxm_control cycle 0") != std::string::npos, "missing cycle 0 log")) {
        return 1;
    }
    if (!require(text.find("tile 0 IW col=7") != std::string::npos, "missing tile 0 IW log")) {
        return 1;
    }
    if (!require(text.find("tile 19 IW col=7") != std::string::npos, "missing tile 19 IW log")) {
        return 1;
    }
    if (!require(text.find("IW load matrix=0x") != std::string::npos, "missing IW load log")) {
        return 1;
    }
    if (!require(text.find("load_matrix:") != std::string::npos, "missing 20x20 load matrix log")) {
        return 1;
    }
    if (!require(text.find("row 0: .......L............") != std::string::npos, "missing row 0 load marker")) {
        return 1;
    }
    if (!require(text.find("row 19: .......L............") != std::string::npos, "missing row 19 load marker")) {
        return 1;
    }
    if (!require(text.find("row 1: ....................") != std::string::npos, "missing empty row marker")) {
        return 1;
    }

    return 0;
}
