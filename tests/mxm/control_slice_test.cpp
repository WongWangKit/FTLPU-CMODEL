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
    const auto base = static_cast<std::uint8_t>(
        (tile * ftlpu::hw::kMxmSupercellsPerPlane + supercell_column) & 0xff);
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
    const auto base = static_cast<std::uint8_t>(
        (tile * ftlpu::hw::kMxmSupercellsPerPlane + supercell_column) & 0xff);
    return static_cast<std::int8_t>(base + lane + stream);
}

} // namespace

int main()
{
    auto array = std::make_unique<ftlpu::MxmArray>();
    ftlpu::MxmControlSlice control(*array);
    std::ostringstream log;
    constexpr std::size_t kBuffer = 1;

    for (std::size_t column = 0; column < ftlpu::hw::kMxmSupercellsPerPlane; ++column) {
        control.issue_south(ftlpu::MxmControlInstruction::IW(kBuffer));
    }

    auto provider = [&control](std::size_t tile) {
        const auto token = control.cycle() - tile;
        const auto target_column = ftlpu::hw::kMxmSupercellsPerPlane - 1 - token;
        return row_input(tile, target_column);
    };

    for (std::size_t cycle = 0; cycle < 2 * ftlpu::hw::kMxmSupercellsPerPlane - 1; ++cycle) {
        control.tick(log, provider);
    }

    for (std::size_t tile = 0; tile < ftlpu::hw::kMxmSupercellsPerPlane; ++tile) {
        for (std::size_t column = 0; column < ftlpu::hw::kMxmSupercellsPerPlane; ++column) {
            if (!require(array->buffered_weight(kBuffer, tile, column, 15, 15) == expected_weight(tile, column, 15, 15), "buffered weight mismatch")) {
                return 1;
            }
            if (!require(array->weight(0, tile, column, 15, 15) == 0, "IW should not modify the other buffer")) {
                return 1;
            }
            if (!require(control.loaded_cell(kBuffer, tile, column), "IW should set loaded-cell marker for selected buffer")) {
                return 1;
            }
        }
    }

    auto parallel_array = std::make_unique<ftlpu::MxmArray>();
    ftlpu::MxmControlSlice parallel_control(*parallel_array);
    constexpr std::size_t kActivationStream = 7;
    constexpr std::size_t kOutputStream = 36;
    parallel_control.issue_south(ftlpu::MxmControlInstruction::IW(kBuffer));
    parallel_control.issue_south(ftlpu::MxmControlInstruction::Compute(kBuffer, kActivationStream, kOutputStream));
    parallel_control.set_weight_input(0, row_input(0, 0));
    parallel_control.tick(log);
    if (!require(parallel_control.compute_active(0), "Compute should issue in parallel with IW")) {
        return 1;
    }
    if (!require(parallel_control.compute_weight_buffer(0).value_or(99) == kBuffer, "Compute should carry selected buffer")) {
        return 1;
    }
    if (!require(
            parallel_control.compute_activation_stream_base(0).value_or(99) == kActivationStream,
            "Compute should carry activation stream base")) {
        return 1;
    }
    if (!require(
            parallel_control.output_stream_base(0).value_or(99) == kOutputStream,
            "Compute should carry output stream base")) {
        return 1;
    }
    if (!require(
            parallel_array->buffered_weight(kBuffer, 0, 0, 15, 15) == expected_weight(0, 0, 15, 15),
            "parallel IW should fill weight buffer")) {
        return 1;
    }

    for (std::size_t tile = 0; tile < ftlpu::hw::kMxmSupercellsPerPlane; ++tile) {
        for (std::size_t column = 0; column < ftlpu::hw::kMxmSupercellsPerPlane; ++column) {
            if (!require(array->weight(kBuffer, tile, column, 0, 0) == expected_weight(tile, column, 0, 0), "weight(0,0) mismatch")) {
                return 1;
            }
            if (!require(array->weight(kBuffer, tile, column, 7, 9) == expected_weight(tile, column, 7, 9), "weight(7,9) mismatch")) {
                return 1;
            }
            if (!require(array->weight(kBuffer, tile, column, 15, 15) == expected_weight(tile, column, 15, 15), "weight(15,15) mismatch")) {
                return 1;
            }
        }
    }

    bool caught = false;
    try {
        control.issue_south(ftlpu::MxmControlInstruction::IW(20));
    } catch (const std::out_of_range&) {
        caught = true;
    }
    if (!require(caught, "expected bad weight buffer to throw")) {
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
    if (!require(text.find("tile 0 IW b1 inject") != std::string::npos, "missing tile 0 IW log")) {
        return 1;
    }
    if (!require(text.find("tile 3 IW b1 inject") != std::string::npos, "missing last-tile IW log")) {
        return 1;
    }
    if (!require(text.find("IW buffer1=0x") != std::string::npos, "missing IW buffer log")) {
        return 1;
    }
    if (!require(text.find("load_matrix:") != std::string::npos, "missing load matrix log")) {
        return 1;
    }
    if (!require(text.find("row 0: LLLL") != std::string::npos, "missing row 0 load marker")) {
        return 1;
    }
    if (!require(text.find("row 3: LLLL") != std::string::npos, "missing last-row load marker")) {
        return 1;
    }
    if (!require(text.find("row 1: LLLL") != std::string::npos, "missing row 1 load marker")) {
        return 1;
    }

    return 0;
}
