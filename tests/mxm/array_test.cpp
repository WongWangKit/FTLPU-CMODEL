#include "ftlpu/mxm/array.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

ftlpu::MxmArray::InputVector cell_input(std::size_t supercell_row, std::size_t supercell_column)
{
    ftlpu::MxmArray::InputVector input{};
    const auto base = static_cast<std::uint8_t>(
        (supercell_row * ftlpu::hw::kMxmSupercellsPerPlane + supercell_column) & 0xff);
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

std::int8_t expected_weight(
    std::size_t supercell_row,
    std::size_t supercell_column,
    std::size_t lane,
    std::size_t stream)
{
    const auto base = static_cast<std::uint8_t>(
        (supercell_row * ftlpu::hw::kMxmSupercellsPerPlane + supercell_column) & 0xff);
    return static_cast<std::int8_t>(base + lane + stream);
}

} // namespace

int main()
{
    auto array = std::make_unique<ftlpu::MxmArray>();
    std::ostringstream log;

    for (std::size_t row = 0; row < ftlpu::hw::kMxmSupercellsPerPlane; ++row) {
        for (std::size_t column = 0; column < ftlpu::hw::kMxmSupercellsPerPlane; ++column) {
            array->load_weights(row, column, cell_input(row, column), log);
        }
    }

    for (std::size_t row = 0; row < ftlpu::hw::kMxmSupercellsPerPlane; ++row) {
        for (std::size_t column = 0; column < ftlpu::hw::kMxmSupercellsPerPlane; ++column) {
            assert(array->weight(row, column, 0, 0) == expected_weight(row, column, 0, 0));
            assert(array->weight(row, column, 7, 9) == expected_weight(row, column, 7, 9));
            assert(array->weight(row, column, 15, 15) == expected_weight(row, column, 15, 15));
        }
    }

    bool caught = false;
    try {
        array->load_weights(ftlpu::hw::kMxmSupercellsPerPlane, 0, cell_input(0, 0), log);
    } catch (const std::out_of_range&) {
        caught = true;
    }
    assert(caught);

    caught = false;
    try {
        array->load_weights(0, ftlpu::hw::kMxmSupercellsPerPlane, cell_input(0, 0), log);
    } catch (const std::out_of_range&) {
        caught = true;
    }
    assert(caught);

    const auto text = log.str();
    assert(text.find("mxm_array cell(0,0) mxm_supercell: IW buffer0=0x") != std::string::npos);
    assert(text.find("mxm_array cell(3,3) mxm_supercell: IW buffer0=0x") != std::string::npos);

    return 0;
}
