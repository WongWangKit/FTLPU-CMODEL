#include "ftlpu/mxm/control_slice.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace {

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

} // namespace

int main(int argc, char** argv)
{
    const std::string log_path = argc > 1 ? argv[1] : "mxm_control_trace.log";
    std::ofstream log(log_path);
    if (!log) {
        std::cerr << "failed to open log file: " << log_path << '\n';
        return 1;
    }

    auto array = std::make_unique<ftlpu::MxmArray>();
    ftlpu::MxmControlSlice control(*array);
    constexpr std::size_t kColumn = 7;

    control.issue_south(ftlpu::MxmControlInstruction::IW(kColumn, 0));
    for (std::size_t cycle = 0; cycle < ftlpu::hw::kMxmSupercellsPerPlane; ++cycle) {
        control.set_weight_input(cycle, row_input(cycle, kColumn));
        control.tick(log);
    }

    std::cout << "wrote MXM control trace log: " << log_path << '\n';
    return 0;
}
