#include "ftlpu/vxm/lane.hpp"

#include <cstdint>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    const std::string log_path = argc > 1 ? argv[1] : "vxm_lane_trace.log";

    auto lane = ftlpu::VxmLane {};
    const auto params = ftlpu::VxmLane::SwigluParams {
        0.25f,
        0.5f,
        0.125f,
        0,
    };
    const std::vector<std::int32_t> gates {2, 4, -3, 8};
    const std::vector<std::int32_t> ups {11, 7, 5, -9};
    lane.load_pipelined_swiglu_program(params, gates.size());

    std::ofstream log(log_path);
    if (!log) {
        throw std::runtime_error("failed to open VXM lane trace log");
    }

    for (std::size_t cycle = 0; cycle < gates.size() + 9; ++cycle) {
        if (cycle < gates.size()) {
            lane.set_swiglu_input(
                ftlpu::VxmLane::pack_int32(gates[cycle]),
                ftlpu::VxmLane::pack_int32(ups[cycle]));
        }
        lane.tick();
        lane.print_last_trace(log);
        log << '\n';
    }

    std::cout << "wrote VXM lane trace log: " << log_path << '\n';
    return 0;
}
