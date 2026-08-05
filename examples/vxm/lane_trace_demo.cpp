#include "ftlpu/vxm/superlane.hpp"

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

    auto superlane = ftlpu::VxmSuperlane {};
    const auto params = ftlpu::VxmLane::SwigluParams {
        0.25f,
        0.5f,
        0.125f,
        0,
    };
    const std::vector<std::int32_t> gates {2, 4, -3, 8};
    const std::vector<std::int32_t> ups {11, 7, 5, -9};
    superlane.load_pipelined_swiglu_program(
        params, gates.size());

    std::ofstream log(log_path);
    if (!log) {
        throw std::runtime_error("failed to open VXM lane trace log");
    }

    for (std::size_t cycle = 0; cycle < gates.size() + 9; ++cycle) {
        if (cycle < gates.size()) {
            auto streams =
                ftlpu::VxmSuperlane::StreamMatrix {};
            for (auto& lane_streams : streams) {
                const auto gate =
                    ftlpu::VxmLane::pack_int32(
                        gates[cycle]);
                const auto up =
                    ftlpu::VxmLane::pack_int32(
                        ups[cycle]);
                for (std::size_t byte = 0;
                     byte < 4;
                     ++byte) {
                    lane_streams[byte] = gate[byte];
                    lane_streams[4 + byte] = up[byte];
                }
            }
            superlane.set_stream_inputs(
                ftlpu::Hemisphere::East,
                streams);
        }
        superlane.tick();
        superlane.print_lane_trace(log, 0);
        log << '\n';
    }

    std::cout << "wrote VXM lane trace log: " << log_path << '\n';
    return 0;
}
