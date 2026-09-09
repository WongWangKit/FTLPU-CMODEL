#pragma once

#include "ftlpu/c2c/slice.hpp"
#include "ftlpu/core/stream_topology_descriptor.hpp"
#include "ftlpu/mem/mem_array.hpp"
#include "ftlpu/sxm/slice.hpp"
#include "ftlpu/system/stream_topology.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace ftlpu {

// A resolved endpoint in one instance of the fabric template.
struct StreamEndpoint {
    std::size_t column{0};
    StreamDirection direction{StreamDirection::East};
};

struct StreamSystemTransfer {
    std::string name{};
    target::StreamSystemTransferKind kind{
        target::StreamSystemTransferKind::PassiveBridge};
    std::size_t source_fabric{0};
    StreamEndpoint source{};
    std::size_t destination_fabric{0};
    StreamEndpoint destination{};
    std::size_t latency_cycles{1};
};

// Runtime CModel view derived from an immutable target descriptor.  The
// topology is the per-fabric route graph; the remaining fields bind modeled
// functional units and transfers between fabric replicas to that graph.
struct StreamLayout {
    StreamTopology topology{};
    std::vector<std::string> fabric_names{};
    MemStreamPortMap mem_ports;
    SxmStreamPortMap sxm_ports;
    C2cStreamPortMap c2c_ports{};
    StreamEndpoint mxm_weight_input{};
    StreamEndpoint mxm_activation_input{};
    StreamEndpoint mxm_result_output{};
    StreamEndpoint vxm_input{};
    StreamEndpoint vxm_output{};
    std::vector<StreamSystemTransfer> system_transfers{};
};

StreamLayout make_stream_layout(
    const target::StreamTopologyDescriptor& descriptor);

// Construct the stream layout from the constexpr descriptor generated for the
// hardware configuration selected at CMake configure time.
StreamLayout make_configured_stream_layout();

} // namespace ftlpu
