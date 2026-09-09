#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace ftlpu::target {

enum class StreamDirection : std::uint8_t {
    East,
    West,
};

enum class StreamRouteKind : std::uint8_t {
    Normal,
    Bypass,
};

enum class StreamSystemTransferKind : std::uint8_t {
    PassiveBridge,
};

struct StreamColumnDescriptor {
    std::string_view name{};
};

struct StreamRouteDescriptor {
    std::string_view name{};
    std::size_t source_column{0};
    std::size_t destination_column{0};
    StreamDirection direction{StreamDirection::East};
    StreamRouteKind kind{StreamRouteKind::Normal};
    bool enabled_by_default{true};
    bool multicast_allowed{false};
    std::size_t latency_cycles{1};
};

struct StreamPortDescriptor {
    std::size_t column{0};
    StreamDirection direction{StreamDirection::East};
};

struct SxmStreamBindingDescriptor {
    StreamPortDescriptor east_input{};
    StreamPortDescriptor east_output{};
    StreamPortDescriptor west_input{};
    StreamPortDescriptor west_output{};
};

struct MxmStreamBindingDescriptor {
    StreamPortDescriptor weight_input{};
    StreamPortDescriptor activation_input{};
    StreamPortDescriptor result_output{};
};

struct VxmStreamBindingDescriptor {
    StreamPortDescriptor input{};
    StreamPortDescriptor output{};
};

struct C2cStreamBindingDescriptor {
    StreamPortDescriptor tx_input{};
    StreamPortDescriptor rx_output{};
};

struct StreamSystemTransferDescriptor {
    std::string_view name{};
    StreamSystemTransferKind kind{StreamSystemTransferKind::PassiveBridge};
    std::size_t source_fabric{0};
    StreamPortDescriptor source{};
    std::size_t destination_fabric{0};
    StreamPortDescriptor destination{};
    std::size_t latency_cycles{1};
};

// Immutable physical stream-network data generated from the shared target
// JSON. It intentionally has no dependency on CModel execution types, LLVM,
// or MLIR so both the model and compiler can consume the same descriptor.
struct StreamTopologyDescriptor {
    std::span<const std::string_view> fabric_names{};
    std::span<const StreamColumnDescriptor> columns{};
    std::span<const StreamRouteDescriptor> routes{};

    std::size_t mem_slices_per_group{0};
    std::span<const std::size_t> mem_boundary_columns{};
    SxmStreamBindingDescriptor sxm{};
    MxmStreamBindingDescriptor mxm{};
    VxmStreamBindingDescriptor vxm{};
    C2cStreamBindingDescriptor c2c{};

    std::span<const StreamSystemTransferDescriptor> system_transfers{};
};

} // namespace ftlpu::target
