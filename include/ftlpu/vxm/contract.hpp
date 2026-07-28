#pragma once

#include "ftlpu/core/hardware_params.hpp"

#include <cstddef>

namespace ftlpu {

enum class VxmCapability {
    Native,
    Composite,
    NotImplemented,
};

// Stable description of the VXM implementation retained on feat/cjh and
// extended with the minimum f621374 workload-facing operand/routing fields.
struct VxmInterfaceContract {
    static constexpr std::size_t lane_count = hw::kVxmLaneCount;
    static constexpr std::size_t alu_queue_count = hw::kVxmAluCount;
    static constexpr std::size_t pipeline_stages =
        hw::kVxmPipelineStages;
    static constexpr std::size_t initiation_interval_cycles = 1;
    static constexpr std::size_t encoded_stream_index_bits = 6;
    static constexpr std::size_t encoded_alu_index_bits = 6;

    static constexpr bool hemisphere_selected_by_port_map = false;
    static constexpr bool hemisphere_encoded_in_instruction = true;
    static constexpr bool float16_stream_operand = true;
    static constexpr bool float32_stream_operand = true;
    static constexpr bool int8_stream_operand = true;
    static constexpr bool one_finite_iq_per_alu = true;

    static constexpr VxmCapability exp = VxmCapability::Native;
    static constexpr VxmCapability sqrt = VxmCapability::Native;
    static constexpr VxmCapability reciprocal =
        VxmCapability::Composite; // Divide(1, x)
    static constexpr VxmCapability rsqrt =
        VxmCapability::Composite; // Sqrt, then Divide(1, x)
    static constexpr VxmCapability lane_max_reduction =
        VxmCapability::NotImplemented;
    static constexpr VxmCapability lane_sum_reduction =
        VxmCapability::NotImplemented;
};

static_assert(
    VxmInterfaceContract::pipeline_stages
        == VxmInterfaceContract::alu_queue_count,
    "retained VXM requires one physical pipeline stage per ALU IQ");
static_assert(
    (std::size_t {1}
        << VxmInterfaceContract::encoded_stream_index_bits)
        >= hw::kStreams,
    "VXM packet stream field cannot encode the configured SR streams");

} // namespace ftlpu
