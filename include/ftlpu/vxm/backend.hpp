#pragma once

#include "ftlpu/vxm/slice.hpp"
#include "ftlpu/vxm_distributed/slice.hpp"

namespace ftlpu {

enum class VxmBackendKind {
    Established16Alu,
    DistributedCompact,
};

using EstablishedVxmSlice = VxmSlice;
using DistributedVxmSlice = distributed_vxm::VxmSlice;

template <VxmBackendKind Kind>
struct VxmBackendTraits;

template <>
struct VxmBackendTraits<VxmBackendKind::Established16Alu> {
    using Slice = EstablishedVxmSlice;
    static constexpr std::size_t instruction_queues = Slice::kAluQueues;
};

template <>
struct VxmBackendTraits<VxmBackendKind::DistributedCompact> {
    using Slice = DistributedVxmSlice;
    static constexpr std::size_t instruction_queues = Slice::kAluQueues;
};

} // namespace ftlpu
