#include "ftlpu/vxm/contract.hpp"

#include <cassert>

int main()
{
    using ftlpu::VxmCapability;
    using ftlpu::VxmInterfaceContract;
    static_assert(VxmInterfaceContract::initiation_interval_cycles == 1);
    static_assert(VxmInterfaceContract::one_finite_iq_per_alu);
    static_assert(!VxmInterfaceContract::hemisphere_selected_by_port_map);
    static_assert(VxmInterfaceContract::hemisphere_encoded_in_instruction);
    static_assert(VxmInterfaceContract::float16_stream_operand);
    static_assert(VxmInterfaceContract::float32_stream_operand);
    assert(VxmInterfaceContract::exp == VxmCapability::Native);
    assert(VxmInterfaceContract::reciprocal == VxmCapability::Composite);
    assert(VxmInterfaceContract::rsqrt == VxmCapability::Composite);
    assert(VxmInterfaceContract::lane_max_reduction
        == VxmCapability::NotImplemented);
    assert(VxmInterfaceContract::lane_sum_reduction
        == VxmCapability::NotImplemented);
    return 0;
}
