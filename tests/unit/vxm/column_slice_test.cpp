#include "ftlpu/vxm/column_slice.hpp"
#include "ftlpu/vxm/slice.hpp"

#include <cassert>
#include <cstddef>

int main()
{
    using namespace ftlpu;

    static_assert(VxmSlice::kAluQueues == hw::kVxmAluCount);
    static_assert(VxmSuperlaneInstructionControl::kColumnCount
        == hw::kVxmAluCount);

    // The first and last configured physical columns have separate
    // FIFO/decode/config/repeat state for any supported physical width.
    constexpr std::size_t lastColumn = hw::kVxmAluCount - 1;
    auto control = VxmSuperlaneInstructionControl {};
    auto low = VxmLaneAluInstruction {
        VxmAluOpcode::Bypass, VxmLaneOperand::StreamFloat16()};
    low.repeat_count = 2;
    auto high = low;
    high.repeat_count = 1;
    control.enqueue(0, VxmChainDepth::Four, low);
    control.enqueue(lastColumn, VxmChainDepth::Four, high);
    control.issue(VxmChainDepth::Four); // FIFO -> Decoder
    const auto issued = control.issue(VxmChainDepth::Four);
    assert(issued[0] && issued[lastColumn]);
    auto executed = VxmLaneExecutionMask {};
    executed[lastColumn] = true;
    control.consume(executed);
    assert(control.remaining_in_current(0) == 2);
    assert(control.remaining_in_current(lastColumn) == 0);

    return 0;
}
