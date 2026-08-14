#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/icu/distributed_queue.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::logic_error(message);
}

} // namespace

int main()
try {
    using namespace ftlpu;
    const IcuRepeat2D repeat {
        4, 1, -1,
        3, 8, 16,
        IcuInductionTarget::MemAddress,
    };
    const auto decoded = isa::decode_icu_repeat_2d(
        isa::encode_icu_repeat_2d(repeat));
    require(decoded.inner_count == 4 && decoded.inner_interval == 1
            && decoded.inner_stride == -1
            && decoded.outer_count == 3 && decoded.outer_interval == 8
            && decoded.outer_stride == 16
            && decoded.induction_target
                == IcuInductionTarget::MemAddress,
        "ICU Repeat2D codec round trip failed");

    using MemQueue = DistributedIcuQueue<MemInstruction, 96, 64, 16, 1>;
    MemQueue mem;
    mem.push_instruction(MemInstruction::Read(100, 0));
    mem.push_repeat_2d(repeat);
    std::vector<std::size_t> cycles;
    std::vector<std::size_t> addresses;
    for (std::size_t cycle = 0; cycle <= 20; ++cycle) {
        if (const auto instruction = mem.tick()) {
            cycles.push_back(cycle);
            addresses.push_back(instruction->address);
        }
    }
    require(cycles == std::vector<std::size_t> {
                0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19},
        "ICU Repeat2D issued MEM instructions at incorrect cycles");
    require(addresses == std::vector<std::size_t> {
                100, 99, 98, 97, 116, 115, 114, 113,
                132, 131, 130, 129},
        "ICU Repeat2D applied incorrect MEM address induction");

    using MxmQueue = DistributedIcuQueue<
        MxmControlInstruction, 128, 64, 16, 1>;
    MxmQueue mxm;
    mxm.push_instruction(MxmControlInstruction::IW(0, 0));
    mxm.push_repeat_2d(IcuRepeat2D {
        4, 1, 1,
        2, 8, 4,
        IcuInductionTarget::MxmWeightColumn,
    });
    std::vector<std::size_t> columns;
    for (std::size_t cycle = 0; cycle <= 12; ++cycle)
        if (const auto instruction = mxm.tick())
            columns.push_back(instruction->weight_column);
    require(columns == std::vector<std::size_t> {0, 1, 2, 3, 4, 5, 6, 7},
        "ICU Repeat2D applied incorrect MXM column induction");

    MxmQueue compute;
    compute.push_instruction(MxmControlInstruction::Compute(
        0, 0, 0, 4, 1, MxmAccumulatorDestination::Sram,
        MxmDataFormat::BFloat16, MxmComputeMode::Block8, false));
    compute.push_repeat_2d(IcuRepeat2D {
        4, 1, 0,
        3, 8, 4,
        IcuInductionTarget::MxmAccumulatorAddress,
    });
    std::vector<std::size_t> accumulator_addresses;
    for (std::size_t cycle = 0; cycle <= 20; ++cycle)
        if (const auto instruction = compute.tick())
            accumulator_addresses.push_back(
                instruction->accumulator_address);
    require(accumulator_addresses == std::vector<std::size_t> {
                4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12},
        "ICU Repeat2D applied incorrect MXM accumulator induction");

    std::cout << "icu_repeat_2d_test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "icu_repeat_2d_test failed: " << error.what() << '\n';
    return 1;
}
