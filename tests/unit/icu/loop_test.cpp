#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/icu/distributed_queue.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using Queue = ftlpu::DistributedIcuQueue<
    ftlpu::MemInstruction, 64, 64, 8, 1>;
using MxmQueue = ftlpu::DistributedIcuQueue<
    ftlpu::MxmControlInstruction, 128, 64, 8, 1>;

void require(bool condition, const char* message)
{
    if (!condition) throw std::logic_error(message);
}

} // namespace

int main()
{
    try {
        const ftlpu::IcuLoop loop {3, 2, 5, 16};
        const auto encoded = ftlpu::isa::encode_icu_loop(loop);
        require(ftlpu::isa::decode_icu_command_opcode(encoded)
                == ftlpu::isa::IcuCommandOpcode::Loop,
            "Loop opcode roundtrip failed");
        const auto decoded = ftlpu::isa::decode_icu_loop(encoded);
        require(decoded.window_size == loop.window_size
                && decoded.count == loop.count
                && decoded.interval == loop.interval
                && decoded.address_stride == loop.address_stride,
            "Loop field roundtrip failed");

        Queue queue;
        queue.push_instruction(ftlpu::MemInstruction::Read(100, 0));
        queue.push_instruction(ftlpu::MemInstruction::Read(101, 1));
        queue.push_instruction(ftlpu::MemInstruction::Read(102, 2));
        queue.push_loop(loop);

        std::vector<std::size_t> cycles;
        std::vector<std::size_t> addresses;
        while (!queue.done()) {
            const auto cycle = queue.cycle();
            if (auto instruction = queue.tick()) {
                cycles.push_back(cycle);
                addresses.push_back(instruction->address);
            }
        }
        constexpr std::array<std::size_t, 9> expectedCycles {
            0, 1, 2, 3, 4, 5, 8, 9, 10};
        constexpr std::array<std::size_t, 9> expectedAddresses {
            100, 101, 102, 116, 117, 118, 132, 133, 134};
        require(std::equal(cycles.begin(), cycles.end(),
                    expectedCycles.begin(), expectedCycles.end()),
            "Loop issue-cycle sequence is wrong");
        require(std::equal(addresses.begin(), addresses.end(),
                    expectedAddresses.begin(), expectedAddresses.end()),
            "Loop MEM address-stride sequence is wrong");

        MxmQueue mxm;
        mxm.push_instruction(ftlpu::MxmControlInstruction::Compute(
            0, 0, 0, 4, 1,
            ftlpu::MxmAccumulatorDestination::Sram,
            ftlpu::MxmDataFormat::BFloat16,
            ftlpu::MxmComputeMode::Block8));
        mxm.push_loop(ftlpu::IcuLoop {1, 2, 2, 3});
        std::vector<std::size_t> accumulatorAddresses;
        while (!mxm.done()) {
            if (auto instruction = mxm.tick())
                accumulatorAddresses.push_back(
                    instruction->accumulator_address);
        }
        constexpr std::array<std::size_t, 3> expectedAccumulatorAddresses {
            4, 7, 10};
        require(std::equal(accumulatorAddresses.begin(),
                    accumulatorAddresses.end(),
                    expectedAccumulatorAddresses.begin(),
                    expectedAccumulatorAddresses.end()),
            "Loop MXM accumulator-address stride sequence is wrong");

        Queue invalid;
        invalid.push_instruction(ftlpu::MemInstruction::Read(0, 0));
        invalid.push_loop(ftlpu::IcuLoop {2, 1, 2, 0});
        require(invalid.tick().has_value(),
            "invalid-window setup did not issue its instruction");
        bool rejected = false;
        try {
            static_cast<void>(invalid.tick());
        } catch (const std::logic_error&) {
            rejected = true;
        }
        require(rejected, "Loop accepted a window larger than history");

        std::cout << "icu_loop_test passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "icu_loop_test failed: " << error.what() << '\n';
        return 1;
    }
}
