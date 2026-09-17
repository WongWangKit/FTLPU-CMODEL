#include "ftlpu/system/icu.hpp"

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

template <typename Func>
void require_throws(Func&& func, const char* message)
{
    try {
        func();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

struct Issue {
    std::size_t cycle;
    ftlpu::MemInstruction instruction;
};

std::vector<Issue> run(ftlpu::MemIcuWriteRead2DInstruction descriptor,
    bool encoded = false)
{
    using namespace ftlpu;
    InstructionControlUnit icu;
    const auto index = InstructionControlUnit::mem_queue(
        Hemisphere::East, 0, 0);
    if (encoded)
        icu.mem_iq(index).push_encoded_mem_write_read_2d_packet(
            ftlpu::isa::encode_mem_icu_write_read_2d_instruction(
                descriptor));
    else
        icu.enqueue_mem_write_read_2d(index, descriptor);
    auto& queue = icu.mem_iq(index);
    require(queue.imem_occupancy() == 3,
        "WRITE_READ_2D must occupy exactly three 96-bit i-MEM words");
    require(queue.iq_depth == 16,
        "WRITE_READ_2D must fit the physical 16-word IQ");

    std::vector<Issue> issues;
    for (std::size_t step = 0; step < 256 && !queue.done(); ++step) {
        const auto cycle = queue.cycle();
        if (auto instruction = queue.tick())
            issues.push_back({cycle, *instruction});
    }
    require(queue.done(), "WRITE_READ_2D did not retire");
    require(issues.size() == 2 * descriptor.counts[0]
            * descriptor.counts[1],
        "WRITE_READ_2D issued the wrong number of FU instructions");
    return issues;
}

} // namespace

int main() try
{
    using namespace ftlpu;
    using namespace ftlpu::isa;

    // One Q-projection staging group: 4 rows x 2 lanes first receive MXM
    // results; each row is then read toward a RoPE input stream selected by
    // its outer coordinate. No other coarse MEM command is needed per bank.
    const MemIcuWriteRead2DInstruction qGroup {
        2, {4, 2}, {1, 4}, {1, 4}, 12,
        100, {1, 4}, StreamId::West(0).packed(),
        StreamId::East(0).packed(), 1};
    const auto packet = encode_mem_icu_write_read_2d_instruction(qGroup);
    const auto decoded = decode_mem_icu_write_read_2d_instruction(packet);
    require(decoded.counts == qGroup.counts
            && decoded.read_start_offset == qGroup.read_start_offset
            && decoded.address_strides == qGroup.address_strides
            && decoded.read_stream_outer_stride
                == qGroup.read_stream_outer_stride,
        "WRITE_READ_2D codec roundtrip failed");
    const auto serial = run(qGroup);
    for (std::size_t index = 0; index < 8; ++index) {
        require(serial[index].instruction.opcode == MemOpcode::Write
                && serial[index].instruction.address == 100 + index
                && serial[index].instruction.stream
                    == StreamId::West(0).packed(),
            "Q-group write FU order/address/stream is wrong");
        require(serial[index + 8].instruction.opcode == MemOpcode::Read
                && serial[index + 8].instruction.address == 100 + index
                && serial[index + 8].instruction.stream
                    == StreamId::East(index / 4).packed(),
            "Q-group read FU order/address/stream is wrong");
    }
    require(serial[8].cycle - serial[0].cycle == 12,
        "WRITE_READ_2D did not honor read_start_offset");
    auto noWait = qGroup;
    noWait.start_wait = 0;
    require(serial[0].cycle - run(noWait)[0].cycle == 2,
        "WRITE_READ_2D did not honor queue-relative start_wait");

    auto interleaved = qGroup;
    interleaved.start_wait = 0;
    interleaved.write_cycle_strides = {2, 8};
    interleaved.read_cycle_strides = {2, 8};
    interleaved.read_start_offset = 1;
    const auto alternating = run(interleaved, true);
    for (std::size_t index = 0; index < alternating.size(); ++index) {
        require(alternating[index].instruction.opcode
                == (index % 2 == 0 ? MemOpcode::Write : MemOpcode::Read),
            "WRITE_READ_2D did not interleave the two static sweeps");
        if (index != 0)
            require(alternating[index].cycle
                    == alternating[index - 1].cycle + 1,
                "WRITE_READ_2D parity schedule inserted a bubble");
    }

    auto conflict = interleaved;
    conflict.read_start_offset = 2;
    require_throws([&] {
        static_cast<void>(encode_mem_icu_write_read_2d_instruction(conflict));
    }, "same-cycle single-port conflict was accepted");

    auto reused = qGroup;
    reused.address_strides[1] = 0;
    require_throws([&] {
        static_cast<void>(encode_mem_icu_write_read_2d_instruction(reused));
    }, "unproved row overwrite was accepted");

    auto badPacket = packet;
    badPacket.words[2].lanes[2] |= std::uint32_t {1} << 31;
    require_throws([&] {
        static_cast<void>(decode_mem_icu_write_read_2d_instruction(
            badPacket));
    }, "nonzero reserved bit was accepted");

    std::cout << "MEM WRITE_READ_2D codec and raw queue test passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "MEM WRITE_READ_2D test failed: " << error.what() << '\n';
    return 1;
}
