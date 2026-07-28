#pragma once

#include "ftlpu/program/packet_encoder.hpp"
#include "ftlpu/program/program_image.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ftlpu::program {

// Splits a logical IQ program into 640-byte blocks. Every non-final block
// starts with Fetch(next), so later blocks enter the same finite IQ through
// the normal IFetch path. The loader must present the matching SRAM blocks on
// instruction_stream in placement order.
inline ProgramSection build_chained_section(
    IcuLocation target,
    StreamId instruction_stream,
    const std::vector<isa::EncodedInstructionPacket>& payload,
    std::string metadata = {})
{
    if (payload.empty()) {
        throw std::invalid_argument("chained program payload must not be empty");
    }
    static_assert(
        hw::kIcuFetchPackets >= 2,
        "chained IFetch needs room for a Fetch packet and program payload");

    ProgramSection result {target, {}, 0, std::move(metadata)};
    std::size_t cursor = 0;
    while (payload.size() - cursor > hw::kIcuFetchPackets) {
        result.packets.push_back(encode_packet(
            IcuControlInstruction::Fetch(instruction_stream)));
        const auto payload_in_block = hw::kIcuFetchPackets - 1;
        result.packets.insert(
            result.packets.end(),
            payload.begin() + cursor,
            payload.begin() + cursor + payload_in_block);
        cursor += payload_in_block;
    }
    result.packets.insert(
        result.packets.end(), payload.begin() + cursor, payload.end());
    return result;
}

inline std::size_t chained_block_count(std::size_t payload_packet_count)
{
    if (payload_packet_count == 0) {
        return 0;
    }
    if (payload_packet_count <= hw::kIcuFetchPackets) {
        return 1;
    }
    return 1
        + (payload_packet_count - hw::kIcuFetchPackets
            + (hw::kIcuFetchPackets - 2))
            / (hw::kIcuFetchPackets - 1);
}

} // namespace ftlpu::program
