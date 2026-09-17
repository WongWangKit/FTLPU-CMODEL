#pragma once

#include "ftlpu/c2c/dma_instruction.hpp"
#include "ftlpu/c2c/instruction.hpp"
#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/hemisphere.hpp"

#include <cstddef>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ftlpu {

// C2C has fixed local-iMEM words, independent of the richer CModel command
// objects below.  Endpoint commands always occupy one 96-bit word; DMA
// commands occupy two consecutive 96-bit words.  Bit zero is the least
// significant bit of lanes[0].
struct C2cEndpointIcuPacket {
    static constexpr std::size_t kWordBits = 96;
    static constexpr std::size_t kWordCount = 1;
    static constexpr std::size_t kLanesPerWord = 3;
    std::array<std::uint32_t, kLanesPerWord> lanes{};
};

struct C2cDmaIcuPacket {
    static constexpr std::size_t kWordBits = 96;
    static constexpr std::size_t kWordCount = 2;
    static constexpr std::size_t kLanesPerWord = 3;
    std::array<C2cEndpointIcuPacket, kWordCount> words{};
};

static_assert(sizeof(C2cEndpointIcuPacket) == 12,
    "a C2C endpoint ICU packet must be exactly 96 bits");
static_assert(sizeof(C2cDmaIcuPacket) == 24,
    "a C2C DMA ICU packet must be exactly two 96-bit words");

// Hardware-facing C2C ICU instructions.  Endpoint transport and SRAM writes
// are deliberately separate: an RX instruction names the ordinary SR that
// receives a vector and may route a completion token to one MEM ICU, but it
// never carries an SRAM row or row stride.  Those fields belong to the MEM
// ICU instruction waiting on its synchronization tag.

struct C2cMemNotifyRoute {
    bool enabled{false};
    Hemisphere hemisphere{Hemisphere::East};
    std::size_t mem_slice{0};
    std::size_t mem_bank{0};

    static C2cMemNotifyRoute Disabled() noexcept { return {}; }

    static C2cMemNotifyRoute Mem(Hemisphere hemisphere,
        std::size_t mem_slice, std::size_t mem_bank)
    {
        C2cMemNotifyRoute route {
            true, hemisphere, mem_slice, mem_bank};
        route.validate();
        return route;
    }

    void validate() const
    {
        if (hemisphere_index(hemisphere) >= hw::kHemispheres)
            throw std::out_of_range(
                "C2C RX notification hemisphere is outside the chip");
        if (mem_slice >= hw::kMemSliceColumns)
            throw std::out_of_range(
                "C2C RX notification slice is outside the MEM array");
        if (mem_bank >= hw::kMemBanksPerSlice)
            throw std::out_of_range(
                "C2C RX notification bank is outside the MEM slice");
        if (!enabled
            && (hemisphere != Hemisphere::East
                || mem_slice != 0 || mem_bank != 0))
            throw std::invalid_argument(
                "disabled C2C RX notification route must be canonical");
    }
};

struct C2cTxIcuInstruction {
    Hemisphere endpoint_hemisphere{Hemisphere::East};
    std::size_t lane{0};
    std::size_t fabric_stream{0};
    std::size_t vector_count{1};
    std::uint32_t sync_tag{0};

    static C2cTxIcuInstruction Send(Hemisphere endpoint_hemisphere,
        std::size_t lane, std::size_t fabric_stream,
        std::size_t vector_count = 1, std::uint32_t sync_tag = 0)
    {
        C2cTxIcuInstruction instruction {endpoint_hemisphere, lane,
            fabric_stream, vector_count, sync_tag};
        instruction.validate();
        return instruction;
    }

    static C2cTxIcuInstruction FromLegacy(Hemisphere endpoint_hemisphere,
        const C2cInstruction& instruction,
        std::uint32_t sync_tag = std::numeric_limits<std::uint32_t>::max())
    {
        if (instruction.opcode != C2cOpcode::Send)
            throw std::invalid_argument(
                "C2C TX ICU adapter requires a Send instruction");
        return Send(endpoint_hemisphere, instruction.stream_index,
            instruction.fabric_stream_index, instruction.vector_count,
            sync_tag == std::numeric_limits<std::uint32_t>::max()
                ? instruction.sync_tag : sync_tag);
    }

    C2cInstruction to_legacy() const
    {
        validate();
        auto instruction = C2cInstruction::Send(
            lane, vector_count, fabric_stream);
        instruction.sync_tag = sync_tag;
        return instruction;
    }

    void validate() const
    {
        validate_endpoint(endpoint_hemisphere, lane, fabric_stream);
        if (vector_count == 0)
            throw std::invalid_argument(
                "C2C TX ICU vector_count must be non-zero");
    }

private:
    // Kept in one helper so TX, RX, and DMA enforce the same physical lane
    // geometry before their packed codecs apply narrower field limits.
    static void validate_endpoint(Hemisphere endpoint_hemisphere,
        std::size_t lane, std::size_t fabric_stream = 0,
        bool validate_fabric_stream = true)
    {
        if (hemisphere_index(endpoint_hemisphere) >= hw::kHemispheres)
            throw std::out_of_range(
                "C2C ICU endpoint hemisphere is outside the chip");
        if (lane >= hw::kC2cStreamsPerDirection)
            throw std::out_of_range(
                "C2C ICU lane is outside the directional transport lanes");
        if (validate_fabric_stream
            && fabric_stream >= hw::kStreamsPerDirection)
            throw std::out_of_range(
                "C2C ICU stream is outside the ordinary SR file");
    }

    friend struct C2cRxIcuInstruction;
    friend struct C2cDmaIcuInstruction;
};

struct C2cRxIcuInstruction {
    Hemisphere endpoint_hemisphere{Hemisphere::East};
    std::size_t lane{0};
    std::size_t fabric_stream{0};
    std::size_t vector_count{1};
    std::uint32_t sync_tag{0};
    C2cMemNotifyRoute notify{};

    static C2cRxIcuInstruction Receive(Hemisphere endpoint_hemisphere,
        std::size_t lane, std::size_t fabric_stream,
        std::size_t vector_count = 1, std::uint32_t sync_tag = 0,
        C2cMemNotifyRoute notify = C2cMemNotifyRoute::Disabled())
    {
        C2cRxIcuInstruction instruction {endpoint_hemisphere, lane,
            fabric_stream, vector_count, sync_tag, notify};
        instruction.validate();
        return instruction;
    }

    static C2cRxIcuInstruction FromLegacy(Hemisphere endpoint_hemisphere,
        const C2cInstruction& instruction,
        std::uint32_t sync_tag = std::numeric_limits<std::uint32_t>::max())
    {
        if (instruction.opcode != C2cOpcode::Receive)
            throw std::invalid_argument(
                "C2C RX ICU adapter requires a Receive instruction");
        const auto route = instruction.consumer.notify_mem
            ? C2cMemNotifyRoute::Mem(instruction.consumer.hemisphere,
                  instruction.consumer.mem_slice,
                  instruction.consumer.mem_bank)
            : C2cMemNotifyRoute::Disabled();
        return Receive(endpoint_hemisphere, instruction.stream_index,
            instruction.fabric_stream_index,
            instruction.consumer.vector_count,
            sync_tag == std::numeric_limits<std::uint32_t>::max()
                ? instruction.sync_tag : sync_tag, route);
    }

    // The compatibility object still combines endpoint receive and SRAM
    // placement.  Callers must explicitly supply those MEM-owned fields;
    // they are intentionally absent from the packed RX ICU instruction.
    C2cInstruction to_legacy(
        std::size_t base_row, std::size_t row_stride = 1) const
    {
        validate();
        auto instruction = C2cInstruction::Receive(lane,
            notify.enabled ? notify.hemisphere : Hemisphere::East,
            notify.enabled ? notify.mem_slice : 0,
            notify.enabled ? notify.mem_bank : 0,
            notify.enabled, base_row, vector_count, row_stride,
            fabric_stream);
        instruction.sync_tag = sync_tag;
        return instruction;
    }

    void validate() const
    {
        C2cTxIcuInstruction::validate_endpoint(
            endpoint_hemisphere, lane, fabric_stream);
        if (vector_count == 0)
            throw std::invalid_argument(
                "C2C RX ICU vector_count must be non-zero");
        notify.validate();
    }
};

struct C2cDmaIcuInstruction {
    C2cDmaDirection direction{C2cDmaDirection::Ddr4ToC2c};
    Hemisphere endpoint_hemisphere{Hemisphere::East};
    std::size_t lane{0};
    std::uint64_t ddr4_address{0};
    std::size_t vector_count{1};
    std::size_t address_stride_bytes{hw::kPhysicalVectorBytes};
    std::uint32_t sync_tag{0};

    static C2cDmaIcuInstruction Load(Hemisphere endpoint_hemisphere,
        std::size_t lane, std::uint64_t ddr4_address,
        std::size_t vector_count = 1,
        std::size_t address_stride_bytes = hw::kPhysicalVectorBytes,
        std::uint32_t sync_tag = 0)
    {
        C2cDmaIcuInstruction instruction {C2cDmaDirection::Ddr4ToC2c,
            endpoint_hemisphere, lane, ddr4_address, vector_count,
            address_stride_bytes, sync_tag};
        instruction.validate();
        return instruction;
    }

    static C2cDmaIcuInstruction Store(Hemisphere endpoint_hemisphere,
        std::size_t lane, std::uint64_t ddr4_address,
        std::size_t vector_count = 1,
        std::size_t address_stride_bytes = hw::kPhysicalVectorBytes,
        std::uint32_t sync_tag = 0)
    {
        C2cDmaIcuInstruction instruction {C2cDmaDirection::C2cToDdr4,
            endpoint_hemisphere, lane, ddr4_address, vector_count,
            address_stride_bytes, sync_tag};
        instruction.validate();
        return instruction;
    }

    static C2cDmaIcuInstruction FromLegacy(Hemisphere endpoint_hemisphere,
        const C2cDmaInstruction& instruction,
        std::uint32_t sync_tag = std::numeric_limits<std::uint32_t>::max())
    {
        auto result = C2cDmaIcuInstruction {instruction.direction,
            endpoint_hemisphere, instruction.stream_index,
            instruction.ddr4_address, instruction.vector_count,
            instruction.address_stride_bytes,
            sync_tag == std::numeric_limits<std::uint32_t>::max()
                ? instruction.sync_tag : sync_tag};
        result.validate();
        return result;
    }

    // vector_tag_base remains CModel-only diagnostic metadata and is not part
    // of the hardware packet.  The caller may provide it when adapting back.
    C2cDmaInstruction to_legacy(std::uint64_t vector_tag_base = 0) const
    {
        validate();
        auto instruction = direction == C2cDmaDirection::Ddr4ToC2c
            ? C2cDmaInstruction::Load(ddr4_address, vector_count,
                  address_stride_bytes, vector_tag_base, lane)
            : C2cDmaInstruction::Store(ddr4_address, vector_count,
                  address_stride_bytes, lane);
        instruction.sync_tag = sync_tag;
        return instruction;
    }

    std::uint64_t vector_address(std::size_t vector_index) const
    {
        if (vector_index >= vector_count)
            throw std::out_of_range(
                "C2C DMA ICU vector index is outside the instruction");
        if (address_stride_bytes != 0
            && vector_index
                > (std::numeric_limits<std::uint64_t>::max()
                    - ddr4_address) / address_stride_bytes)
            throw std::overflow_error("C2C DMA ICU DDR4 address overflow");
        return ddr4_address + vector_index * address_stride_bytes;
    }

    void validate() const
    {
        C2cTxIcuInstruction::validate_endpoint(
            endpoint_hemisphere, lane, 0, false);
        if (vector_count == 0)
            throw std::invalid_argument(
                "C2C DMA ICU vector_count must be non-zero");
        (void)vector_address(vector_count - 1);
    }
};

class C2cIcuPacketCodec {
public:
    // Endpoint word: [1:0] Instruction envelope, [2] RX selector,
    // [3] endpoint hemisphere, [6:4] lane, [11:7] ordinary fabric SR,
    // [27:12] vector_count - 1, [43:28] sync tag, [44] MEM notify valid,
    // [45] target hemisphere, [51:46] target slice, [52] target bank.
    // All remaining bits are zero.
    static C2cEndpointIcuPacket encode(const C2cTxIcuInstruction& value)
    {
        value.validate();
        auto packet = C2cEndpointIcuPacket {};
        write(packet, 3, 1, hemisphere_index(value.endpoint_hemisphere));
        write(packet, 4, 3, value.lane);
        write(packet, 7, 5, value.fabric_stream);
        write(packet, 12, 16, value.vector_count - 1);
        write(packet, 28, 16, checked_u16(value.sync_tag, "C2C TX sync tag"));
        return packet;
    }

    static C2cEndpointIcuPacket encode(const C2cRxIcuInstruction& value)
    {
        value.validate();
        auto packet = C2cEndpointIcuPacket {};
        write(packet, 2, 1, 1);
        write(packet, 3, 1, hemisphere_index(value.endpoint_hemisphere));
        write(packet, 4, 3, value.lane);
        write(packet, 7, 5, value.fabric_stream);
        write(packet, 12, 16, value.vector_count - 1);
        write(packet, 28, 16, checked_u16(value.sync_tag, "C2C RX sync tag"));
        write(packet, 44, 1, value.notify.enabled ? 1 : 0);
        if (value.notify.enabled) {
            write(packet, 45, 1, hemisphere_index(value.notify.hemisphere));
            write(packet, 46, 6, value.notify.mem_slice);
            write(packet, 52, 1, value.notify.mem_bank);
        }
        return packet;
    }

    static C2cTxIcuInstruction decode_tx(const C2cEndpointIcuPacket& packet)
    {
        validate_endpoint_reserved(packet);
        if (read(packet, 0, 2) != 0 || read(packet, 2, 1) != 0)
            throw std::invalid_argument("C2C endpoint packet is RX, not TX");
        return C2cTxIcuInstruction::Send(
            decode_hemisphere(read(packet, 3, 1)), read(packet, 4, 3),
            read(packet, 7, 5), read(packet, 12, 16) + 1,
            static_cast<std::uint32_t>(read(packet, 28, 16)));
    }

    static C2cRxIcuInstruction decode_rx(const C2cEndpointIcuPacket& packet)
    {
        validate_endpoint_reserved(packet);
        if (read(packet, 0, 2) != 0 || read(packet, 2, 1) != 1)
            throw std::invalid_argument("C2C endpoint packet is TX, not RX");
        const auto notify = read(packet, 44, 1) != 0
            ? C2cMemNotifyRoute::Mem(decode_hemisphere(read(packet, 45, 1)),
                  read(packet, 46, 6), read(packet, 52, 1))
            : C2cMemNotifyRoute::Disabled();
        return C2cRxIcuInstruction::Receive(
            decode_hemisphere(read(packet, 3, 1)), read(packet, 4, 3),
            read(packet, 7, 5), read(packet, 12, 16) + 1,
            static_cast<std::uint32_t>(read(packet, 28, 16)), notify);
    }

    // DMA word 0: [1:0] Instruction envelope, [2] direction,
    // [3] endpoint hemisphere, [6:4] lane, [22:7] vector_count - 1,
    // [38:23] byte stride, [54:39] sync tag. DMA word 1 carries DDR
    // address at [63:0] and a continuation marker at [95].
    static C2cDmaIcuPacket encode(const C2cDmaIcuInstruction& value)
    {
        value.validate();
        if (value.vector_count > (std::uint64_t {1} << 16)
            || value.address_stride_bytes > std::numeric_limits<std::uint16_t>::max()
            || value.sync_tag > std::numeric_limits<std::uint16_t>::max())
            throw std::out_of_range("C2C DMA field does not fit its fixed packet");
        auto packet = C2cDmaIcuPacket {};
        write(packet.words[0], 2, 1,
            value.direction == C2cDmaDirection::C2cToDdr4 ? 1 : 0);
        write(packet.words[0], 3, 1, hemisphere_index(value.endpoint_hemisphere));
        write(packet.words[0], 4, 3, value.lane);
        write(packet.words[0], 7, 16, value.vector_count - 1);
        write(packet.words[0], 23, 16, value.address_stride_bytes);
        write(packet.words[0], 39, 16, value.sync_tag);
        write(packet.words[1], 0, 64, value.ddr4_address);
        write(packet.words[1], 95, 1, 1);
        return packet;
    }

    static C2cDmaIcuInstruction decode_dma(const C2cDmaIcuPacket& packet)
    {
        if (read(packet.words[0], 0, 2) != 0
            || !zero(packet.words[0], 55, 96)
            || !zero(packet.words[1], 64, 95)
            || read(packet.words[1], 95, 1) != 1)
            throw std::invalid_argument("C2C DMA packet has non-zero reserved bits");
        const auto direction = read(packet.words[0], 2, 1) == 0
            ? C2cDmaDirection::Ddr4ToC2c : C2cDmaDirection::C2cToDdr4;
        const auto hemisphere = decode_hemisphere(read(packet.words[0], 3, 1));
        const auto lane = read(packet.words[0], 4, 3);
        const auto address = read(packet.words[1], 0, 64);
        const auto count = read(packet.words[0], 7, 16) + 1;
        const auto stride = read(packet.words[0], 23, 16);
        const auto tag = static_cast<std::uint32_t>(read(packet.words[0], 39, 16));
        return direction == C2cDmaDirection::Ddr4ToC2c
            ? C2cDmaIcuInstruction::Load(hemisphere, lane, address, count, stride, tag)
            : C2cDmaIcuInstruction::Store(hemisphere, lane, address, count, stride, tag);
    }

private:
    static std::uint64_t checked_u16(std::uint32_t value, const char* field)
    {
        if (value > std::numeric_limits<std::uint16_t>::max())
            throw std::out_of_range(std::string(field) + " does not fit packet");
        return value;
    }
    static void write(C2cEndpointIcuPacket& packet, std::size_t offset,
        unsigned width, std::uint64_t value)
    {
        if (width > 64 || offset + width > 96
            || (width != 64 && value >= (std::uint64_t {1} << width)))
            throw std::out_of_range("C2C packet bitfield write is invalid");
        for (unsigned bit = 0; bit < width; ++bit)
            if ((value & (std::uint64_t {1} << bit)) != 0)
                packet.lanes[(offset + bit) / 32]
                    |= std::uint32_t {1} << ((offset + bit) % 32);
    }
    static std::uint64_t read(const C2cEndpointIcuPacket& packet,
        std::size_t offset, unsigned width)
    {
        if (width > 64 || offset + width > 96)
            throw std::out_of_range("C2C packet bitfield read is invalid");
        std::uint64_t value = 0;
        for (unsigned bit = 0; bit < width; ++bit)
            if ((packet.lanes[(offset + bit) / 32]
                    & (std::uint32_t {1} << ((offset + bit) % 32))) != 0)
                value |= std::uint64_t {1} << bit;
        return value;
    }
    static bool zero(const C2cEndpointIcuPacket& packet,
        std::size_t begin, std::size_t end)
    {
        for (auto bit = begin; bit < end; ++bit)
            if ((packet.lanes[bit / 32] & (std::uint32_t {1} << (bit % 32))) != 0)
                return false;
        return true;
    }
    static Hemisphere decode_hemisphere(std::uint64_t value)
    {
        if (value >= hw::kHemispheres)
            throw std::out_of_range("C2C packet hemisphere is invalid");
        return static_cast<Hemisphere>(value);
    }
    static void validate_endpoint_reserved(const C2cEndpointIcuPacket& packet)
    {
        if (!zero(packet, 53, 96))
            throw std::invalid_argument("C2C endpoint packet has non-zero reserved bits");
        if (read(packet, 0, 2) != 0)
            throw std::invalid_argument("C2C endpoint packet has invalid ICU opcode");
        if (read(packet, 2, 1) == 0 && !zero(packet, 44, 53))
            throw std::invalid_argument("C2C TX packet carries RX routing fields");
        if (read(packet, 44, 1) == 0 && !zero(packet, 45, 53))
            throw std::invalid_argument("C2C RX packet carries disabled route fields");
    }
};

} // namespace ftlpu
