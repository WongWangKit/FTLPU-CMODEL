#include "ftlpu/c2c/icu_instruction.hpp"
#include "ftlpu/system/icu.hpp"

#include <array>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

void endpoint_codec_round_trip()
{
    using namespace ftlpu;
    const auto original = C2cRxIcuInstruction::Receive(Hemisphere::East,
        3, 21, 9, 0x22aa,
        C2cMemNotifyRoute::Mem(Hemisphere::West, 51, 1));
    const auto packet = C2cIcuPacketCodec::encode(original);
    require((packet.lanes[0] & 3U) == 0,
        "C2C endpoint packet does not use the ICU Instruction envelope");
    const auto decoded = C2cIcuPacketCodec::decode_rx(packet);
    require(decoded.lane == 3 && decoded.fabric_stream == 21
            && decoded.vector_count == 9 && decoded.sync_tag == 0x22aa
            && decoded.notify.enabled
            && decoded.notify.hemisphere == Hemisphere::West
            && decoded.notify.mem_slice == 51 && decoded.notify.mem_bank == 1,
        "C2C RX packet changed a hardware field");
    auto corrupt = packet;
    corrupt.lanes[2] |= 1U << 31;
    try {
        (void)C2cIcuPacketCodec::decode_rx(corrupt);
        throw std::runtime_error("C2C RX packet accepted reserved bits");
    } catch (const std::invalid_argument&) {
    }

    const auto txOriginal = C2cTxIcuInstruction::Send(Hemisphere::West,
        5, 17, 7, 0x315a,
        C2cMemNotifyRoute::Mem(Hemisphere::East, 36, 1));
    const auto txPacket = C2cIcuPacketCodec::encode(txOriginal);
    const auto txDecoded = C2cIcuPacketCodec::decode_tx(txPacket);
    require(txDecoded.endpoint_hemisphere == Hemisphere::West
            && txDecoded.lane == 5 && txDecoded.fabric_stream == 17
            && txDecoded.vector_count == 7
            && txDecoded.sync_tag == 0x315a
            && txDecoded.notify.enabled
            && txDecoded.notify.hemisphere == Hemisphere::East
            && txDecoded.notify.mem_slice == 36
            && txDecoded.notify.mem_bank == 1,
        "C2C TX packet changed its MEM_READ_SYNC notify route");
}

void dma_codec_and_raw_queue()
{
    using namespace ftlpu;
    const auto original = C2cDmaIcuInstruction::Store(Hemisphere::West, 6,
        0x123456789abcdef0ULL, 257, 96, 0xcafe);
    const auto packet = C2cIcuPacketCodec::encode(original);
    require((packet.words[0].lanes[0] & 3U) == 0
            && (packet.words[1].lanes[2] & 0x80000000U) != 0,
        "C2C DMA packet does not carry fixed-word markers");
    const auto decoded = C2cIcuPacketCodec::decode_dma(packet);
    require(decoded.ddr4_address == original.ddr4_address
            && decoded.sync_tag == original.sync_tag
            && decoded.vector_count == original.vector_count,
        "C2C DMA packet changed a hardware field");
    require(decoded.to_legacy().vector_tag_base == 0,
        "C2C DMA packet encoded vector_tag_base");

    auto icu = InstructionControlUnit {};
    icu.enqueue_c2c_rx_raw(Hemisphere::East,
        C2cIcuPacketCodec::encode(C2cRxIcuInstruction::Receive(
            Hemisphere::East, 0, 1, 1)));
    require(icu.c2c_rx_iq(Hemisphere::East).imem_occupancy() == 1,
        "C2C endpoint packet did not occupy one physical i-MEM word");
    icu.append_c2c_dma_raw_word(Hemisphere::West, packet.words[0]);
    require(icu.c2c_dma_raw_word_pending(Hemisphere::West),
        "C2C DMA did not retain its first physical raw word");
    require(icu.c2c_dma_iq(Hemisphere::West).imem_occupancy() == 1,
        "C2C DMA header was staged outside physical i-MEM");
    icu.append_c2c_dma_raw_word(Hemisphere::West, packet.words[1]);
    require(icu.c2c_dma_iq(Hemisphere::West).imem_occupancy() == 2,
        "C2C DMA packet did not occupy two physical i-MEM words");
    const auto command = icu.c2c_dma_iq(Hemisphere::West).dispatch_next();
    require(command.has_value() && command->sync_tag == 0xcafe,
        "C2C DMA raw words did not decode into the engine command");
    require(icu.c2c_dma_iq(Hemisphere::West).issued_count() == 1,
        "C2C DMA packet issued as more than one ICU instruction");
}

void invalid_mem_location_does_not_alias_west_queue()
{
    using namespace ftlpu;
    constexpr std::size_t kTaggedEvent = 0x321;
    const auto westLocation = IcuLocation::Mem(Hemisphere::West, 0, 0);
    const auto westQueue = InstructionControlUnit::mem_queue(
        Hemisphere::West, 0, 0);
    const std::array invalidLocations {
        IcuLocation::Mem(Hemisphere::East, hw::kMemSliceColumns, 0),
        IcuLocation::Mem(Hemisphere::East,
            hw::kMemSliceColumns - 1, hw::kMemBanksPerSlice),
    };

    InstructionControlUnit icu;
    icu.enqueue_control(westLocation, IcuControlInstruction::Sync());
    icu.enqueue_control(
        westLocation, IcuControlInstruction::WaitEvent(kTaggedEvent));
    const auto westOccupancy = icu.mem_iq(westQueue).imem_occupancy();

    const auto requireOutOfRange = [](auto&& operation) {
        try {
            operation();
        } catch (const std::out_of_range&) {
            return;
        }
        throw std::runtime_error(
            "invalid MEM location did not throw std::out_of_range");
    };
    for (const auto& invalid : invalidLocations) {
        requireOutOfRange([&] {
            icu.enqueue_control(invalid, IcuControlInstruction::Notify());
        });
        requireOutOfRange([&] { icu.notify(invalid); });
        requireOutOfRange([&] { icu.notify_tagged(invalid, kTaggedEvent); });
        requireOutOfRange([&] {
            icu.notify_mem_synchronized(invalid, kTaggedEvent);
        });
    }

    auto& west = icu.mem_iq(westQueue);
    require(west.imem_occupancy() == westOccupancy,
        "invalid East MEM control aliased the West MEM i-MEM");
    static_cast<void>(west.tick());
    require(west.last_trace().action == IcuQueueAction::SyncWait,
        "invalid East MEM notify aliased the West MEM queue");
    icu.notify(westLocation);
    static_cast<void>(west.tick());
    require(west.last_trace().action == IcuQueueAction::SyncRelease,
        "valid West MEM notify did not release its queue");
    static_cast<void>(west.tick());
    require(west.last_trace().action == IcuQueueAction::EventWait,
        "invalid East tagged MEM notify aliased the West MEM queue");
}

} // namespace

int main()
{
    try {
        endpoint_codec_round_trip();
        dma_codec_and_raw_queue();
        invalid_mem_location_does_not_alias_west_queue();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "icu_c2c_packet_codec_test failed: " << error.what() << '\n';
        return 1;
    }
}
