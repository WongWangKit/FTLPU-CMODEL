#include "ftlpu/dma/dma.hpp"
#include "ftlpu/program/program.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace {

void provide_mxm0_weight_streams(ftlpu::TspSliceSystem& system)
{
    auto& fabric = system.mem().stream_fabric();
    constexpr auto kColumn = ftlpu::hw::kMemBoundaryStreamRegisterColumns - 1;
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t stream = 0;
             stream < ftlpu::hw::kMxmLoadStreamsPerCycle;
             ++stream) {
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                fabric.initialize_cell(
                    kColumn,
                    tile,
                    lane,
                    ftlpu::StreamId::East(stream),
                    ftlpu::StreamCell::Valid(
                        static_cast<std::uint8_t>(1 + stream),
                        lane + 1 == ftlpu::hw::kLanesPerTile));
            }
        }
    }
}

} // namespace

int main()
{
    static_assert(
        ftlpu::hw::kIcuFetchPackets
            * ftlpu::hw::kEncodedInstructionPacketBytes
        == ftlpu::hw::kIcuFetchBufferBytes);
    static_assert(
        ftlpu::hw::kIcuFetchVectorCount
            * ftlpu::hw::kPhysicalVectorBytes
        == ftlpu::hw::kIcuFetchBufferBytes);

    // A short section is padded with encoded legal NOP packets, and a block
    // beginning at bank0:word4095 is explicitly moved to bank1:word0.
    {
        ftlpu::ProgramImage short_image;
        short_image.add_section(ftlpu::ProgramSection {
            ftlpu::IcuLocation::MxmLoad(0),
            {ftlpu::program::encode_packet(
                ftlpu::MxmControlInstruction::IW(0))},
            0,
            "padding test",
        });
        const auto short_layout = ftlpu::ProgramSramLayout::Build(
            short_image,
            ftlpu::MemGlobalAddress24::FromFields(
                0,
                40,
                ftlpu::MemLocalWordAddress13::FromFields(0, 4095)
                    .slice_byte_address()));
        assert(short_layout.host_bytes().size()
            == ftlpu::hw::kIcuFetchBufferBytes);
        const auto local = short_layout.placements().front().memory_address
                               .slice_byte_address().local_word_address();
        assert(local.bank() == 1 && local.word() == 0);
        ftlpu::isa::EncodedInstructionPacket padding{};
        std::copy_n(
            short_layout.host_bytes().begin()
                + ftlpu::hw::kEncodedInstructionPacketBytes,
            ftlpu::hw::kEncodedInstructionPacketBytes,
            padding.bytes.begin());
        const auto nop = ftlpu::isa::decode_icu_packet(padding);
        assert(nop.opcode == ftlpu::IcuControlOpcode::Nop);
        assert(nop.count == 0);
    }

    // Two complete blocks keep the MXM load IQ supplied while the second
    // 640-byte Ifetch is in flight.
    ftlpu::ProgramImage image;
    ftlpu::ProgramSection section {
        ftlpu::IcuLocation::MxmLoad(0), {}, 0, "two MXM load blocks"};
    for (std::size_t packet = 0;
         packet < 2 * ftlpu::hw::kIcuFetchPackets;
         ++packet) {
        section.packets.push_back(ftlpu::program::encode_packet(
            ftlpu::MxmControlInstruction::IW(packet < 40 ? 0 : 1)));
    }
    image.add_section(std::move(section));

    const auto base = ftlpu::MemGlobalAddress24::FromFields(
        0,
        40,
        ftlpu::MemLocalWordAddress13::FromFields(0, 4094)
            .slice_byte_address());
    const auto layout = ftlpu::ProgramSramLayout::Build(image, base);
    assert(layout.placements().size() == 2);
    const auto first_local = layout.placements()[0].memory_address
                                 .slice_byte_address().local_word_address();
    const auto second_local = layout.placements()[1].memory_address
                                  .slice_byte_address().local_word_address();
    assert(first_local.bank() == 0 && first_local.word() == 4094);
    assert(second_local.bank() == 1 && second_local.word() == 0);

    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    ftlpu::GlobalMemoryAddressSpace global_memory;
    global_memory.bind_hemisphere(0, system->mem().memory_model());
    ftlpu::HostMemorySpace host;
    const auto program_buffer = host.register_buffer(layout.host_bytes());
    ftlpu::DmaEngine dma(host, global_memory);
    const auto descriptors = layout.make_dma_descriptors(program_buffer);
    assert(descriptors.size() == 2);
    assert(descriptors[0].purpose == ftlpu::DmaPurpose::Program);
    assert(descriptors[0].vector_count == 2);
    assert(descriptors[1].vector_count == 2);
    for (const auto& descriptor : descriptors) {
        const auto local = descriptor.memory_address.slice_byte_address()
                               .local_word_address();
        assert(local.word() + descriptor.vector_count
            <= ftlpu::hw::kSramWordsPerBank);
        assert(dma.enqueue(descriptor).valid());
    }
    while (!dma.idle()) {
        assert(dma.tick());
    }
    assert(dma.completion_history().size() == 2);

    const auto kInstructionStream = ftlpu::StreamId::East(31);
    auto preamble = ftlpu::BootstrapPreambleBuilder::ForProgramBlock(
        layout.placements()[0],
        kInstructionStream,
        ftlpu::IcuLocation::Mem(0));
    // One resident instruction behind Fetch proves that frontend collection
    // and existing-IQ dispatch overlap. It remains bootstrap state, not part
    // of the DMA-loaded program body.
    preamble.entries.insert(
        preamble.entries.begin() + 3,
        ftlpu::BootstrapEntry {
            ftlpu::IcuLocation::MxmLoad(0),
            ftlpu::MxmControlInstruction::IW(1)});
    ftlpu::load_bootstrap_preamble(system->icu(), preamble);

    // Schedule the second SRAM block to appear exactly when Fetch #2 reaches
    // the head. The first decoded block is already behind Fetch #2, so the IQ
    // does not drain while the second frontend transaction runs.
    system->icu().enqueue_mxm_control(
        0,
        ftlpu::InstructionControlUnit::MxmIcuPort::Load,
        ftlpu::IcuControlInstruction::Fetch(kInstructionStream));
    system->icu().enqueue_mem_control(
        40, ftlpu::IcuControlInstruction::Nop(35));
    system->icu().enqueue_mem(
        40, ftlpu::MemInstruction::Read(second_local, kInstructionStream));
    system->icu().enqueue_mem(
        40,
        ftlpu::MemInstruction::Read(
            second_local.next_word(), kInstructionStream));

    auto& target_iq = system->icu().mxm_iq(
        0, ftlpu::InstructionControlUnit::MxmIcuPort::Load);
    bool overlap_observed = false;
    bool sync_block_observed = false;
    bool first_fetch_committed = false;
    bool second_fetch_started = false;
    bool second_fetch_committed = false;
    bool fetched_mxm_dispatched = false;
    std::size_t sync_release_cycle = 0;

    for (std::size_t cycle = 0; cycle < 120; ++cycle) {
        const auto was_active = target_iq.fetch_active();
        const auto old_fetch_count = target_iq.fetch_count();
        provide_mxm0_weight_streams(*system);
        system->tick(ftlpu::TspSliceSystem::LogSinks {});

        if (cycle == 1) {
            overlap_observed = target_iq.fetch_active()
                && target_iq.iq_occupancy() == 2;
        }
        sync_block_observed = sync_block_observed
            || target_iq.blocked_on_sync();
        if (sync_block_observed && !target_iq.blocked_on_sync()
            && sync_release_cycle == 0) {
            sync_release_cycle = cycle;
        }

        if (was_active && !target_iq.fetch_active()
            && old_fetch_count == 1) {
            first_fetch_committed = true;
            assert(target_iq.iq_occupancy() >= 42);
        }
        if (target_iq.fetch_count() == 2 && target_iq.fetch_active()) {
            second_fetch_started = true;
            assert(target_iq.iq_occupancy() > 0);
        }
        if (second_fetch_started && was_active
            && !target_iq.fetch_active()
            && old_fetch_count == 2) {
            second_fetch_committed = true;
            // More than one full block remains: Fetch #2 completed before
            // the first block could drain the IQ.
            assert(target_iq.iq_occupancy() > ftlpu::hw::kIcuFetchPackets);
        }
        fetched_mxm_dispatched = fetched_mxm_dispatched
            || system->mxm_unit(0).control().loaded_cell(0, 0, 0);
        if (second_fetch_committed && fetched_mxm_dispatched) {
            break;
        }
    }

    assert(overlap_observed);
    assert(sync_block_observed);
    assert(sync_release_cycle >= ftlpu::hw::kIcuBarrierLatencyCycles);
    assert(first_fetch_committed);
    assert(second_fetch_started);
    assert(second_fetch_committed);
    assert(fetched_mxm_dispatched);
    return 0;
}
