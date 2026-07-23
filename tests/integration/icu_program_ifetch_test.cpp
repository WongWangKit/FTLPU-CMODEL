#include "ftlpu/dma/dma.hpp"
#include "ftlpu/program/program.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <variant>
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

    // The first fetched block contains Fetch #2 itself. Nothing after the
    // initial bootstrap Fetch is directly inserted into the MXM IQ by C++.
    const auto kInstructionStream = ftlpu::StreamId::East(31);
    ftlpu::ProgramImage image;
    ftlpu::ProgramSection section {
        ftlpu::IcuLocation::MxmLoad(0), {}, 0, "two MXM load blocks"};
    section.packets.push_back(ftlpu::program::encode_packet(
        ftlpu::IcuControlInstruction::Fetch(kInstructionStream)));
    for (std::size_t packet = 1;
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

    // This is the MEM ICU's own program. It is DMA-loaded into the same local
    // MEM slice and later fetched without injecting Read instructions.
    ftlpu::ProgramImage mem_loader_image;
    mem_loader_image.add_section(ftlpu::build_mem_icu_loader_section(
        40,
        kInstructionStream,
        {
            {layout.placements()[0], 0, true},
            {layout.placements()[1], 35, false},
        }));
    const auto mem_loader_layout = ftlpu::ProgramSramLayout::Build(
        mem_loader_image,
        ftlpu::MemGlobalAddress24::FromFields(
            0,
            40,
            ftlpu::MemLocalWordAddress13::FromFields(1, 100)
                .slice_byte_address()));
    assert(mem_loader_layout.placements().size() == 1);
    assert(mem_loader_layout.placements()[0].target
        == ftlpu::IcuLocation::Mem(40));

    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    ftlpu::GlobalMemoryAddressSpace global_memory;
    global_memory.bind_hemisphere(0, system->mem().memory_model());
    ftlpu::HostMemorySpace host;
    const auto program_buffer = host.register_buffer(layout.host_bytes());
    const auto mem_loader_buffer =
        host.register_buffer(mem_loader_layout.host_bytes());
    ftlpu::DmaEngine dma(host, global_memory);
    auto descriptors = layout.make_dma_descriptors(program_buffer);
    const auto loader_descriptors =
        mem_loader_layout.make_dma_descriptors(mem_loader_buffer);
    descriptors.insert(
        descriptors.end(), loader_descriptors.begin(), loader_descriptors.end());
    assert(descriptors.size() == 3);
    for (const auto& descriptor : descriptors) {
        assert(descriptor.purpose == ftlpu::DmaPurpose::Program);
        assert(descriptor.vector_count == 2);
        const auto local = descriptor.memory_address.slice_byte_address()
                               .local_word_address();
        assert(local.word() + descriptor.vector_count
            <= ftlpu::hw::kSramWordsPerBank);
        assert(dma.enqueue(descriptor).valid());
    }
    while (!dma.idle()) {
        assert(dma.tick());
    }
    assert(dma.completion_history().size() == 3);

    auto preamble = ftlpu::BootstrapPreambleBuilder::ForAutonomousMem(
        mem_loader_layout.placements()[0],
        layout.placements()[0],
        kInstructionStream);
    assert(preamble.mem_local_bootstraps.size() == 1);
    assert(preamble.entries.size() == 2);
    for (const auto& entry : preamble.entries) {
        assert(!std::holds_alternative<ftlpu::MemInstruction>(entry.instruction));
        assert(!std::holds_alternative<ftlpu::MxmControlInstruction>(entry.instruction));
    }
    ftlpu::load_bootstrap_preamble(system->icu(), preamble);
    assert(system->icu().mem_local_fetch_active(40));
    assert(system->icu().mem_iq(40).iq_occupancy() == 0);

    auto& target_iq = system->icu().mxm_iq(
        0, ftlpu::InstructionControlUnit::MxmIcuPort::Load);
    bool mem_local_bootstrap_completed = false;
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

        if (cycle == 0) {
            const auto* state = system->icu().mem_local_fetch_state(40);
            assert(state != nullptr && state->next_vector == 1);
            assert(system->icu().mem_iq(40).iq_occupancy() == 0);
        }
        if (cycle == 1) {
            mem_local_bootstrap_completed =
                !system->icu().mem_local_fetch_active(40)
                && system->icu().mem_iq(40).iq_occupancy()
                    == ftlpu::hw::kIcuFetchPackets;
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
            assert(target_iq.iq_occupancy() >= 41);
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

    assert(mem_local_bootstrap_completed);
    assert(sync_block_observed);
    assert(sync_release_cycle >= ftlpu::hw::kIcuBarrierLatencyCycles);
    assert(first_fetch_committed);
    assert(second_fetch_started);
    assert(second_fetch_committed);
    assert(fetched_mxm_dispatched);
    return 0;
}
