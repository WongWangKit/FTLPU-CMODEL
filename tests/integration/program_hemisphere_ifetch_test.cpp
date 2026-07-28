#include "ftlpu/dma/dma.hpp"
#include "ftlpu/program/program.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

namespace {

ftlpu::MemGlobalAddress24 address(
    ftlpu::Hemisphere hemisphere,
    std::size_t mem_slice,
    std::size_t bank,
    std::size_t row)
{
    return ftlpu::MemGlobalAddress24::FromFields(
        ftlpu::hemisphere_index(hemisphere),
        mem_slice,
        ftlpu::MemLocalWordAddress13::FromFields(bank, row)
            .slice_byte_address());
}

} // namespace

int main()
{
    constexpr auto kSide = ftlpu::Hemisphere::West;
    constexpr std::size_t kSourceSlice = 0;
    constexpr std::size_t kDestinationSlice = 4;
    constexpr std::size_t kDataRow = 0;
    constexpr std::size_t kOutputRow = 17;
    const auto stream = ftlpu::StreamId::East(0);

    ftlpu::ProgramImage image(ftlpu::ProgramImageHeader {
        ftlpu::ProgramImageHeader::kMagic,
        1,
        "west hemisphere autonomous MEM copy",
        "host -> DMA -> west SRAM -> local IFetch -> MEM/SR -> west SRAM",
    });
    image.add_section(ftlpu::ProgramSection {
        ftlpu::IcuLocation::Mem(kSide, kSourceSlice),
        {
            ftlpu::program::encode_packet(
                ftlpu::MemInstruction::Read(kDataRow, stream)),
        },
        0,
        "west source MEM ICU",
    });
    image.add_section(ftlpu::ProgramSection {
        ftlpu::IcuLocation::Mem(kSide, kDestinationSlice),
        {
            ftlpu::program::encode_packet(
                // The source Read first reaches group 0's compatibility
                // boundary and then advances through one physical SR hop to
                // group 1.  Delay the destination by those two cycles.
                ftlpu::IcuControlInstruction::Nop(2)),
            ftlpu::program::encode_packet(
                ftlpu::MemInstruction::Write(kOutputRow, stream)),
        },
        0,
        "west destination MEM ICU",
    });

    auto input = std::vector<std::uint8_t>(
        ftlpu::hw::kPhysicalVectorBytes);
    for (std::size_t byte = 0; byte < input.size(); ++byte) {
        input[byte] =
            static_cast<std::uint8_t>((byte * 13 + 7) & 0xffu);
    }
    image.add_data_section(ftlpu::ProgramDataSection {
        "west_input",
        ftlpu::DmaPurpose::InputTensor,
        address(kSide, kSourceSlice, 0, kDataRow),
        input,
        {ftlpu::hw::kTileRows, ftlpu::hw::kLanesPerTile},
        "distinct bytes prove hemisphere isolation",
    });

    const std::vector section_bases {
        address(kSide, kSourceSlice, 1, 100),
        address(kSide, kDestinationSlice, 1, 100),
    };
    const auto layout =
        ftlpu::ProgramSramLayout::BuildAtSectionBases(
            image, section_bases);

    auto system = std::make_unique<ftlpu::TspSliceSystem>();
    ftlpu::GlobalMemoryAddressSpace memory;
    memory.bind_hemisphere(
        ftlpu::hemisphere_index(ftlpu::Hemisphere::East),
        system->mem(ftlpu::Hemisphere::East).memory_model());
    memory.bind_hemisphere(
        ftlpu::hemisphere_index(ftlpu::Hemisphere::West),
        system->mem(ftlpu::Hemisphere::West).memory_model());
    ftlpu::HostMemorySpace host;
    const auto buffer =
        host.register_buffer(layout.host_bytes());
    ftlpu::DmaEngine dma(host, memory);
    for (const auto& descriptor :
         layout.make_dma_descriptors(buffer)) {
        assert(dma.enqueue(descriptor).valid());
    }
    while (!dma.idle()) {
        assert(dma.tick());
    }
    while (dma.completion_ready()) {
        (void)dma.pop_completion();
    }

    ftlpu::BootstrapPreamble preamble;
    for (const auto& placement : layout.placements()) {
        preamble.mem_local_bootstraps.push_back({
            placement.target.index,
            placement.memory_address
                .slice_byte_address()
                .local_word_address(),
            kSide,
        });
    }
    ftlpu::load_bootstrap_preamble(
        system->icu(), preamble);
    std::size_t west_reads = 0;
    std::size_t west_writes = 0;
    for (std::size_t cycle = 0;
         cycle < ftlpu::hw::kTileRows + 8;
         ++cycle) {
        system->tick({});
        for (const auto& transfer :
             system->mem(kSide)
                 .memory_model()
                 .executed_transfers()) {
            if (transfer.mem_slice == kSourceSlice
                && transfer.kind
                    == ftlpu::MemArrayModel::MemTransfer::Kind::
                        LoadSramToStream) {
                ++west_reads;
            }
            if (transfer.mem_slice == kDestinationSlice
                && transfer.kind
                    == ftlpu::MemArrayModel::MemTransfer::Kind::
                        StoreStreamToSram) {
                ++west_writes;
            }
        }
    }
    if (west_reads != ftlpu::hw::kTileRows
        || west_writes != ftlpu::hw::kTileRows) {
        std::cerr << "unexpected west transfer counts: reads="
                  << west_reads << " writes=" << west_writes
                  << " expected=" << ftlpu::hw::kTileRows << '\n';
        system->icu().print_diagnostic_status(std::cerr);
        return 1;
    }

    for (std::size_t tile = 0;
         tile < ftlpu::hw::kTileRows;
         ++tile) {
        for (std::size_t lane = 0;
             lane < ftlpu::hw::kLanesPerTile;
             ++lane) {
            const auto offset =
                tile * ftlpu::hw::kLanesPerTile + lane;
            const auto actual =
                system->mem(kSide).sram_lane_byte(
                    kDestinationSlice,
                    tile,
                    kOutputRow,
                    lane);
            if (actual != input[offset]) {
                std::cerr << "west copy mismatch tile=" << tile
                          << " lane=" << lane
                          << " actual=" << static_cast<unsigned>(actual)
                          << " expected="
                          << static_cast<unsigned>(input[offset])
                          << '\n';
                system->icu().print_diagnostic_status(std::cerr);
                return 1;
            }
            assert(
                system->mem(ftlpu::Hemisphere::East)
                    .sram_lane_byte(
                        kDestinationSlice,
                        tile,
                        kOutputRow,
                        lane)
                == 0);
        }
    }
    assert(
        system->icu()
                .mem_iq(
                    ftlpu::InstructionControlUnit::mem_queue(
                        kSide, kSourceSlice))
                .fetch_count()
            == 1);
    return 0;
}
