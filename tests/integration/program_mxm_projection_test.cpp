#include "ftlpu/dma/dma.hpp"
#include "ftlpu/program/program.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <cassert>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kWeightStreams =
    ftlpu::hw::kMxmLoadStreamsPerCycle;
constexpr std::size_t kBlocks =
    ftlpu::hw::kMxmSupercellsPerPlane;
constexpr std::size_t kActivationMemSlice = 32;
constexpr std::size_t kOutputSliceBase = 40;
constexpr std::size_t kOutputRow = 500;
// Select the first east-hemisphere MXM in either configuration. GroqLike has
// no west MXMs (index 0); TransformerEval has two (east starts at index 2).
constexpr std::size_t kMxm = ftlpu::hw::kWestMxmCount;
constexpr std::size_t kFirstIwCycle = 43;
constexpr std::size_t kComputeCycle = 83;
constexpr std::size_t kFirstOutputWriteCycle =
    kComputeCycle + kBlocks;

ftlpu::MemGlobalAddress24 address(
    std::size_t mem_slice,
    std::size_t bank,
    std::size_t row)
{
    return ftlpu::MemGlobalAddress24::FromFields(
        ftlpu::hemisphere_index(
            ftlpu::Hemisphere::East),
        mem_slice,
        ftlpu::MemLocalWordAddress13::FromFields(
            bank, row).slice_byte_address());
}

std::vector<std::uint8_t> filled_vectors(
    std::size_t vector_count,
    std::uint8_t value)
{
    return std::vector<std::uint8_t>(
        vector_count
            * ftlpu::hw::kPhysicalVectorBytes,
        value);
}

ftlpu::ProgramImage make_workload()
{
    auto image = ftlpu::ProgramImage(
        ftlpu::ProgramImageHeader {
            ftlpu::ProgramImageHeader::kMagic,
            1,
            "one-row all-ones MXM projection",
            "static schedule -> ProgramImage -> DMA -> SRAM -> "
            "bootstrap -> IFetch -> MEM/MXM -> MEM",
        });
    auto schedule =
        ftlpu::program::StaticSchedule {};

    for (std::size_t stream = 0;
         stream < kWeightStreams;
         ++stream) {
        const auto group =
            stream / ftlpu::hw::kMemSlicesPerGroup;
        const auto east_latency =
            (ftlpu::hw::kStreamRegisterColumns - 1)
            - group;
        const auto first_read =
            kFirstIwCycle - east_latency - 1;
        for (std::size_t block = 0;
             block < kBlocks;
             ++block) {
            schedule.mem_at(
                first_read + block,
                stream,
                ftlpu::MemInstruction::Read(
                    ftlpu::MemLocalWordAddress13::
                        FromFields(
                            0,
                            kBlocks - 1 - block),
                    ftlpu::StreamId::East(stream)));
        }
    }
    for (std::size_t block = 0;
         block < kBlocks;
         ++block) {
        schedule.mxm_at(
            kFirstIwCycle + block,
            kMxm,
            ftlpu::MxmControlInstruction::IW(0));
    }

    const auto activation_group =
        kActivationMemSlice
        / ftlpu::hw::kMemSlicesPerGroup;
    const auto activation_latency =
        (ftlpu::hw::kStreamRegisterColumns - 1)
        - activation_group;
    schedule.mem_at(
        kComputeCycle - activation_latency - 1,
        kActivationMemSlice,
        ftlpu::MemInstruction::Read(
            ftlpu::MemLocalWordAddress13::
                FromFields(0, 0),
            ftlpu::StreamId::East(16)));
    if (ftlpu::hw::kMxmActivationBytesPerValue
        == 2) {
        schedule.mem_at(
            kComputeCycle - activation_latency - 1,
            kActivationMemSlice + 1,
            ftlpu::MemInstruction::Read(
                ftlpu::MemLocalWordAddress13::
                    FromFields(0, 0),
                ftlpu::StreamId::East(17)));
    }
    schedule.mxm_at(
        kComputeCycle,
        kMxm,
        ftlpu::MxmControlInstruction::
            ComputeToAccumulator(
                0,
                0,
                16,
                0,
                false,
                true,
                true));

    for (std::size_t byte = 0;
         byte < sizeof(std::uint32_t);
         ++byte) {
        schedule.mem_at(
            kFirstOutputWriteCycle,
            kOutputSliceBase + byte,
            ftlpu::MemInstruction::Write(
                ftlpu::MemLocalWordAddress13::
                    FromFields(0, kOutputRow),
                ftlpu::StreamId::West(byte)));
    }
    schedule.append_to(image, "projection");

    for (std::size_t stream = 0;
         stream < kWeightStreams;
         ++stream) {
        const auto weight_byte =
            ftlpu::hw::kMxmWeightBytesPerValue == 1
            ? std::uint8_t {1}
            : (stream % 2 == 0
                   ? std::uint8_t {0x00}
                   : std::uint8_t {0x3c});
        image.add_data_section(
            ftlpu::ProgramDataSection {
                "weight_stream_"
                    + std::to_string(stream),
                ftlpu::DmaPurpose::Model,
                address(stream, 0, 0),
                filled_vectors(
                    kBlocks, weight_byte),
                {
                    kBlocks,
                    ftlpu::hw::kPhysicalVectorBytes,
                },
                "all-one weights",
            });
    }
    image.add_data_section(
        ftlpu::ProgramDataSection {
            "activation_low",
            ftlpu::DmaPurpose::InputTensor,
            address(
                kActivationMemSlice, 0, 0),
            filled_vectors(
                1,
                ftlpu::hw::
                            kMxmActivationBytesPerValue
                        == 1
                    ? std::uint8_t {1}
                    : std::uint8_t {0x00}),
            {1, ftlpu::hw::kMxmRows},
            "one all-one activation row, low byte",
        });
    if (ftlpu::hw::kMxmActivationBytesPerValue
        == 2) {
        image.add_data_section(
            ftlpu::ProgramDataSection {
                "activation_high",
                ftlpu::DmaPurpose::InputTensor,
                address(
                    kActivationMemSlice + 1,
                    0,
                    0),
                filled_vectors(1, 0x3c),
                {1, ftlpu::hw::kMxmRows},
                "FP16 one activation high byte",
            });
    }
    return image;
}

} // namespace

int main()
try
{
    const auto launched =
        ftlpu::program::AutonomousProgramBuilder::
            Build(make_workload());

    auto system =
        std::make_unique<ftlpu::TspSliceSystem>();
    ftlpu::GlobalMemoryAddressSpace memory;
    for (std::size_t hemisphere = 0;
         hemisphere < ftlpu::hw::kHemispheres;
         ++hemisphere) {
        memory.bind_hemisphere(
            hemisphere,
            system
                ->mem(static_cast<ftlpu::Hemisphere>(
                    hemisphere))
                .memory_model());
    }
    ftlpu::HostMemorySpace host;
    const auto host_buffer =
        host.register_buffer(
            launched.layout.host_bytes());
    ftlpu::DmaEngine dma(host, memory);
    const auto descriptors =
        launched.layout.make_dma_descriptors(
            host_buffer);
    for (const auto& descriptor : descriptors) {
        assert(dma.enqueue(descriptor).valid());
    }
    while (!dma.idle()) {
        assert(dma.tick());
    }
    std::size_t completions = 0;
    while (dma.completion_ready()) {
        assert(dma.pop_completion().id.valid());
        ++completions;
    }
    assert(completions == descriptors.size());

    ftlpu::load_bootstrap_preamble(
        system->icu(), launched.preamble);

    const auto cycle_limit =
        launched.schedule_epoch_cycle + 150;
    for (std::size_t cycle = 0;
         cycle < cycle_limit;
         ++cycle) {
        try {
            system->tick({});
        } catch (const std::exception& error) {
            std::cerr
                << "autonomous projection failed at cycle "
                << cycle << ": " << error.what()
                << '\n';
            system->icu().print_diagnostic_status(
                std::cerr);
            return 1;
        }
    }

    assert(
        system->icu()
                .mxm_iq(
                    kMxm,
                    ftlpu::InstructionControlUnit::
                        MxmIcuPort::Load)
                .fetch_count()
            == 1);
    assert(
        system->icu()
                .mxm_iq(
                    kMxm,
                    ftlpu::InstructionControlUnit::
                        MxmIcuPort::Compute)
                .fetch_count()
            == 1);
    for (std::size_t tile = 0;
         tile < kBlocks;
         ++tile) {
        for (std::size_t column = 0;
             column < kBlocks;
             ++column) {
            assert(
                system->mxm_unit(kMxm)
                    .control()
                    .loaded_cell(
                        0, tile, column));
        }
    }

    for (std::size_t tile = 0;
         tile < ftlpu::hw::kTileRows;
         ++tile) {
        for (std::size_t lane = 0;
             lane < ftlpu::hw::kLanesPerTile;
             ++lane) {
            std::uint32_t value = 0;
            for (std::size_t byte = 0;
                 byte < sizeof(value);
                 ++byte) {
                value |=
                    static_cast<std::uint32_t>(
                        system->mem()
                            .memory_model()
                            .sram_lane_byte(
                                kOutputSliceBase
                                    + byte,
                                tile,
                                ftlpu::
                                    MemLocalWordAddress13::
                                        FromFields(
                                            0,
                                            kOutputRow),
                                lane))
                    << (8 * byte);
            }
            const auto expected =
                ftlpu::hw::kMxmWeightBytesPerValue == 2
                ? std::bit_cast<std::uint32_t>(
                      static_cast<float>(
                          ftlpu::hw::kMxmRows))
                : static_cast<std::uint32_t>(
                      ftlpu::hw::kMxmRows);
            if (value != expected) {
                std::cerr
                    << "projection result mismatch"
                    << " tile=" << tile
                    << " lane=" << lane
                    << " actual=" << value
                    << " expected="
                    << expected
                    << '\n';
                system->icu()
                    .print_diagnostic_status(
                        std::cerr);
                return 1;
            }
        }
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr
        << "program_mxm_projection_test failed: "
        << error.what() << '\n';
    return 1;
}
