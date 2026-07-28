#include "ftlpu/program/program.hpp"

#include <cassert>
#include <cstddef>

int main()
{
    ftlpu::ProgramImage workload(
        ftlpu::ProgramImageHeader {
            ftlpu::ProgramImageHeader::kMagic,
            1,
            "autonomous launch builder test",
            "no functional program is directly enqueued",
        });
    workload.add_section(ftlpu::ProgramSection {
        ftlpu::IcuLocation::Mem(
            ftlpu::Hemisphere::West, 7),
        {
            ftlpu::program::encode_packet(
                ftlpu::IcuControlInstruction::Nop(5)),
            ftlpu::program::encode_packet(
                ftlpu::MemInstruction::Read(
                    4, ftlpu::StreamId::East(0))),
        },
        0,
        "west MEM schedule",
    });
    workload.add_section(ftlpu::ProgramSection {
        ftlpu::IcuLocation::MxmLoad(0),
        {
            ftlpu::program::encode_packet(
                ftlpu::IcuControlInstruction::Nop(9)),
            ftlpu::program::encode_packet(
                ftlpu::MxmControlInstruction::IW(0)),
        },
        0,
        "MXM load schedule",
    });
    workload.add_section(ftlpu::ProgramSection {
        ftlpu::IcuLocation::Vxm(3),
        {
            ftlpu::program::encode_packet(
                ftlpu::IcuControlInstruction::Nop(11)),
            ftlpu::program::encode_packet(
                ftlpu::VxmLaneAluInstruction {}),
        },
        0,
        "VXM schedule",
    });

    const auto program =
        ftlpu::program::AutonomousProgramBuilder::Build(
            workload);
    assert(program.image.sections().size() == 5);
    assert(program.layout.placements().size() == 5);
    assert(program.preamble.entries.size() == 4);
    assert(
        program.preamble.mem_local_bootstraps.size()
        == 3);
    assert(program.schedule_epoch_cycle
        > ftlpu::hw::kIcuBarrierLatencyCycles);

    // Non-MEM target programs are stored in their loader SRAM, while every
    // generated MEM loader program is physically local to its own target IQ.
    for (const auto& placement :
         program.layout.placements()) {
        if (placement.target.kind
            == ftlpu::IcuLocationKind::Mem) {
            assert(
                placement.target.unit
                == placement.memory_address.hemisphere());
            assert(
                placement.target.index
                == placement.memory_address.mem_slice());
        }
    }
    return 0;
}
