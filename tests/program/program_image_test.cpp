#include "ftlpu/program/program.hpp"

#include <cassert>
#include <stdexcept>
#include <vector>

namespace {

template <typename Fn>
bool throws(Fn&& fn)
{
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

ftlpu::MemGlobalAddress24 address(std::size_t row)
{
    return ftlpu::MemGlobalAddress24::FromFields(
        0,
        7,
        ftlpu::MemLocalWordAddress13::FromFields(0, row)
            .slice_byte_address());
}

} // namespace

int main()
{
    std::vector<ftlpu::isa::EncodedInstructionPacket> payload;
    for (std::size_t packet = 0; packet < 79; ++packet) {
        payload.push_back(ftlpu::program::encode_packet(
            ftlpu::MemInstruction::Read(
                packet, ftlpu::StreamId::East(3))));
    }
    const auto section = ftlpu::program::build_chained_section(
        ftlpu::IcuLocation::Mem(7),
        ftlpu::StreamId::East(31),
        payload,
        "three finite-IQ IFetch blocks");
    assert(ftlpu::program::chained_block_count(payload.size()) == 2);
    assert(section.packets.size() == 80);
    assert(ftlpu::isa::decode_icu_packet(section.packets[0]).opcode
        == ftlpu::IcuControlOpcode::Fetch);
    assert(ftlpu::isa::decode_mem_packet(section.packets[1]).address
        == ftlpu::MemLocalWordAddress13(0));
    assert(ftlpu::isa::decode_mem_packet(section.packets[40]).address
        == ftlpu::MemLocalWordAddress13(39));

    ftlpu::ProgramImage image(ftlpu::ProgramImageHeader {
        ftlpu::ProgramImageHeader::kMagic,
        1,
        "layout test",
        "instruction and tensor metadata",
    });
    image.add_section(section);
    image.add_data_section(ftlpu::ProgramDataSection {
        "input",
        ftlpu::DmaPurpose::InputTensor,
        address(10),
        std::vector<std::uint8_t>(
            ftlpu::hw::kPhysicalVectorBytes, 0x5a),
        {1, ftlpu::hw::kPhysicalVectorBytes},
        "one vector",
    });
    const auto layout = ftlpu::ProgramSramLayout::Build(image, address(20));
    assert(layout.placements().size() == 2);
    assert(layout.data_placements().size() == 1);
    assert(layout.host_bytes().size()
        == 2 * ftlpu::hw::kIcuFetchBufferBytes
            + ftlpu::hw::kPhysicalVectorBytes);

    ftlpu::ProgramImage overlap;
    overlap.add_section(ftlpu::ProgramSection {
        ftlpu::IcuLocation::Mem(7),
        {ftlpu::program::padding_nop_packet()},
        0,
        "program",
    });
    overlap.add_data_section(ftlpu::ProgramDataSection {
        "overlap",
        ftlpu::DmaPurpose::InputTensor,
        address(30),
        std::vector<std::uint8_t>(
            ftlpu::hw::kPhysicalVectorBytes, 0),
        {},
        {},
    });
    assert(throws([&] {
        (void)ftlpu::ProgramSramLayout::Build(overlap, address(30));
    }));
    return 0;
}
