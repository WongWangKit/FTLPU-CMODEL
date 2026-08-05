#include "ftlpu/program/static_schedule.hpp"

#include <cassert>
#include <cstddef>
#include <stdexcept>

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

} // namespace

int main()
{
    ftlpu::program::StaticSchedule schedule;
    for (std::size_t row = 0; row < 128; ++row) {
        schedule.mem_at(
            17 + row,
            ftlpu::Hemisphere::West,
            9,
            ftlpu::MemInstruction::Read(
                100 + row * 8,
                ftlpu::StreamId::East(3)));
        schedule.mxm_at(
            31 + row,
            0,
            ftlpu::MxmControlInstruction::Compute(
                1, 8));
    }
    assert(schedule.active_queue_count() == 2);
    assert(schedule.last_cycle() == 158);

    const auto sections = schedule.sections("compressed");
    assert(sections.size() == 2);
    assert(sections[0].target
        == ftlpu::IcuLocation::Mem(
            ftlpu::Hemisphere::West, 9));
    assert(sections[0].packets.size() == 3);
    assert(
        ftlpu::isa::decode_icu_packet(
            sections[0].packets[0]).count
        == 17);
    assert(
        ftlpu::isa::decode_mem_packet(
            sections[0].packets[1]).address
        == ftlpu::MemLocalWordAddress13(100));
    const auto mem_repeat =
        ftlpu::isa::decode_icu_packet(
            sections[0].packets[2]);
    assert(mem_repeat.count == 127);
    assert(mem_repeat.address_stride == 8);

    assert(sections[1].target
        == ftlpu::IcuLocation::MxmCompute(0));
    assert(sections[1].packets.size() == 3);
    const auto mxm_repeat =
        ftlpu::isa::decode_icu_packet(
            sections[1].packets[2]);
    assert(mxm_repeat.count == 127);
    assert(mxm_repeat.address_stride == 0);

    ftlpu::program::StaticSchedule collision;
    collision.mem_at(
        4, 0, ftlpu::MemInstruction::Read(0, 0));
    collision.mem_at(
        4, 0, ftlpu::MemInstruction::Read(1, 0));
    assert(throws([&] {
        (void)collision.sections();
    }));
    return 0;
}
