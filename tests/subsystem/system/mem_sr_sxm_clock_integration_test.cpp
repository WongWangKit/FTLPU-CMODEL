#include "ftlpu/system/tsp_slice_system.hpp"
#include "../../support/sxm_macro_fixture.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

ftlpu::SxmInstruction::StreamList stream_range(
    std::size_t first, std::size_t count)
{
    ftlpu::SxmInstruction::StreamList streams;
    streams.reserve(count);
    for (std::size_t index = 0; index < count; ++index)
        streams.push_back(ftlpu::SxmStreamId {first + index});
    return streams;
}

bool has_state_cell(
    const ftlpu::StreamRegisterFabric::HardwareInterfaceView& view,
    std::size_t column,
    std::size_t tile,
    std::size_t lane,
    ftlpu::StreamId stream)
{
    return std::any_of(view.state_valid_o.begin(), view.state_valid_o.end(),
        [&](const auto& cell) {
            return cell.column == column && cell.tile == tile
                && cell.lane == lane && cell.stream == stream;
        });
}

} // namespace

int main()
{
    try {
        using ftlpu::Hemisphere;
        using ftlpu::MemArrayModel;
        using ftlpu::MemInstruction;
        using ftlpu::StreamCell;
        using ftlpu::StreamId;
        using ftlpu::TspSliceSystem;

        // RegisteredR1: cycle A samples SRAM, cycle B stages the registered
        // response into SR Next, and cycle C observes that committed SR state.
        auto system = std::make_unique<TspSliceSystem>();
        auto& mem = system->mem_array(Hemisphere::East);
        const auto stream = StreamId::East(0);
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane)
            mem.set_sram_lane_byte(0, 0, 9, lane,
                static_cast<std::uint8_t>(0x40 + lane));
        mem.enqueue_instruction(0, MemInstruction::Read(9, stream));

        system->tick(TspSliceSystem::LogSinks {}); // A
        const auto& sample = mem.hardware_interface_view();
        require(sample.bank_issue_valid_i.size() == 1,
            "MEM native issue was not visible at sample edge A");
        require(sample.bank_issue_i.size() == 1,
            "MEM native issue payload was not encoded");
        require(sample.mem_producer_valid_o.empty(),
            "RegisteredR1 drove SR in the SRAM sampling cycle");
        require(!system->stream_fabric(Hemisphere::East)
                    .cell(1, 0, 0, stream).valid,
            "MEM response became SR Current before edge B");

        system->tick(TspSliceSystem::LogSinks {}); // B
        const auto& response = mem.hardware_interface_view();
        require(response.mem_producer_valid_o.size() == 1,
            "registered MEM response was not driven at edge B");
        require(response.mem_producer_boundary_o.front() == 1,
            "MEM response used the wrong SR boundary");
        require(response.mem_producer_data_o.front()[0] == 0x40,
            "MEM response data changed before SR commit");
        require(system->stream_fabric(Hemisphere::East)
                    .cell(1, 0, 0, stream).valid,
            "MEM response did not commit into SR Current at edge B");

        system->tick(TspSliceSystem::LogSinks {}); // C
        const auto& sr_view = system->stream_fabric(Hemisphere::East)
                                  .hardware_interface_view();
        require(has_state_cell(sr_view, 1, 0, 0, stream),
            "SR PRE_POSEDGE view did not sample edge-B Current");
        require(system->stream_fabric(Hemisphere::East)
                    .cell(2, 0, 0, stream).valid,
            "passive SR route did not commit one column at edge C");
        require(system->cycle() == 3 && system->phase_count() == 12,
            "global four-phase clock advanced more than once per tick");

        // Production TspSliceSystem issue path: ICU issues one canonical
        // 416-bit Transpose command, SXM consumes SR Current, and no HIGH
        // callback repeats the capture.
        auto sxm_system = std::make_unique<TspSliceSystem>();
        const auto input_column = sxm_system->sxm_unit(Hemisphere::East)
                                      .ports().input_column(
                                          ftlpu::StreamDirection::East);
        for (std::size_t packed = 0; packed < 16; ++packed) {
            for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
                sxm_system->stream_fabric(Hemisphere::East).initialize_cell(
                    input_column, 0, lane, StreamId::East(packed),
                    StreamCell::Valid(
                        static_cast<std::uint8_t>(packed * 8 + lane)));
            }
        }
        ftlpu::test::SxmMacroFixture sxm_program;
        sxm_program.add(0, 1, 1,
            ftlpu::SxmInstruction::Transpose(
                stream_range(0, 16), stream_range(16, 16)));
        sxm_program.load(
            sxm_system->icu().sxm_transpose_iq(Hemisphere::East), true);
        sxm_system->icu().prime_raw_macro_frontends();
        sxm_system->tick(TspSliceSystem::LogSinks {});

        const auto& sxm_view = sxm_system->sxm_unit(Hemisphere::East)
                                   .hardware_interface_view();
        require(sxm_view.transpose_issue_valid_i,
            "SXM canonical Transpose issue interface was not asserted");
        require(sxm_view.east_inner_valid_i.size() == 16,
            "SXM did not capture all sixteen selected input streams");
        require(sxm_system->stream_fabric(Hemisphere::East)
                    .hardware_interface_view().consume_i.size()
                == 16 * ftlpu::hw::kLanesPerTile,
            "SXM consume interface was not sampled lane-accurately");
        require(sxm_system->sxm_unit(Hemisphere::East).cycle() == 1,
            "SXM state advanced more than once in one physical cycle");

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mem_sr_sxm_clock_integration_test: "
                  << error.what() << '\n';
        return 1;
    }
}
