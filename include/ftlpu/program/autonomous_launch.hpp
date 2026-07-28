#pragma once

#include "ftlpu/program/preamble.hpp"
#include "ftlpu/program/program_image.hpp"
#include "ftlpu/program/sram_layout.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ftlpu::program {

struct AutonomousProgram {
    ProgramImage image{};
    ProgramSramLayout layout{};
    BootstrapPreamble preamble{};
    // System cycle at which schedule cycle zero is expected to dispatch.
    // This is a CModel launch-contract value, not a hardware performance
    // claim.
    std::size_t schedule_epoch_cycle{0};
};

// Converts one-block per-IQ schedule sections into a completely SRAM-backed
// launch image. MEM programs bootstrap from their own local SRAM. Every other
// functional IQ is fed by a dedicated MEM loader and waits at Sync until one
// conservative coordinator Notify releases all loaded targets together.
class AutonomousProgramBuilder {
public:
    static AutonomousProgram Build(
        const ProgramImage& workload,
        std::size_t program_bank = hw::kSramBanksPerTileBlock - 1,
        std::size_t first_program_row = 0)
    {
        if (workload.sections().empty()) {
            throw std::invalid_argument(
                "autonomous launch needs at least one program section");
        }
        if (!workload.entry_points().empty()) {
            throw std::invalid_argument(
                "autonomous launch derives entry points from sections");
        }
        if (program_bank >= hw::kSramBanksPerTileBlock) {
            throw std::out_of_range(
                "autonomous launch program bank is outside SRAM");
        }

        BuildState state {
            workload,
            program_bank,
            first_program_row,
        };
        state.collect_sections();
        state.assign_non_mem_loaders();
        state.build_output();
        return state.finish();
    }

private:
    static constexpr std::size_t kMemQueueCount =
        hw::kHemispheres * hw::kMemSliceColumns;

    struct TargetLoad {
        ProgramSection section{};
        Hemisphere loader_hemisphere{Hemisphere::East};
        std::size_t loader_mem_slice{0};
        StreamId stream{StreamId::East(0)};
        MemLocalWordAddress13 target_address{};
    };

    struct LocalProgram {
        std::optional<ProgramSection> schedule{};
        std::optional<std::size_t> target_load{};
        bool coordinator{false};
        MemLocalWordAddress13 program_address{};
    };

    struct BuildState {
        const ProgramImage& workload;
        std::size_t program_bank{0};
        std::size_t first_program_row{0};
        std::array<LocalProgram, kMemQueueCount> locals{};
        std::vector<TargetLoad> targets{};
        ProgramImage output;
        std::vector<MemGlobalAddress24> bases{};
        BootstrapPreamble preamble{};
        std::size_t schedule_epoch_from_local_start{0};

        BuildState(
            const ProgramImage& input,
            std::size_t bank,
            std::size_t row)
            : workload(input)
            , program_bank(bank)
            , first_program_row(row)
            , output(input.header())
        {
        }

        void collect_sections()
        {
            for (const auto& section : workload.sections()) {
                section.validate();
                if (section.packets.size()
                    > hw::kIcuFetchPackets) {
                    throw StaticScheduleError(
                        "autonomous launch currently requires each IQ "
                        "schedule to fit one 640-byte IFetch block");
                }
                if (section.target.kind == IcuLocationKind::Mem) {
                    const auto hemisphere =
                        checked_hemisphere(section.target.unit);
                    check_mem_slice(section.target.index);
                    auto& local =
                        locals[mem_queue(
                            hemisphere, section.target.index)];
                    if (local.schedule.has_value()) {
                        throw StaticScheduleError(
                            "autonomous launch has two schedules for one MEM IQ");
                    }
                    local.schedule = section;
                } else {
                    targets.push_back(TargetLoad {section});
                }
            }
        }

        void assign_non_mem_loaders()
        {
            std::array<std::size_t, hw::kHemispheres>
                next_east_loader{};
            std::array<std::size_t, hw::kHemispheres>
                next_east_stream{};
            auto next_vxm_loader = hw::kMemSliceColumns;
            auto next_vxm_stream = std::size_t {0};

            for (std::size_t target_index = 0;
                 target_index < targets.size();
                 ++target_index) {
                auto& target = targets[target_index];
                const auto location = target.section.target;
                if (location.kind == IcuLocationKind::Vxm) {
                    if (location.index >= hw::kVxmAluCount) {
                        throw std::out_of_range(
                            "VXM program target exceeds configured ALUs");
                    }
                    if (next_vxm_loader == 0
                        || next_vxm_stream
                            >= hw::kWestStreams) {
                        throw StaticScheduleError(
                            "autonomous launch has no westward VXM loader port");
                    }
                    target.loader_hemisphere = Hemisphere::East;
                    target.loader_mem_slice =
                        --next_vxm_loader;
                    target.stream =
                        StreamId::West(next_vxm_stream++);
                } else {
                    const auto hemisphere =
                        target_hemisphere(location);
                    const auto side =
                        hemisphere_index(hemisphere);
                    if (next_east_loader[side]
                            >= next_vxm_loader
                        || next_east_stream[side]
                            >= hw::kEastStreams) {
                        throw StaticScheduleError(
                            "autonomous launch has no eastward target loader port");
                    }
                    target.loader_hemisphere = hemisphere;
                    target.loader_mem_slice =
                        next_east_loader[side]++;
                    target.stream =
                        StreamId::East(
                            next_east_stream[side]++);
                }

                auto& local =
                    locals[mem_queue(
                        target.loader_hemisphere,
                        target.loader_mem_slice)];
                if (local.target_load.has_value()) {
                    throw StaticScheduleError(
                        "one autonomous MEM loader was assigned two targets");
                }
                local.target_load = target_index;
            }

            if (!targets.empty()) {
                for (auto& local : locals) {
                    if (local.target_load.has_value()) {
                        local.coordinator = true;
                        break;
                    }
                }
                const auto notify_cycle =
                    hw::kIcuFetchVectorCount
                    + hw::kTileRows
                    + hw::kMemBoundaryStreamRegisterColumns
                    + 2;
                schedule_epoch_from_local_start =
                    notify_cycle
                    + hw::kIcuBarrierLatencyCycles
                    + 2;
            }
        }

        void build_output()
        {
            allocate_target_addresses();

            // Target sections precede their local loader sections. Their
            // explicit bases point into the assigned loader's SRAM.
            for (const auto& target : targets) {
                output.add_section(target.section);
                bases.push_back(global_address(
                    target.loader_hemisphere,
                    target.loader_mem_slice,
                    target.target_address));
                preamble.entries.push_back(BootstrapEntry {
                    target.section.target,
                    IcuControlInstruction::Fetch(
                        target.stream)});
                preamble.entries.push_back(BootstrapEntry {
                    target.section.target,
                    IcuControlInstruction::Sync()});
            }

            const auto notify_cycle =
                targets.empty()
                ? 0
                : hw::kIcuFetchVectorCount
                    + hw::kTileRows
                    + hw::kMemBoundaryStreamRegisterColumns
                    + 2;
            for (std::size_t queue = 0;
                 queue < locals.size();
                 ++queue) {
                auto& local = locals[queue];
                if (!local.schedule.has_value()
                    && !local.target_load.has_value()
                    && !local.coordinator) {
                    continue;
                }
                const auto hemisphere =
                    static_cast<Hemisphere>(
                        queue / hw::kMemSliceColumns);
                const auto mem_slice =
                    queue % hw::kMemSliceColumns;
                auto packets =
                    make_local_packets(
                        local,
                        notify_cycle);
                auto local_section = ProgramSection {
                    IcuLocation::Mem(
                        hemisphere, mem_slice),
                    std::move(packets),
                    0,
                    "autonomous local loader/MEM schedule",
                };
                local_section.validate();
                if (local_section.packets.size()
                    > hw::kIcuFetchPackets) {
                    std::ostringstream os;
                    os << "local MEM." << hemisphere_short_name(hemisphere)
                       << mem_slice << " loader plus schedule needs "
                       << local_section.packets.size()
                       << " packets, exceeding one bootstrap block";
                    throw StaticScheduleError(os.str());
                }
                output.add_section(std::move(local_section));
                bases.push_back(global_address(
                    hemisphere,
                    mem_slice,
                    local.program_address));
                preamble.mem_local_bootstraps.push_back(
                    MemIcuLocalBootstrap {
                        mem_slice,
                        local.program_address,
                        hemisphere,
                    });
            }

            for (const auto& data :
                 workload.data_sections()) {
                output.add_data_section(data);
            }
        }

        AutonomousProgram finish()
        {
            auto layout =
                ProgramSramLayout::BuildAtSectionBases(
                    output, bases);
            return AutonomousProgram {
                std::move(output),
                std::move(layout),
                std::move(preamble),
                hw::kIcuFetchVectorCount
                    + schedule_epoch_from_local_start,
            };
        }

        void allocate_target_addresses()
        {
            std::array<std::size_t, kMemQueueCount>
                next_row{};
            next_row.fill(first_program_row);
            for (auto& target : targets) {
                const auto queue =
                    mem_queue(
                        target.loader_hemisphere,
                        target.loader_mem_slice);
                target.target_address =
                    checked_program_address(
                        next_row[queue]);
                next_row[queue] +=
                    hw::kIcuFetchVectorCount;
            }
            for (std::size_t queue = 0;
                 queue < locals.size();
                 ++queue) {
                auto& local = locals[queue];
                if (!local.schedule.has_value()
                    && !local.target_load.has_value()
                    && !local.coordinator) {
                    continue;
                }
                local.program_address =
                    checked_program_address(
                        next_row[queue]);
                next_row[queue] +=
                    hw::kIcuFetchVectorCount;
            }
        }

        std::vector<isa::EncodedInstructionPacket>
        make_local_packets(
            const LocalProgram& local,
            std::size_t notify_cycle) const
        {
            std::vector<isa::EncodedInstructionPacket>
                packets;
            auto cursor = std::size_t {0};
            if (local.target_load.has_value()) {
                const auto& target =
                    targets[*local.target_load];
                packets.push_back(encode_packet(
                    MemInstruction::Read(
                        target.target_address,
                        target.stream)));
                ++cursor;
                if (hw::kIcuFetchVectorCount > 1) {
                    packets.push_back(encode_packet(
                        IcuControlInstruction::Repeat(
                            hw::kIcuFetchVectorCount - 1,
                            1,
                            1)));
                    cursor +=
                        hw::kIcuFetchVectorCount - 1;
                }
            }
            if (local.coordinator) {
                append_nop_until(
                    packets, cursor, notify_cycle);
                packets.push_back(encode_packet(
                    IcuControlInstruction::Notify()));
                ++cursor;
            }
            if (!targets.empty()) {
                append_nop_until(
                    packets,
                    cursor,
                    schedule_epoch_from_local_start);
            }
            if (local.schedule.has_value()) {
                packets.insert(
                    packets.end(),
                    local.schedule->packets.begin(),
                    local.schedule->packets.end());
            }
            if (packets.empty()) {
                packets.push_back(
                    padding_nop_packet());
            }
            return packets;
        }

        static void append_nop_until(
            std::vector<isa::EncodedInstructionPacket>& packets,
            std::size_t& cursor,
            std::size_t target_cycle)
        {
            if (cursor > target_cycle) {
                throw StaticScheduleError(
                    "autonomous loader prefix exceeds common schedule epoch");
            }
            const auto cycles = target_cycle - cursor;
            if (cycles != 0) {
                packets.push_back(encode_packet(
                    IcuControlInstruction::Nop(cycles)));
                cursor = target_cycle;
            }
        }

        MemLocalWordAddress13 checked_program_address(
            std::size_t row) const
        {
            if (row + hw::kIcuFetchVectorCount
                > hw::kSramWordsPerBank) {
                throw std::out_of_range(
                    "autonomous program placement exceeds selected SRAM bank");
            }
            return MemLocalWordAddress13::FromFields(
                program_bank, row);
        }

        static MemGlobalAddress24 global_address(
            Hemisphere hemisphere,
            std::size_t mem_slice,
            MemLocalWordAddress13 local)
        {
            return MemGlobalAddress24::FromFields(
                hemisphere_index(hemisphere),
                mem_slice,
                local.slice_byte_address());
        }

        static Hemisphere checked_hemisphere(
            std::size_t unit)
        {
            if (unit >= hw::kHemispheres) {
                throw std::out_of_range(
                    "program target hemisphere is invalid");
            }
            return static_cast<Hemisphere>(unit);
        }

        static Hemisphere target_hemisphere(
            IcuLocation location)
        {
            switch (location.kind) {
            case IcuLocationKind::MxmLoad:
            case IcuLocationKind::MxmCompute:
                if (location.unit >= hw::kMxmCount) {
                    throw std::out_of_range(
                        "program target MXM is invalid");
                }
                return location.unit < hw::kWestMxmCount
                    ? Hemisphere::West
                    : Hemisphere::East;
            case IcuLocationKind::Sxm:
                return checked_hemisphere(location.unit);
            case IcuLocationKind::Vxm:
                return Hemisphere::East;
            case IcuLocationKind::Mem:
                return checked_hemisphere(location.unit);
            }
            throw std::logic_error(
                "unknown autonomous target kind");
        }

        static void check_mem_slice(std::size_t mem_slice)
        {
            if (mem_slice >= hw::kMemSliceColumns) {
                throw std::out_of_range(
                    "autonomous program MEM slice is invalid");
            }
        }

        static constexpr std::size_t mem_queue(
            Hemisphere hemisphere,
            std::size_t mem_slice) noexcept
        {
            return hemisphere_index(hemisphere)
                * hw::kMemSliceColumns
                + mem_slice;
        }
    };
};

} // namespace ftlpu::program
