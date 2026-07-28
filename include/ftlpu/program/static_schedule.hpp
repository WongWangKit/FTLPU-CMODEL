#pragma once

#include "ftlpu/program/packet_encoder.hpp"
#include "ftlpu/program/program_image.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ftlpu::program {

// Host-side representation of an exact-cycle static schedule.  It deliberately
// produces encoded ProgramImage sections rather than mutating an ICU.  Repeated
// row instructions are folded into the architectural Repeat command, keeping
// realistic Transformer schedules inside one finite 640-byte IQ fetch block.
class StaticSchedule {
public:
    void mem_at(
        std::size_t cycle,
        Hemisphere hemisphere,
        std::size_t mem_slice,
        MemInstruction instruction)
    {
        if (mem_slice >= hw::kMemSliceColumns) {
            throw std::out_of_range(
                "static schedule MEM slice is outside its hemisphere");
        }
        mem_[mem_queue(hemisphere, mem_slice)].events.push_back(
            Event<MemInstruction> {cycle, std::move(instruction)});
        last_cycle_ = std::max(last_cycle_, cycle);
    }

    void mem_at(
        std::size_t cycle,
        std::size_t mem_slice,
        MemInstruction instruction)
    {
        mem_at(
            cycle,
            Hemisphere::East,
            mem_slice,
            std::move(instruction));
    }

    void mxm_at(
        std::size_t cycle,
        std::size_t mxm,
        MxmControlInstruction instruction)
    {
        if (mxm >= hw::kMxmCount) {
            throw std::out_of_range(
                "static schedule MXM index is outside the configured units");
        }
        auto& queue = instruction.opcode == MxmControlOpcode::Compute
            ? mxm_compute_[mxm]
            : mxm_load_[mxm];
        queue.events.push_back(
            Event<MxmControlInstruction> {
                cycle, std::move(instruction)});
        last_cycle_ = std::max(last_cycle_, cycle);
    }

    void vxm_at(
        std::size_t cycle,
        std::size_t alu,
        VxmLaneAluInstruction instruction)
    {
        if (alu >= hw::kVxmAluCount) {
            throw std::out_of_range(
                "static schedule VXM ALU is outside the configured queues");
        }
        vxm_[alu].events.push_back(
            Event<VxmLaneAluInstruction> {
                cycle, std::move(instruction)});
        last_cycle_ = std::max(last_cycle_, cycle);
    }

    void sxm_at(
        std::size_t cycle,
        Hemisphere hemisphere,
        SxmInstruction instruction)
    {
        sxm_[hemisphere_index(hemisphere)].events.push_back(
            Event<SxmInstruction> {cycle, std::move(instruction)});
        last_cycle_ = std::max(last_cycle_, cycle);
    }

    std::size_t last_cycle() const noexcept
    {
        return last_cycle_;
    }

    bool empty() const noexcept
    {
        return active_queue_count() == 0;
    }

    std::size_t active_queue_count() const noexcept
    {
        std::size_t result = 0;
        count_active(mem_, result);
        count_active(mxm_load_, result);
        count_active(mxm_compute_, result);
        count_active(vxm_, result);
        count_active(sxm_, result);
        return result;
    }

    std::vector<ProgramSection> sections(
        std::string metadata_prefix = "static schedule") const
    {
        std::vector<ProgramSection> result;
        result.reserve(active_queue_count());
        for (std::size_t queue = 0; queue < mem_.size(); ++queue) {
            if (mem_[queue].events.empty()) continue;
            const auto hemisphere =
                static_cast<Hemisphere>(
                    queue / hw::kMemSliceColumns);
            const auto mem_slice =
                queue % hw::kMemSliceColumns;
            append_section(
                result,
                IcuLocation::Mem(hemisphere, mem_slice),
                mem_[queue].events,
                metadata_prefix + " MEM."
                    + hemisphere_short_name(hemisphere)
                    + std::to_string(mem_slice));
        }
        for (std::size_t mxm = 0; mxm < hw::kMxmCount; ++mxm) {
            if (!mxm_load_[mxm].events.empty()) {
                append_section(
                    result,
                    IcuLocation::MxmLoad(mxm),
                    mxm_load_[mxm].events,
                    metadata_prefix + " MXM"
                        + std::to_string(mxm) + ".load");
            }
            if (!mxm_compute_[mxm].events.empty()) {
                append_section(
                    result,
                    IcuLocation::MxmCompute(mxm),
                    mxm_compute_[mxm].events,
                    metadata_prefix + " MXM"
                        + std::to_string(mxm) + ".compute");
            }
        }
        for (std::size_t alu = 0; alu < vxm_.size(); ++alu) {
            if (vxm_[alu].events.empty()) continue;
            append_section(
                result,
                IcuLocation::Vxm(alu),
                vxm_[alu].events,
                metadata_prefix + " VXM.ALU"
                    + std::to_string(alu));
        }
        for (std::size_t side = 0; side < sxm_.size(); ++side) {
            if (sxm_[side].events.empty()) continue;
            const auto hemisphere =
                static_cast<Hemisphere>(side);
            append_section(
                result,
                IcuLocation::Sxm(hemisphere),
                sxm_[side].events,
                metadata_prefix + " SXM."
                    + hemisphere_short_name(hemisphere));
        }
        return result;
    }

    void append_to(
        ProgramImage& image,
        std::string metadata_prefix = "static schedule") const
    {
        for (auto& section : sections(std::move(metadata_prefix))) {
            image.add_section(std::move(section));
        }
    }

private:
    template <typename Instruction>
    struct Event {
        std::size_t cycle{0};
        Instruction instruction{};
    };

    template <typename Instruction>
    struct Queue {
        std::vector<Event<Instruction>> events{};
    };

    static constexpr std::size_t mem_queue(
        Hemisphere hemisphere,
        std::size_t mem_slice) noexcept
    {
        return hemisphere_index(hemisphere)
            * hw::kMemSliceColumns
            + mem_slice;
    }

    template <typename Queues>
    static void count_active(
        const Queues& queues,
        std::size_t& result) noexcept
    {
        for (const auto& queue : queues) {
            result += queue.events.empty() ? 0 : 1;
        }
    }

    static void append_nops(
        std::vector<isa::EncodedInstructionPacket>& packets,
        std::size_t cycles)
    {
        constexpr std::size_t kMaxEncodedNop = 0xffff;
        while (cycles != 0) {
            const auto chunk =
                std::min(cycles, kMaxEncodedNop);
            packets.push_back(encode_packet(
                IcuControlInstruction::Nop(chunk)));
            cycles -= chunk;
        }
    }

    static bool same_non_address_mem_fields(
        const MemInstruction& lhs,
        const MemInstruction& rhs)
    {
        return lhs.opcode == rhs.opcode
            && lhs.stream == rhs.stream
            && lhs.map_stream == rhs.map_stream
            && lhs.accumulator_destination
                == rhs.accumulator_destination
            && lhs.write_address == rhs.write_address
            && lhs.write_stream == rhs.write_stream;
    }

    static std::optional<std::int64_t> repeat_stride(
        const MemInstruction& first,
        const MemInstruction& second)
    {
        if (!same_non_address_mem_fields(first, second)
            || first.opcode == MemOpcode::ReadWrite) {
            return std::nullopt;
        }
        const auto lhs =
            static_cast<std::int64_t>(first.address.encoded());
        const auto rhs =
            static_cast<std::int64_t>(second.address.encoded());
        const auto stride = rhs - lhs;
        if (stride < -2048 || stride > 2047) {
            return std::nullopt;
        }
        return stride;
    }

    template <typename Instruction>
    static std::optional<std::int64_t> repeat_stride(
        const Instruction& first,
        const Instruction& second)
    {
        return encode_packet(first) == encode_packet(second)
            ? std::optional<std::int64_t> {0}
            : std::nullopt;
    }

    static bool matches_repeat(
        const MemInstruction& first,
        const MemInstruction& candidate,
        std::int64_t stride,
        std::size_t index)
    {
        if (!same_non_address_mem_fields(first, candidate)
            || first.opcode == MemOpcode::ReadWrite) {
            return false;
        }
        const auto expected =
            static_cast<std::int64_t>(first.address.encoded())
            + stride * static_cast<std::int64_t>(index);
        return expected >= 0
            && expected
                < static_cast<std::int64_t>(
                    hw::kMemLocalWordAddressCount)
            && candidate.address.encoded()
                == static_cast<std::size_t>(expected);
    }

    template <typename Instruction>
    static bool matches_repeat(
        const Instruction& first,
        const Instruction& candidate,
        std::int64_t stride,
        std::size_t)
    {
        return stride == 0
            && encode_packet(first) == encode_packet(candidate);
    }

    template <typename Instruction>
    static std::vector<isa::EncodedInstructionPacket> encode_events(
        std::vector<Event<Instruction>> events,
        const std::string& queue_name)
    {
        std::stable_sort(
            events.begin(),
            events.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.cycle < rhs.cycle;
            });

        std::vector<isa::EncodedInstructionPacket> packets;
        std::size_t cursor = 0;
        std::size_t event = 0;
        while (event < events.size()) {
            if (events[event].cycle < cursor) {
                std::ostringstream os;
                os << queue_name
                   << " has more than one instruction in cycle "
                   << events[event].cycle;
                throw StaticScheduleError(os.str());
            }
            append_nops(
                packets, events[event].cycle - cursor);
            packets.push_back(
                encode_packet(events[event].instruction));

            auto run = std::size_t {1};
            std::optional<std::int64_t> stride{};
            if (event + 1 < events.size()
                && events[event + 1].cycle
                    == events[event].cycle + 1) {
                stride = repeat_stride(
                    events[event].instruction,
                    events[event + 1].instruction);
            }
            constexpr std::size_t kMaxRun = 512;
            while (stride.has_value()
                   && run < kMaxRun
                   && event + run < events.size()
                   && events[event + run].cycle
                       == events[event].cycle + run
                   && matches_repeat(
                       events[event].instruction,
                       events[event + run].instruction,
                       *stride,
                       run)) {
                ++run;
            }
            if (run > 1) {
                packets.push_back(encode_packet(
                    IcuControlInstruction::Repeat(
                        run - 1, 1, *stride)));
            }
            cursor = events[event].cycle + run;
            event += run;
        }
        return packets;
    }

    template <typename Instruction>
    static void append_section(
        std::vector<ProgramSection>& sections,
        IcuLocation target,
        const std::vector<Event<Instruction>>& events,
        std::string metadata)
    {
        auto packets = encode_events(events, metadata);
        if (packets.size() > hw::kIcuFetchPackets) {
            std::ostringstream os;
            os << metadata << " encodes to " << packets.size()
               << " packets, exceeding one "
               << hw::kIcuFetchBufferBytes
               << "-byte IFetch block; split the schedule at an "
                  "explicit synchronization point";
            throw StaticScheduleError(os.str());
        }
        sections.push_back(ProgramSection {
            target,
            std::move(packets),
            0,
            std::move(metadata),
        });
    }

    std::array<
        Queue<MemInstruction>,
        hw::kHemispheres * hw::kMemSliceColumns> mem_{};
    std::array<Queue<MxmControlInstruction>, hw::kMxmCount>
        mxm_load_{};
    std::array<Queue<MxmControlInstruction>, hw::kMxmCount>
        mxm_compute_{};
    std::array<Queue<VxmLaneAluInstruction>, hw::kVxmAluCount>
        vxm_{};
    std::array<Queue<SxmInstruction>, hw::kHemispheres>
        sxm_{};
    std::size_t last_cycle_{0};
};

} // namespace ftlpu::program
