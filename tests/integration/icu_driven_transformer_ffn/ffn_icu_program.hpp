#pragma once

#include "ftlpu/icu/icu.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace icu_ffn_test {

template <typename Instruction>
struct TimedInstruction {
    std::size_t cycle{0};
    Instruction instruction{};
};

// Compiler-side collection of the FFN's independently schedulable physical
// instruction endpoints. The first pass records absolute issue cycles.
// load_into() lowers each endpoint to one local program with queue-local NOPs,
// writes the corresponding i-MEM, and completes the initial IQ prefetch before
// system cycle zero.
class FfnIcuProgram {
public:
    static constexpr std::size_t kPrefetchCycles = 16;
    using Icu = ftlpu::InstructionControlUnit;

    void record_mem(
        std::size_t cycle,
        ftlpu::Hemisphere hemisphere,
        std::size_t slice,
        ftlpu::MemInstruction instruction)
    {
        mem_.at(Icu::mem_queue(hemisphere, slice)).push_back(
            {cycle, std::move(instruction)});
        ++functional_events_;
    }

    void record_mxm(
        std::size_t cycle,
        std::size_t mxm,
        ftlpu::MxmControlInstruction instruction)
    {
        auto* queue = &mxm_load_.at(mxm);
        if (instruction.opcode == ftlpu::MxmControlOpcode::Compute)
            queue = &mxm_compute_.at(mxm);
        else if (instruction.opcode
                 == ftlpu::MxmControlOpcode::ActivationDequantize)
            queue = &mxm_dequant_.at(mxm);
        queue->push_back({cycle, std::move(instruction)});
        ++functional_events_;
    }

    void record_vxm(
        std::size_t cycle,
        std::size_t alu,
        ftlpu::VxmCompactInstruction instruction)
    {
        vxm_.at(alu).push_back({cycle, std::move(instruction)});
        ++functional_events_;
    }

    void record_sxm(
        std::size_t cycle,
        ftlpu::Hemisphere hemisphere,
        ftlpu::SxmInstruction instruction)
    {
        sxm_.at(ftlpu::hemisphere_index(hemisphere))
            .at(static_cast<std::size_t>(sxm_port(instruction)))
            .push_back(
            {cycle, std::move(instruction)});
        ++functional_events_;
    }

    std::size_t functional_events() const noexcept
    {
        return functional_events_;
    }

    std::size_t active_queues() const noexcept
    {
        auto count = std::size_t{0};
        count += nonempty(mem_);
        count += nonempty(mxm_load_);
        count += nonempty(mxm_compute_);
        count += nonempty(mxm_dequant_);
        count += nonempty(vxm_);
        for (const auto& ports : sxm_) count += nonempty(ports);
        return count;
    }

    void load_into(Icu& icu) const
    {
        for (std::size_t queue = 0; queue < mem_.size(); ++queue)
            load_queue(
                icu.mem_iq(queue), mem_[queue],
                "MEM" + std::to_string(queue));
        for (std::size_t mxm = 0; mxm < mxm_load_.size(); ++mxm) {
            load_queue(
                icu.mxm_iq(mxm, Icu::MxmIcuPort::Load),
                mxm_load_[mxm],
                "MXM" + std::to_string(mxm) + ".Load");
            load_queue(
                icu.mxm_iq(mxm, Icu::MxmIcuPort::Compute),
                mxm_compute_[mxm],
                "MXM" + std::to_string(mxm) + ".Compute");
            load_queue(
                icu.mxm_iq(mxm, Icu::MxmIcuPort::Dequant),
                mxm_dequant_[mxm],
                "MXM" + std::to_string(mxm) + ".Dequant");
        }
        for (std::size_t alu = 0; alu < vxm_.size(); ++alu)
            load_queue(
                icu.vxm_iq(alu), vxm_[alu],
                "VXM" + std::to_string(alu));
        for (std::size_t side = 0; side < sxm_.size(); ++side)
            for (std::size_t port = 0;
                 port < Icu::kSxmIcusPerHemisphere; ++port)
                load_queue(
                    icu.sxm_iq(
                        side, static_cast<Icu::SxmIcuPort>(port)),
                    sxm_[side][port],
                    "SXM" + std::to_string(side) + "."
                        + std::to_string(port));
    }

private:
    template <typename Queues>
    static std::size_t nonempty(const Queues& queues) noexcept
    {
        return static_cast<std::size_t>(std::count_if(
            queues.begin(), queues.end(),
            [](const auto& queue) { return !queue.empty(); }));
    }

    template <typename Queue, typename Instruction>
    static void load_queue(
        Queue& queue,
        const std::vector<TimedInstruction<Instruction>>& events,
        const std::string& queue_name)
    {
        if (events.empty()) return;
        auto ordered = events;
        std::stable_sort(
            ordered.begin(), ordered.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.cycle < rhs.cycle;
            });

        auto program = std::vector<typename Queue::Entry>{};
        auto cursor = std::size_t{0};
        for (const auto& event : ordered) {
            if (event.cycle < cursor) {
                throw ftlpu::StaticScheduleError(
                    "FFN compiler placed two instructions in "
                    + queue_name + " at cycle "
                    + std::to_string(event.cycle));
            }
            if (event.cycle > cursor) {
                program.push_back(typename Queue::Entry {
                    std::in_place_type<ftlpu::IcuControlInstruction>,
                    ftlpu::IcuControlInstruction::Nop(
                        event.cycle - cursor)});
            }
            program.push_back(typename Queue::Entry {
                std::in_place_type<Instruction>, event.instruction});
            cursor = event.cycle + 1;
        }

        queue.load_imem(0, std::move(program));
        queue.configure({0, queue.imem_occupancy(), kPrefetchCycles});
        for (std::size_t cycle = 0; cycle < kPrefetchCycles; ++cycle)
            queue.prefetch_only();
    }

    static Icu::SxmIcuPort sxm_port(
        const ftlpu::SxmInstruction& instruction)
    {
        switch (instruction.opcode) {
        case ftlpu::SxmOpcode::Transpose:
            return Icu::SxmIcuPort::Transpose;
        case ftlpu::SxmOpcode::Permute:
            return Icu::SxmIcuPort::Permute;
        case ftlpu::SxmOpcode::Distribute:
        case ftlpu::SxmOpcode::ShiftSelect:
            return Icu::SxmIcuPort::Stream;
        }
        throw std::logic_error("unknown SXM opcode in FFN compiler");
    }

    std::array<
        std::vector<TimedInstruction<ftlpu::MemInstruction>>,
        Icu::kMemQueues> mem_{};
    std::array<
        std::vector<TimedInstruction<ftlpu::MxmControlInstruction>>,
        Icu::kMxmUnitCount> mxm_load_{};
    std::array<
        std::vector<TimedInstruction<ftlpu::MxmControlInstruction>>,
        Icu::kMxmUnitCount> mxm_compute_{};
    std::array<
        std::vector<TimedInstruction<ftlpu::MxmControlInstruction>>,
        Icu::kMxmUnitCount> mxm_dequant_{};
    std::array<
        std::vector<TimedInstruction<ftlpu::VxmCompactInstruction>>,
        Icu::kVxmQueues> vxm_{};
    std::array<
        std::array<
            std::vector<TimedInstruction<ftlpu::SxmInstruction>>,
            Icu::kSxmIcusPerHemisphere>,
        Icu::kSxmQueues> sxm_{};
    std::size_t functional_events_{0};
};

} // namespace icu_ffn_test
