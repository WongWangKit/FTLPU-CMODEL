#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/hemisphere.hpp"
#include "ftlpu/icu/distributed_queue.hpp"
#include "ftlpu/icu/location.hpp"
#include "ftlpu/mem/tile_array.hpp"
#include "ftlpu/mxm/mxm.hpp"
#include "ftlpu/sxm/slice.hpp"
#include "ftlpu/vxm/backend.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <ostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ftlpu {

class InstructionControlUnit {
public:
    static constexpr std::size_t kVxmQueues = VxmSlice::kAluQueues;
    static constexpr std::size_t kDistributedVxmQueues =
        DistributedVxmSlice::kAluQueues;
    static constexpr std::size_t kMemQueuesPerHemisphere = hw::kSliceColumns;
    static constexpr std::size_t kMxmQueuesPerHemisphere =
        hw::kMxmsPerHemisphere;
    static constexpr std::size_t kSxmQueuesPerHemisphere = 2;
    static constexpr std::size_t kMemQueues = hw::kHemispheres * kMemQueuesPerHemisphere;
    static constexpr std::size_t kMxmQueues = hw::kMxmCount;
    static constexpr std::size_t kSxmQueues = hw::kHemispheres * kSxmQueuesPerHemisphere;

    static constexpr std::size_t mem_queue(Hemisphere hemisphere, std::size_t column)
    {
        return hemisphere_index(hemisphere) * kMemQueuesPerHemisphere + column;
    }

    static constexpr std::size_t mxm_queue(Hemisphere hemisphere, std::size_t local_mxm)
    {
        return hemisphere_index(hemisphere) * kMxmQueuesPerHemisphere + local_mxm;
    }

    using Repeat = IcuRepeat;
    using VxmIcu = DistributedIcuQueue<
        VxmLaneAluInstruction,
        hw::kIcuVxmInstructionBits,
        hw::kIcuVxmImemDepth,
        hw::kIcuVxmIqDepth,
        hw::kIcuFetchLatencyCycles>;
    using DistributedVxmIcu = DistributedIcuQueue<
        distributed_vxm::VxmCompactInstruction,
        hw::kIcuVxmInstructionBits,
        hw::kIcuDistributedVxmImemDepth,
        hw::kIcuVxmIqDepth,
        hw::kIcuFetchLatencyCycles>;
    using MemIcu = DistributedIcuQueue<
        MemInstruction,
        hw::kIcuMemInstructionBits,
        hw::kIcuMemImemDepth,
        hw::kIcuMemIqDepth,
        hw::kIcuFetchLatencyCycles>;
    using MxmIcu = DistributedIcuQueue<
        MxmControlInstruction,
        hw::kIcuMxmInstructionBits,
        hw::kIcuMxmImemDepth,
        hw::kIcuMxmIqDepth,
        hw::kIcuFetchLatencyCycles>;
    using MxmDequantIcu = DistributedIcuQueue<
        MxmDequantInstruction,
        hw::kIcuMxmInstructionBits,
        hw::kIcuMxmImemDepth,
        hw::kIcuMxmIqDepth,
        hw::kIcuFetchLatencyCycles>;
    using SxmIcu = DistributedIcuQueue<
        SxmInstruction,
        hw::kIcuSxmInstructionBits,
        hw::kIcuSxmImemDepth,
        hw::kIcuSxmIqDepth,
        hw::kIcuFetchLatencyCycles>;

    explicit InstructionControlUnit(
        std::size_t barrier_latency_cycles = hw::kIcuBarrierLatencyCycles)
        : barrier_latency_cycles_(barrier_latency_cycles)
    {
    }
    void reset()
    {
        for (auto& queue : vxm_queues_) queue.reset();
        for (auto& queue : distributed_vxm_queues_) queue.reset();
        for (auto& queue : mem_queues_) {
            queue.reset();
        }
        for (auto& queue : mxm_load_queues_) {
            queue.reset();
        }
        for (auto& queue : mxm_dequant_queues_) {
            queue.reset();
        }
        for (auto& queue : mxm_compute_queues_) {
            queue.reset();
        }
        for (auto& queue : sxm_transpose_queues_) queue.reset();
        for (auto& queue : sxm_permute_queues_) queue.reset();
        barrier_events_.clear();
        cycle_ = 0;
    }

    void enqueue_nop(std::size_t cycles)
    {
        for (auto& queue : vxm_queues_) queue.push_nop(cycles);
        for (auto& queue : mem_queues_) {
            queue.push_nop(cycles);
        }
        for (auto& queue : mxm_load_queues_) {
            queue.push_nop(cycles);
        }
        for (auto& queue : mxm_dequant_queues_) {
            queue.push_nop(cycles);
        }
        for (auto& queue : mxm_compute_queues_) {
            queue.push_nop(cycles);
        }
        for (auto& queue : sxm_transpose_queues_) queue.push_nop(cycles);
        for (auto& queue : sxm_permute_queues_) queue.push_nop(cycles);
    }

    void enqueue_control(
        IcuLocation location,
        IcuControlInstruction instruction)
    {
        switch (location.kind) {
        case IcuLocationKind::Mem:
            mem_queues_[mem_queue(
                static_cast<Hemisphere>(location.unit),
                location.index)].append_control(instruction);
            return;
        case IcuLocationKind::Vxm:
            check_vxm_queue(location.index);
            vxm_queues_[location.index].append_control(instruction);
            return;
        case IcuLocationKind::DistributedVxm:
            check_distributed_vxm_queue(location.index);
            distributed_vxm_queues_[location.index].append_control(
                instruction);
            return;
        case IcuLocationKind::MxmLoad:
            check_mxm_queue(location.unit);
            mxm_load_queues_[location.unit].append_control(instruction);
            return;
        case IcuLocationKind::MxmCompute:
            check_mxm_queue(location.unit);
            mxm_compute_queues_[location.unit].append_control(instruction);
            return;
        case IcuLocationKind::MxmDequant:
            check_mxm_queue(location.unit);
            mxm_dequant_queues_[location.unit].append_control(instruction);
            return;
        case IcuLocationKind::Sxm:
            if (location.unit >= hw::kHemispheres) {
                throw std::out_of_range(
                    "ICU SXM hemisphere is outside the chip");
            }
            if (location.index == 0) {
                sxm_transpose_queues_[location.unit].append_control(
                    instruction);
                return;
            }
            if (location.index == 1) {
                sxm_permute_queues_[location.unit].append_control(
                    instruction);
                return;
            }
            throw std::out_of_range(
                "ICU SXM control port must be transpose(0) or permute(1)");
        }
        throw std::logic_error("unknown ICU location kind");
    }
    void enqueue_vxm(std::size_t alu, VxmLaneAluInstruction instruction)
    {
        check_vxm_queue(alu);
        vxm_queues_[alu].push_instruction(instruction);
    }

    void enqueue_vxm_nop(std::size_t alu, std::size_t cycles)
    {
        check_vxm_queue(alu);
        vxm_queues_[alu].push_nop(cycles);
    }

    void enqueue_vxm_repeat(std::size_t alu, std::size_t count, std::size_t interval = 1)
    {
        check_vxm_queue(alu);
        vxm_queues_[alu].push_repeat(Repeat {count, interval, 0});
    }

    void enqueue_distributed_vxm(
        std::size_t alu,
        distributed_vxm::VxmCompactInstruction instruction)
    {
        check_distributed_vxm_queue(alu);
        distributed_vxm_queues_[alu].push_instruction(
            std::move(instruction));
    }

    void enqueue_distributed_vxm_nop(
        std::size_t alu, std::size_t cycles)
    {
        check_distributed_vxm_queue(alu);
        distributed_vxm_queues_[alu].push_nop(cycles);
    }

    void enqueue_distributed_vxm_repeat(
        std::size_t alu,
        std::size_t count,
        std::size_t interval = 1)
    {
        check_distributed_vxm_queue(alu);
        distributed_vxm_queues_[alu].push_repeat(
            Repeat {count, interval, 0});
    }
    void enqueue_mem(std::size_t column, MemInstruction instruction)
    {
        check_mem_queue(column);
        mem_queues_[column].push_instruction(instruction);
    }

    void enqueue_mem_nop(std::size_t column, std::size_t cycles)
    {
        check_mem_queue(column);
        mem_queues_[column].push_nop(cycles);
    }

    void enqueue_mem_repeat(
        std::size_t column,
        std::size_t count,
        std::size_t interval = 1,
        std::int64_t address_stride = 0)
    {
        check_mem_queue(column);
        mem_queues_[column].push_repeat(Repeat {count, interval, address_stride});
    }

    void enqueue_mxm(std::size_t mxm, MxmControlInstruction instruction)
    {
        check_mxm_queue(mxm);
        if (instruction.opcode == MxmControlOpcode::IW
            || (instruction.opcode == MxmControlOpcode::Decode
                && instruction.decode_operation
                    == MxmDecodeOperation::LoadActivation)) {
            mxm_load_queues_[mxm].push_instruction(instruction);
        } else {
            mxm_compute_queues_[mxm].push_instruction(instruction);
        }
    }

    void enqueue_mxm_nop(std::size_t mxm, std::size_t cycles)
    {
        check_mxm_queue(mxm);
        mxm_load_queues_[mxm].push_nop(cycles);
        mxm_dequant_queues_[mxm].push_nop(cycles);
        mxm_compute_queues_[mxm].push_nop(cycles);
    }

    void enqueue_mxm_load_nop(std::size_t mxm, std::size_t cycles)
    {
        check_mxm_queue(mxm);
        mxm_load_queues_[mxm].push_nop(cycles);
    }

    void enqueue_mxm_dequant(
        std::size_t mxm,
        MxmDequantInstruction instruction)
    {
        check_mxm_queue(mxm);
        mxm_dequant_queues_[mxm].push_instruction(instruction);
    }

    void enqueue_mxm_dequant_nop(std::size_t mxm, std::size_t cycles)
    {
        check_mxm_queue(mxm);
        mxm_dequant_queues_[mxm].push_nop(cycles);
    }

    void enqueue_mxm_compute_nop(std::size_t mxm, std::size_t cycles)
    {
        check_mxm_queue(mxm);
        mxm_compute_queues_[mxm].push_nop(cycles);
    }

    void enqueue_mxm_repeat(std::size_t mxm, std::size_t count, std::size_t interval = 1)
    {
        check_mxm_queue(mxm);
        mxm_compute_queues_[mxm].push_repeat(Repeat {count, interval, 0});
    }

    void enqueue_mxm_load_repeat(std::size_t mxm, std::size_t count, std::size_t interval = 1)
    {
        check_mxm_queue(mxm);
        mxm_load_queues_[mxm].push_repeat(Repeat {count, interval, 0});
    }

    void enqueue_mxm_dequant_repeat(
        std::size_t mxm,
        std::size_t count,
        std::size_t interval = 1)
    {
        check_mxm_queue(mxm);
        mxm_dequant_queues_[mxm].push_repeat(
            Repeat {count, interval, 0});
    }

    void enqueue_mxm_compute_repeat(std::size_t mxm, std::size_t count, std::size_t interval = 1)
    {
        check_mxm_queue(mxm);
        mxm_compute_queues_[mxm].push_repeat(Repeat {count, interval, 0});
    }

    void enqueue_sxm_transpose(SxmInstruction instruction)
    {
        enqueue_sxm_transpose(Hemisphere::East, std::move(instruction));
    }

    void enqueue_sxm_transpose(Hemisphere hemisphere, SxmInstruction instruction)
    {
        if (instruction.opcode != SxmOpcode::Transpose) {
            throw std::invalid_argument("ICU SXM transpose queue requires a Transpose instruction");
        }
        sxm_transpose_queues_[hemisphere_index(hemisphere)].push_instruction(std::move(instruction));
    }

    void enqueue_sxm_permute(SxmInstruction instruction)
    {
        enqueue_sxm_permute(Hemisphere::East, std::move(instruction));
    }

    void enqueue_sxm_permute(Hemisphere hemisphere, SxmInstruction instruction)
    {
        if (instruction.opcode != SxmOpcode::Permute) {
            throw std::invalid_argument("ICU SXM permute queue requires a Permute instruction");
        }
        sxm_permute_queues_[hemisphere_index(hemisphere)].push_instruction(std::move(instruction));
    }

    void enqueue_sxm_transpose_nop(std::size_t cycles)
    {
        enqueue_sxm_transpose_nop(Hemisphere::East, cycles);
    }

    void enqueue_sxm_transpose_nop(Hemisphere hemisphere, std::size_t cycles)
    {
        sxm_transpose_queues_[hemisphere_index(hemisphere)].push_nop(cycles);
    }

    void enqueue_sxm_permute_nop(std::size_t cycles)
    {
        enqueue_sxm_permute_nop(Hemisphere::East, cycles);
    }

    void enqueue_sxm_permute_nop(Hemisphere hemisphere, std::size_t cycles)
    {
        sxm_permute_queues_[hemisphere_index(hemisphere)].push_nop(cycles);
    }

    void enqueue_sxm_transpose_repeat(std::size_t count, std::size_t interval = 1)
    {
        enqueue_sxm_transpose_repeat(Hemisphere::East, count, interval);
    }

    void enqueue_sxm_transpose_repeat(Hemisphere hemisphere, std::size_t count, std::size_t interval = 1)
    {
        sxm_transpose_queues_[hemisphere_index(hemisphere)].push_repeat(Repeat {count, interval, 0});
    }

    void enqueue_sxm_permute_repeat(std::size_t count, std::size_t interval = 1)
    {
        enqueue_sxm_permute_repeat(Hemisphere::East, count, interval);
    }

    void enqueue_sxm_permute_repeat(Hemisphere hemisphere, std::size_t count, std::size_t interval = 1)
    {
        sxm_permute_queues_[hemisphere_index(hemisphere)].push_repeat(Repeat {count, interval, 0});
    }

    void notify(IcuLocation location)
    {
        switch (location.kind) {
        case IcuLocationKind::Mem:
            mem_iq(mem_queue(
                static_cast<Hemisphere>(location.unit),
                location.index)).notify();
            return;
        case IcuLocationKind::Vxm:
            vxm_iq(location.index).notify();
            return;
        case IcuLocationKind::DistributedVxm:
            distributed_vxm_iq(location.index).notify();
            return;
        case IcuLocationKind::MxmLoad:
            mxm_load_iq(location.unit).notify();
            return;
        case IcuLocationKind::MxmCompute:
            mxm_compute_iq(location.unit).notify();
            return;
        case IcuLocationKind::MxmDequant:
            mxm_dequant_iq(location.unit).notify();
            return;
        case IcuLocationKind::Sxm:
            if (location.unit >= hw::kHemispheres) {
                throw std::out_of_range(
                    "ICU SXM hemisphere is outside the chip");
            }
            if (location.index == 0) {
                sxm_transpose_queues_[location.unit].notify();
                return;
            }
            if (location.index == 1) {
                sxm_permute_queues_[location.unit].notify();
                return;
            }
            throw std::out_of_range(
                "ICU SXM control port must be transpose(0) or permute(1)");
        }
        throw std::logic_error("unknown ICU location kind");
    }

    void advance_barrier_events()
    {
        for (auto& remaining : barrier_events_) {
            if (remaining != 0) --remaining;
        }
        while (!barrier_events_.empty()
               && barrier_events_.front() == 0) {
            barrier_events_.pop_front();
            broadcast_notification();
        }

        const auto emitted = take_emitted_notifications();
        for (std::size_t event = 0; event < emitted; ++event) {
            if (barrier_latency_cycles_ == 0) {
                broadcast_notification();
            } else {
                barrier_events_.push_back(barrier_latency_cycles_);
            }
        }
    }

    void broadcast_notification()
    {
        for (auto& queue : vxm_queues_) queue.notify();
        for (auto& queue : distributed_vxm_queues_) queue.notify();
        for (auto& queue : mem_queues_) queue.notify();
        for (auto& queue : mxm_load_queues_) queue.notify();
        for (auto& queue : mxm_dequant_queues_) queue.notify();
        for (auto& queue : mxm_compute_queues_) queue.notify();
        for (auto& queue : sxm_transpose_queues_) queue.notify();
        for (auto& queue : sxm_permute_queues_) queue.notify();
    }

    std::size_t barrier_latency_cycles() const noexcept
    {
        return barrier_latency_cycles_;
    }

    std::size_t pending_barrier_event_count() const noexcept
    {
        return barrier_events_.size();
    }
    void dispatch_vxm(VxmSlice& vxm, std::ostream* os = nullptr)
    {
        log_cycle_header(os);
        bool any = false;
        for (std::size_t alu = 0; alu < kVxmQueues; ++alu) {
            const auto instruction = vxm_queues_[alu].dispatch_next();
            if (!instruction.has_value()) continue;
            vxm.issue_south(alu, *instruction);
            any = true;
            if (os != nullptr) *os << "  ICU -> VXM.alu" << alu << ' ' << describe_vxm(*instruction) << '\n';
        }
        log_dispatch_idle(os, any);
        ++cycle_;
    }

    void dispatch_vxm(
        DistributedVxmSlice& vxm,
        std::ostream* os = nullptr)
    {
        log_cycle_header(os);
        bool any = false;
        for (std::size_t alu = 0; alu < kDistributedVxmQueues; ++alu) {
            const auto instruction =
                distributed_vxm_queues_[alu].dispatch_next();
            if (!instruction.has_value()) continue;
            vxm.issue_south(alu, *instruction);
            any = true;
            if (os != nullptr) {
                *os << "  ICU -> VXM.distributed.q" << alu
                    << " compact\n";
            }
        }
        log_dispatch_idle(os, any);
        ++cycle_;
    }
    void dispatch(
        std::array<TileArrayModel, hw::kHemispheres>& mems,
        VxmSlice& vxm,
        std::array<SxmSlice, hw::kHemispheres>& sxms,
        std::array<Mxm, kMxmQueues>& mxms,
        std::ostream* os = nullptr)
    {
        log_cycle_header(os);

        bool any = false;
        for (std::size_t alu = 0; alu < kVxmQueues; ++alu) {
            const auto instruction = vxm_queues_[alu].dispatch_next();
            if (!instruction.has_value()) continue;
            vxm.issue_south(alu, *instruction);
            any = true;
            if (os != nullptr) *os << "  ICU -> VXM.alu" << alu << ' ' << describe_vxm(*instruction) << '\n';
        }

        for (std::size_t column = 0; column < kMemQueues; ++column) {
            const auto instruction = mem_queues_[column].dispatch_next();
            if (!instruction.has_value()) {
                continue;
            }
            const auto hemisphere = column / kMemQueuesPerHemisphere;
            const auto local_column = column % kMemQueuesPerHemisphere;
            mems[hemisphere].enqueue_instruction(local_column, *instruction);
            any = true;
            if (os != nullptr) {
                *os << "  ICU -> MEM." << hemisphere_short_name(static_cast<Hemisphere>(hemisphere))
                    << " q" << local_column << ' ' << describe_mem(*instruction) << '\n';
            }
        }

        for (std::size_t hemisphere = 0; hemisphere < hw::kHemispheres; ++hemisphere) {
            const auto transpose = sxm_transpose_queues_[hemisphere].dispatch_next();
            if (transpose.has_value()) {
                sxms[hemisphere].issue(*transpose);
                any = true;
                if (os != nullptr) {
                    *os << "  ICU -> SXM." << hemisphere_short_name(static_cast<Hemisphere>(hemisphere))
                        << ".transpose " << describe_sxm(*transpose) << '\n';
                }
            }

            const auto permute = sxm_permute_queues_[hemisphere].dispatch_next();
            if (permute.has_value()) {
                sxms[hemisphere].issue(*permute);
                any = true;
                if (os != nullptr) {
                    *os << "  ICU -> SXM." << hemisphere_short_name(static_cast<Hemisphere>(hemisphere))
                        << ".permute " << describe_sxm(*permute) << '\n';
                }
            }
        }

        for (std::size_t mxm = 0; mxm < kMxmQueues; ++mxm) {
            const auto instruction =
                mxm_dequant_queues_[mxm].dispatch_next();
            if (!instruction.has_value()) {
                continue;
            }
            mxms[mxm].control().issue_dequant_south(*instruction);
            any = true;
            if (os != nullptr) {
                *os << "  ICU -> MXM" << mxm
                    << ".dequant scale=" << instruction->scale() << '\n';
            }
        }

        for (std::size_t mxm = 0; mxm < kMxmQueues; ++mxm) {
            const auto instruction = mxm_load_queues_[mxm].dispatch_next();
            if (!instruction.has_value()) {
                continue;
            }
            mxms[mxm].control().issue_south(*instruction);
            any = true;
            if (os != nullptr) {
                *os << "  ICU -> MXM" << mxm << ".load " << describe_mxm(*instruction) << '\n';
            }
        }

        for (std::size_t mxm = 0; mxm < kMxmQueues; ++mxm) {
            const auto instruction = mxm_compute_queues_[mxm].dispatch_next();
            if (!instruction.has_value()) {
                continue;
            }
            mxms[mxm].control().issue_south(*instruction);
            any = true;
            if (os != nullptr) {
                *os << "  ICU -> MXM" << mxm << ".compute " << describe_mxm(*instruction) << '\n';
            }
        }

        log_dispatch_idle(os, any);
        ++cycle_;
    }

    VxmIcu& vxm_iq(std::size_t alu)
    {
        check_vxm_queue(alu);
        return vxm_queues_[alu];
    }

    DistributedVxmIcu& distributed_vxm_iq(std::size_t queue)
    {
        check_distributed_vxm_queue(queue);
        return distributed_vxm_queues_[queue];
    }

    MemIcu& mem_iq(std::size_t queue)
    {
        check_mem_queue(queue);
        return mem_queues_[queue];
    }

    MxmIcu& mxm_load_iq(std::size_t mxm)
    {
        check_mxm_queue(mxm);
        return mxm_load_queues_[mxm];
    }

    MxmDequantIcu& mxm_dequant_iq(std::size_t mxm)
    {
        check_mxm_queue(mxm);
        return mxm_dequant_queues_[mxm];
    }

    MxmIcu& mxm_compute_iq(std::size_t mxm)
    {
        check_mxm_queue(mxm);
        return mxm_compute_queues_[mxm];
    }

    SxmIcu& sxm_transpose_iq(Hemisphere hemisphere)
    {
        return sxm_transpose_queues_[hemisphere_index(hemisphere)];
    }

    SxmIcu& sxm_permute_iq(Hemisphere hemisphere)
    {
        return sxm_permute_queues_[hemisphere_index(hemisphere)];
    }
    std::size_t cycle() const
    {
        return cycle_;
    }

private:
    std::size_t take_emitted_notifications()
    {
        std::size_t count = 0;
        const auto collect = [&count](auto& queues) {
            for (auto& queue : queues) {
                count += queue.take_notify() ? 1U : 0U;
            }
        };
        collect(vxm_queues_);
        collect(distributed_vxm_queues_);
        collect(mem_queues_);
        collect(mxm_load_queues_);
        collect(mxm_dequant_queues_);
        collect(mxm_compute_queues_);
        collect(sxm_transpose_queues_);
        collect(sxm_permute_queues_);
        return count;
    }
    static void check_mem_queue(std::size_t column)
    {
        if (column >= kMemQueues) {
            throw std::out_of_range(
                "ICU MEM queue is outside the configured full-chip MEM queues");
        }
    }

    static void check_mxm_queue(std::size_t mxm)
    {
        if (mxm >= kMxmQueues) {
            throw std::out_of_range("ICU MXM queue is outside the four full-chip MXM queues");
        }
    }

    static void check_vxm_queue(std::size_t alu)
    {
        if (alu >= kVxmQueues) throw std::out_of_range("ICU VXM queue is outside the 16 ALU queues");
    }

    static void check_distributed_vxm_queue(std::size_t alu)
    {
        if (alu >= kDistributedVxmQueues) {
            throw std::out_of_range(
                "ICU distributed VXM queue is outside the 8 compact queues");
        }
    }
    template <typename QueueArray>
    static std::size_t queued_instruction_count(const QueueArray& queues)
    {
        std::size_t count = 0;
        for (const auto& queue : queues) {
            count += queue.queued_count();
        }
        return count;
    }

    void log_cycle_header(std::ostream* os) const
    {
        if (os == nullptr) {
            return;
        }

        *os << "icu cycle " << cycle_ << '\n';
        *os << "  queues:"
            << " vxm=" << queued_instruction_count(vxm_queues_)
            << " mem=" << queued_instruction_count(mem_queues_)
            << " mxm_load=" << queued_instruction_count(mxm_load_queues_)
            << " mxm_dequant="
            << queued_instruction_count(mxm_dequant_queues_)
            << " mxm_compute=" << queued_instruction_count(mxm_compute_queues_)
            << " sxm_transpose=" << queued_instruction_count(sxm_transpose_queues_)
            << " sxm_permute=" << queued_instruction_count(sxm_permute_queues_)
            << '\n';
    }

    static void log_dispatch_idle(std::ostream* os, bool any)
    {
        if (os != nullptr && !any) {
            *os << "  ICU dispatch idle\n";
        }
    }

    static const char* mem_opcode_name(MemOpcode opcode)
    {
        switch (opcode) {
        case MemOpcode::Read:
            return "Read";
        case MemOpcode::Write:
            return "Write";
        case MemOpcode::ReadWrite:
            return "ReadWrite";
        case MemOpcode::Gather:
            return "Gather";
        case MemOpcode::Scatter:
            return "Scatter";
        }
        return "?";
    }

    static std::string describe_mem(const MemInstruction& instruction)
    {
        std::ostringstream os;
        os << mem_opcode_name(instruction.opcode)
           << " address=" << instruction.address
           << " stream=" << instruction.stream;
        if (instruction.opcode == MemOpcode::ReadWrite) {
            os << " write_address=" << instruction.write_address
               << " write_stream=" << instruction.write_stream;
        }
        if (instruction.opcode == MemOpcode::Gather || instruction.opcode == MemOpcode::Scatter) {
            os << " map_stream=" << instruction.map_stream;
        }
        return os.str();
    }

    static std::string describe_mxm(const MxmControlInstruction& instruction)
    {
        std::ostringstream os;
        if (instruction.opcode == MxmControlOpcode::IW) {
            os << "IW b" << instruction.weight_buffer
               << " col=" << instruction.weight_column;
            if (instruction.weight_load_mode == MxmWeightLoadMode::Column) {
                os << " inner=" << instruction.weight_inner_column
                   << " streams="
                   << (instruction.weight_input_mode
                               == MxmWeightInputMode::Int8DequantBf16
                           ? 1
                           : 2);
            }
            os << (instruction.weight_input_mode
                           == MxmWeightInputMode::Int8DequantBf16
                       ? " int8"
                       : " direct16");
        } else if (instruction.opcode == MxmControlOpcode::Compute) {
            os << "Compute b" << instruction.weight_buffer
               << " stream=" << instruction.activation_stream_base
               << " acc=" << instruction.accumulator_address
               << " out=" << instruction.stream_base
               << " mode="
               << (instruction.compute_mode == MxmComputeMode::Block8
                       ? "block8"
                       : "vector");
        } else if (instruction.opcode == MxmControlOpcode::Decode) {
            if (instruction.decode_operation
                == MxmDecodeOperation::LoadActivation) {
                os << "DecodeLoadActivation b"
                   << instruction.weight_buffer
                   << " stream=" << instruction.activation_stream_base
                   << " format="
                   << mxm_data_format_name(instruction.data_format);
            } else {
                os << "DecodeStreamCompute b"
                   << instruction.weight_buffer
                   << " out=" << instruction.stream_base
                   << " weight_streams=E0..E31";
            }
        } else {
            os << "AccumulatorRead address=" << instruction.accumulator_address
               << " out=" << instruction.stream_base
               << " mode="
               << (instruction.compute_mode == MxmComputeMode::Block8
                       ? "block8"
                       : "vector");
        }
        return os.str();
    }

    static std::string describe_sxm(const SxmInstruction& instruction)
    {
        const auto describe_streams = [](const SxmInstruction::StreamList& streams) {
            auto result = std::ostringstream {};
            if (streams.empty()) return result.str();
            const auto first = StreamId::from_packed(streams.front().stream);
            const auto last = StreamId::from_packed(streams.back().stream);
            const auto direction = first.direction() == StreamDirection::East ? 'E' : 'W';
            result << direction << first.index();
            if (streams.size() > 1) result << ".." << direction << last.index();
            result << " (" << streams.size() << ')';
            return result.str();
        };

        std::ostringstream os;
        os << (instruction.opcode == SxmOpcode::Transpose ? "Transpose" : "Permute");
        if (!instruction.src_streams.empty()) os << " src=" << describe_streams(instruction.src_streams);
        if (!instruction.dst_streams.empty()) os << " dst=" << describe_streams(instruction.dst_streams);
        return os.str();
    }

    static std::string describe_vxm(const VxmLaneAluInstruction& instruction)
    {
        std::ostringstream os;
        os << VxmLane::opcode_name(instruction.opcode)
           << " in_hemi=" << hemisphere_short_name(instruction.input_hemisphere)
           << " out_hemi=" << hemisphere_short_name(instruction.output_hemisphere);
        if (instruction.output_stream.has_value()) os << " output=" << *instruction.output_stream;
        return os.str();
    }

    std::array<VxmIcu, kVxmQueues> vxm_queues_{};
    std::array<DistributedVxmIcu, kDistributedVxmQueues>
        distributed_vxm_queues_{};
    std::array<MemIcu, kMemQueues> mem_queues_{};
    std::array<MxmIcu, kMxmQueues> mxm_load_queues_{};
    std::array<MxmDequantIcu, kMxmQueues>
        mxm_dequant_queues_{};
    std::array<MxmIcu, kMxmQueues> mxm_compute_queues_{};
    std::array<SxmIcu, hw::kHemispheres> sxm_transpose_queues_{};
    std::array<SxmIcu, hw::kHemispheres> sxm_permute_queues_{};
    std::size_t barrier_latency_cycles_{hw::kIcuBarrierLatencyCycles};
    std::deque<std::size_t> barrier_events_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
