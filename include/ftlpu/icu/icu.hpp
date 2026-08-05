#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/icu/distributed_queue.hpp"
#include "ftlpu/icu/location.hpp"
#include "ftlpu/mem/tile_array.hpp"
#include "ftlpu/mxm/mxm.hpp"
#include "ftlpu/mxm/activation_dequantizer.hpp"
#include "ftlpu/sxm/slice.hpp"
#include "ftlpu/vxm/slice.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace ftlpu {

// Whole-system collection of distributed ICU endpoints. Each endpoint owns a
// local, function-width i-MEM and a finite prefetch IQ. The array of queue
// heads issued in one cycle is the logical VLIW; no monolithic wide VLIW word
// or instruction transport over the data Stream Registers exists here.
class InstructionControlUnit {
public:
    using Repeat = IcuRepeat;
    enum class MxmIcuPort : std::size_t {
        Load = 0,
        Compute = 1,
        Dequant = 2,
    };
    enum class SxmIcuPort : std::size_t {
        Stream = 0,
        Transpose = 1,
        Permute = 2,
    };

    using VxmIcu = DistributedIcuQueue<
        VxmCompactInstruction,
        hw::kIcuVxmInstructionBits,
        hw::kIcuVxmImemDepth,
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
    using SxmIcu = DistributedIcuQueue<
        SxmInstruction,
        hw::kIcuSxmInstructionBits,
        hw::kIcuSxmImemDepth,
        hw::kIcuSxmIqDepth,
        hw::kIcuFetchLatencyCycles>;

    static_assert(hw::kIcuMxmInstructionBits >= 32);
    static_assert(hw::kIcuVxmInstructionBits >= 96);
    static_assert(hw::kIcuMemInstructionBits >= 96);
    static_assert(hw::kIcuSxmInstructionBits >= 96);

    static constexpr std::size_t kVxmQueues = VxmSlice::kAluQueues;
    static constexpr std::size_t kMemQueuesPerHemisphere = hw::kSliceColumns;
    static constexpr std::size_t kMemQueues =
        hw::kHemispheres * kMemQueuesPerHemisphere;
    static constexpr std::size_t kMxmUnitCount = hw::kMxmCount;
    static constexpr std::size_t kMxmIcusPerUnit = 3;
    static constexpr std::size_t kSxmQueues = hw::kHemispheres;
    static constexpr std::size_t kSxmIcusPerHemisphere = 3;

    static constexpr std::size_t mem_queue(
        Hemisphere hemisphere, std::size_t mem_slice) noexcept
    {
        return hemisphere_index(hemisphere) * kMemQueuesPerHemisphere
            + mem_slice;
    }

    explicit InstructionControlUnit(
        std::size_t barrier_latency_cycles = hw::kIcuBarrierLatencyCycles)
        : barrier_latency_cycles_(barrier_latency_cycles)
    {
    }

    void reset()
    {
        reset_all(vxm_iqs_);
        reset_all(mem_iqs_);
        for (auto& ports : mxm_iqs_) reset_all(ports);
        for (auto& ports : sxm_iqs_) reset_all(ports);
        barrier_events_.clear();
        cycle_ = 0;
    }

    void enqueue_control(
        IcuLocation location, IcuControlInstruction instruction)
    {
        switch (location.kind) {
        case IcuLocationKind::Mem:
            enqueue_mem_control(
                mem_queue(
                    static_cast<Hemisphere>(location.unit), location.index),
                instruction);
            return;
        case IcuLocationKind::Vxm:
            enqueue_vxm_control(location.index, instruction);
            return;
        case IcuLocationKind::MxmLoad:
            enqueue_mxm_control(location.unit, MxmIcuPort::Load, instruction);
            return;
        case IcuLocationKind::MxmCompute:
            enqueue_mxm_control(location.unit, MxmIcuPort::Compute, instruction);
            return;
        case IcuLocationKind::MxmDequant:
            enqueue_mxm_control(location.unit, MxmIcuPort::Dequant, instruction);
            return;
        case IcuLocationKind::Sxm:
            enqueue_sxm_control(
                location.unit,
                static_cast<SxmIcuPort>(location.index),
                instruction);
            return;
        }
        throw std::logic_error("unknown ICU location kind");
    }

    void enqueue_nop(std::size_t cycles)
    {
        if (cycles == 0) return;
        for (auto& iq : vxm_iqs_) iq.append_control(IcuControlInstruction::Nop(cycles));
        for (auto& iq : mem_iqs_) iq.append_control(IcuControlInstruction::Nop(cycles));
        for (auto& ports : mxm_iqs_) {
            for (auto& iq : ports) iq.append_control(IcuControlInstruction::Nop(cycles));
        }
        for (auto& ports : sxm_iqs_)
            for (auto& iq : ports)
                iq.append_control(IcuControlInstruction::Nop(cycles));
    }

    void enqueue_vxm(std::size_t alu, VxmCompactInstruction instruction)
    {
        vxm_iq(alu).append_program(std::move(instruction));
    }

    void enqueue_vxm_control(
        std::size_t alu, IcuControlInstruction instruction)
    {
        vxm_iq(alu).append_control(instruction);
    }

    void enqueue_vxm_nop(std::size_t alu, std::size_t cycles)
    {
        if (cycles != 0) enqueue_vxm_control(alu, IcuControlInstruction::Nop(cycles));
    }

    void enqueue_vxm_repeat(
        std::size_t alu, std::size_t count, std::size_t interval = 1)
    {
        if (count != 0) enqueue_vxm_control(
            alu, IcuControlInstruction::Repeat(count, interval));
    }

    void enqueue_mem(std::size_t queue, MemInstruction instruction)
    {
        mem_iq(queue).append_program(std::move(instruction));
    }

    void enqueue_mem(
        Hemisphere hemisphere,
        std::size_t mem_slice,
        MemInstruction instruction)
    {
        enqueue_mem(
            mem_queue(hemisphere, mem_slice), std::move(instruction));
    }

    void enqueue_mem_control(
        std::size_t queue, IcuControlInstruction instruction)
    {
        mem_iq(queue).append_control(instruction);
    }

    void enqueue_mem_control(
        Hemisphere hemisphere,
        std::size_t mem_slice,
        IcuControlInstruction instruction)
    {
        enqueue_mem_control(mem_queue(hemisphere, mem_slice), instruction);
    }

    void enqueue_mem_nop(std::size_t queue, std::size_t cycles)
    {
        if (cycles != 0) enqueue_mem_control(
            queue, IcuControlInstruction::Nop(cycles));
    }

    void enqueue_mem_repeat(
        std::size_t queue,
        std::size_t count,
        std::size_t interval = 1,
        std::int64_t address_stride = 0)
    {
        if (count != 0) enqueue_mem_control(
            queue,
            IcuControlInstruction::Repeat(
                count, interval, address_stride));
    }

    void enqueue_mxm(std::size_t mxm, MxmControlInstruction instruction)
    {
        const auto port = instruction.opcode == MxmControlOpcode::Compute
            ? MxmIcuPort::Compute
            : instruction.opcode
                    == MxmControlOpcode::ActivationDequantize
                ? MxmIcuPort::Dequant
                : MxmIcuPort::Load;
        mxm_iq(mxm, port).append_program(std::move(instruction));
    }

    void enqueue_mxm_control(
        std::size_t mxm,
        MxmIcuPort port,
        IcuControlInstruction instruction)
    {
        mxm_iq(mxm, port).append_control(instruction);
    }

    void enqueue_mxm_nop(std::size_t mxm, std::size_t cycles)
    {
        enqueue_mxm_load_nop(mxm, cycles);
        enqueue_mxm_compute_nop(mxm, cycles);
        if (cycles != 0) enqueue_mxm_control(
            mxm, MxmIcuPort::Dequant,
            IcuControlInstruction::Nop(cycles));
    }

    void enqueue_mxm_load_nop(std::size_t mxm, std::size_t cycles)
    {
        if (cycles != 0) enqueue_mxm_control(
            mxm, MxmIcuPort::Load, IcuControlInstruction::Nop(cycles));
    }

    void enqueue_mxm_compute_nop(std::size_t mxm, std::size_t cycles)
    {
        if (cycles != 0) enqueue_mxm_control(
            mxm, MxmIcuPort::Compute, IcuControlInstruction::Nop(cycles));
    }

    void enqueue_mxm_repeat(
        std::size_t mxm,
        MxmIcuPort port,
        std::size_t count,
        std::size_t interval = 1)
    {
        if (count != 0) enqueue_mxm_control(
            mxm, port, IcuControlInstruction::Repeat(count, interval));
    }

    void enqueue_mxm_load_repeat(
        std::size_t mxm, std::size_t count, std::size_t interval = 1)
    {
        enqueue_mxm_repeat(mxm, MxmIcuPort::Load, count, interval);
    }

    void enqueue_mxm_compute_repeat(
        std::size_t mxm, std::size_t count, std::size_t interval = 1)
    {
        enqueue_mxm_repeat(mxm, MxmIcuPort::Compute, count, interval);
    }

    void enqueue_sxm(Hemisphere hemisphere, SxmInstruction instruction)
    {
        sxm_iq(hemisphere_index(hemisphere), sxm_port(instruction)).append_program(
            std::move(instruction));
    }

    void enqueue_sxm_control(
        std::size_t hemisphere,
        SxmIcuPort port,
        IcuControlInstruction instruction)
    {
        sxm_iq(hemisphere, port).append_control(instruction);
    }

    void enqueue_sxm_nop(Hemisphere hemisphere, std::size_t cycles)
    {
        if (cycles != 0) {
            for (std::size_t port = 0;
                 port < kSxmIcusPerHemisphere; ++port) {
                enqueue_sxm_control(
                    hemisphere_index(hemisphere),
                    static_cast<SxmIcuPort>(port),
                    IcuControlInstruction::Nop(cycles));
            }
        }
    }

    void enqueue_sxm_repeat(
        Hemisphere hemisphere,
        std::size_t count,
        std::size_t interval = 1)
    {
        if (count != 0) {
            for (std::size_t port = 0;
                 port < kSxmIcusPerHemisphere; ++port) {
                enqueue_sxm_control(
                    hemisphere_index(hemisphere),
                    static_cast<SxmIcuPort>(port),
                    IcuControlInstruction::Repeat(count, interval));
            }
        }
    }

    void notify_vxm(std::size_t alu) { vxm_iq(alu).notify(); }
    void notify_mem(std::size_t queue) { mem_iq(queue).notify(); }
    void notify_mxm(std::size_t mxm, MxmIcuPort port)
    {
        mxm_iq(mxm, port).notify();
    }

    void advance_barrier_events()
    {
        for (auto& remaining : barrier_events_) {
            if (remaining > 0) --remaining;
        }
        while (!barrier_events_.empty() && barrier_events_.front() == 0) {
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
        for (auto& iq : vxm_iqs_) iq.notify();
        for (auto& iq : mem_iqs_) iq.notify();
        for (auto& ports : mxm_iqs_) for (auto& iq : ports) iq.notify();
        for (auto& ports : sxm_iqs_) for (auto& iq : ports) iq.notify();
    }

    std::size_t barrier_latency_cycles() const noexcept
    {
        return barrier_latency_cycles_;
    }

    std::size_t pending_barrier_event_count() const noexcept
    {
        return barrier_events_.size();
    }

    void print_diagnostic_status(std::ostream& os) const
    {
        os << "ICU diagnostic cycle=" << cycle_
           << " barrier_events=" << barrier_events_.size() << '\n';
        const auto print_queue = [&](const char* kind,
                                     std::size_t unit,
                                     std::size_t index,
                                     const auto& queue) {
            if (queue.done()) return;
            os << "  unfinished " << kind << " unit=" << unit
               << " index=" << index
               << " imem=" << queue.imem_occupancy()
               << " iq=" << queue.iq_occupancy()
               << " pending_fetch=" << queue.pending_fetch_count()
               << " fetch_pc=" << queue.fetch_pc()
               << " pending_cycles=" << queue.pending_issue_cycles()
               << " blocked_sync=" << queue.blocked_on_sync()
               << " underflow=" << queue.underflowed() << '\n';
        };
        for (std::size_t alu = 0; alu < kVxmQueues; ++alu)
            print_queue("VXM", 0, alu, vxm_iqs_[alu]);
        for (std::size_t mem = 0; mem < kMemQueues; ++mem)
            print_queue("MEM", 0, mem, mem_iqs_[mem]);
        for (std::size_t mxm = 0; mxm < kMxmUnitCount; ++mxm)
            for (std::size_t port = 0; port < kMxmIcusPerUnit; ++port)
                print_queue("MXM", mxm, port, mxm_iqs_[mxm][port]);
        for (std::size_t side = 0; side < kSxmQueues; ++side)
            for (std::size_t port = 0;
                 port < kSxmIcusPerHemisphere; ++port)
                print_queue("SXM", side, port, sxm_iqs_[side][port]);
    }

    VxmIcu& vxm_iq(std::size_t alu)
    {
        check_vxm_queue(alu);
        return vxm_iqs_[alu];
    }
    const VxmIcu& vxm_iq(std::size_t alu) const
    {
        check_vxm_queue(alu);
        return vxm_iqs_[alu];
    }
    MemIcu& mem_iq(std::size_t queue)
    {
        check_mem_queue(queue);
        return mem_iqs_[queue];
    }
    const MemIcu& mem_iq(std::size_t queue) const
    {
        check_mem_queue(queue);
        return mem_iqs_[queue];
    }
    MxmIcu& mxm_iq(std::size_t mxm, MxmIcuPort port)
    {
        check_mxm_queue(mxm);
        return mxm_iqs_[mxm][static_cast<std::size_t>(port)];
    }
    const MxmIcu& mxm_iq(std::size_t mxm, MxmIcuPort port) const
    {
        check_mxm_queue(mxm);
        return mxm_iqs_[mxm][static_cast<std::size_t>(port)];
    }
    SxmIcu& sxm_iq(std::size_t hemisphere, SxmIcuPort port)
    {
        check_sxm_queue(hemisphere);
        return sxm_iqs_[hemisphere][static_cast<std::size_t>(port)];
    }
    const SxmIcu& sxm_iq(
        std::size_t hemisphere, SxmIcuPort port) const
    {
        check_sxm_queue(hemisphere);
        return sxm_iqs_[hemisphere][static_cast<std::size_t>(port)];
    }

    void dispatch_vxm(VxmSlice& vxm, std::ostream* os = nullptr)
    {
        log_cycle_header(os);
        bool any = false;
        for (std::size_t alu = 0; alu < kVxmQueues; ++alu) {
            const auto instruction = vxm_iqs_[alu].tick();
            if (!instruction.has_value()) continue;
            vxm.issue_south(alu, *instruction);
            any = true;
            if (os != nullptr) {
                *os << "  ICU -> VXM alu" << alu << ' '
                    << describe_vxm(alu, *instruction) << '\n';
            }
        }
        log_dispatch_idle(os, any);
        ++cycle_;
    }

    void dispatch(
        TileArrayModel& mem,
        VxmSlice& vxm,
        std::array<Mxm, kMxmUnitCount>& mxms,
        std::ostream* os = nullptr)
    {
        dispatch_impl(&mem, nullptr, vxm, nullptr, mxms, nullptr, os);
    }

    void dispatch(
        TileArrayModel& mem,
        VxmSlice& vxm,
        std::array<SxmSlice, kSxmQueues>& sxms,
        std::array<Mxm, kMxmUnitCount>& mxms,
        std::ostream* os = nullptr)
    {
        dispatch_impl(&mem, nullptr, vxm, &sxms, mxms, nullptr, os);
    }

    void dispatch(
        std::array<TileArrayModel, hw::kHemispheres>& mems,
        VxmSlice& vxm,
        std::array<SxmSlice, kSxmQueues>& sxms,
        std::array<Mxm, kMxmUnitCount>& mxms,
        std::ostream* os = nullptr)
    {
        dispatch_impl(nullptr, &mems, vxm, &sxms, mxms, nullptr, os);
    }

    void dispatch(
        std::array<TileArrayModel, hw::kHemispheres>& mems,
        VxmSlice& vxm,
        std::array<SxmSlice, kSxmQueues>& sxms,
        std::array<Mxm, kMxmUnitCount>& mxms,
        std::array<std::optional<MxmActivationDequantizer>, kMxmUnitCount>&
            dequantizers,
        std::ostream* os = nullptr)
    {
        dispatch_impl(
            nullptr, &mems, vxm, &sxms, mxms, &dequantizers, os);
    }

    std::size_t cycle() const noexcept { return cycle_; }

    struct Statistics {
        std::size_t programmed_instructions{0};
        std::size_t fetched_instructions{0};
        std::size_t functional_issues{0};
        std::size_t underflowed_queues{0};
        std::size_t unfinished_queues{0};
    };

    Statistics statistics() const
    {
        auto result = Statistics{};
        const auto add = [&](const auto& queue) {
            result.programmed_instructions += queue.imem_occupancy();
            result.fetched_instructions += queue.fetched_count();
            result.functional_issues += queue.issued_count();
            result.underflowed_queues += queue.underflowed() ? 1 : 0;
            result.unfinished_queues += queue.done() ? 0 : 1;
        };
        for (const auto& queue : vxm_iqs_) add(queue);
        for (const auto& queue : mem_iqs_) add(queue);
        for (const auto& ports : mxm_iqs_)
            for (const auto& queue : ports) add(queue);
        for (const auto& ports : sxm_iqs_)
            for (const auto& queue : ports) add(queue);
        return result;
    }

    struct CycleActivity {
        std::size_t imem_read_starts{0};
        std::size_t iq_arrivals{0};
        std::size_t dispatched_entries{0};
        std::size_t waiting_queues{0};
    };

    CycleActivity last_cycle_activity() const
    {
        auto result = CycleActivity{};
        const auto add = [&](const auto& queue) {
            const auto& trace = queue.last_trace();
            result.imem_read_starts += trace.fetch_started_pc.has_value();
            result.iq_arrivals += trace.fetch_completed_pc.has_value();
            result.dispatched_entries += trace.issue_pc.has_value();
            result.waiting_queues +=
                trace.action == IcuQueueAction::NopWait
                || trace.action == IcuQueueAction::RepeatWait
                || trace.action == IcuQueueAction::SyncWait;
        };
        for (const auto& queue : vxm_iqs_) add(queue);
        for (const auto& queue : mem_iqs_) add(queue);
        for (const auto& ports : mxm_iqs_)
            for (const auto& queue : ports) add(queue);
        for (const auto& ports : sxm_iqs_)
            for (const auto& queue : ports) add(queue);
        return result;
    }

private:
    void dispatch_impl(
        TileArrayModel* east_mem,
        std::array<TileArrayModel, hw::kHemispheres>* mems,
        VxmSlice& vxm,
        std::array<SxmSlice, kSxmQueues>* sxms,
        std::array<Mxm, kMxmUnitCount>& mxms,
        std::array<std::optional<MxmActivationDequantizer>, kMxmUnitCount>*
            dequantizers,
        std::ostream* os)
    {
        log_cycle_header(os);
        bool any = false;
        for (std::size_t alu = 0; alu < kVxmQueues; ++alu) {
            const auto instruction = vxm_iqs_[alu].tick();
            if (!instruction.has_value()) continue;
            vxm.issue_south(alu, *instruction);
            any = true;
            if (os != nullptr) *os << "  ICU -> VXM alu" << alu << ' '
                                   << describe_vxm(alu, *instruction) << '\n';
        }
        for (std::size_t queue = 0; queue < kMemQueues; ++queue) {
            const auto instruction = mem_iqs_[queue].tick();
            if (!instruction.has_value()) continue;
            const auto side = queue / kMemQueuesPerHemisphere;
            const auto slice = queue % kMemQueuesPerHemisphere;
            if (mems != nullptr) {
                (*mems)[side].enqueue_instruction(slice, *instruction);
            } else {
                if (side != hemisphere_index(Hemisphere::East)
                    || east_mem == nullptr) {
                    throw std::logic_error(
                        "ICU west MEM dispatch has no connected hemisphere");
                }
                east_mem->enqueue_instruction(slice, *instruction);
            }
            any = true;
            if (os != nullptr) *os << "  ICU -> MEM."
                << hemisphere_short_name(static_cast<Hemisphere>(side))
                << " q" << slice << ' ' << describe_mem(*instruction) << '\n';
        }
        for (std::size_t mxm = 0; mxm < kMxmUnitCount; ++mxm) {
            for (std::size_t port = 0; port < kMxmIcusPerUnit; ++port) {
                const auto instruction = mxm_iqs_[mxm][port].tick();
                if (!instruction.has_value()) continue;
                if (instruction->opcode
                    == MxmControlOpcode::ActivationDequantize) {
                    if (dequantizers == nullptr
                        || !(*dequantizers)[mxm].has_value()) {
                        throw std::logic_error(
                            "ICU dispatched MXM activation dequantize without a connected unit");
                    }
                    (*dequantizers)[mxm]->issue_south();
                } else {
                    mxms[mxm].control().issue_south(*instruction);
                }
                any = true;
                if (os != nullptr) *os << "  ICU -> MXM" << mxm
                    << (port == static_cast<std::size_t>(MxmIcuPort::Load)
                            ? ".load "
                            : port == static_cast<std::size_t>(MxmIcuPort::Compute)
                                ? ".compute " : ".dequant ")
                    << describe_mxm(*instruction) << '\n';
            }
        }
        for (std::size_t side = 0; side < kSxmQueues; ++side) {
            for (std::size_t port = 0;
                 port < kSxmIcusPerHemisphere; ++port) {
                const auto instruction = sxm_iqs_[side][port].tick();
                if (!instruction.has_value()) continue;
                if (sxms == nullptr) {
                    throw std::logic_error(
                        "ICU dispatched SXM without a connected SXM slice");
                }
                (*sxms)[side].issue(*instruction);
                any = true;
                if (os != nullptr) *os << "  ICU -> SXM."
                    << hemisphere_short_name(static_cast<Hemisphere>(side))
                    << " port=" << port
                    << " opcode=" << static_cast<std::size_t>(instruction->opcode)
                    << '\n';
            }
        }
        log_dispatch_idle(os, any);
        ++cycle_;
    }

    std::size_t take_emitted_notifications()
    {
        std::size_t count = 0;
        for (auto& iq : vxm_iqs_) count += iq.take_notify() ? 1 : 0;
        for (auto& iq : mem_iqs_) count += iq.take_notify() ? 1 : 0;
        for (auto& ports : mxm_iqs_)
            for (auto& iq : ports) count += iq.take_notify() ? 1 : 0;
        for (auto& ports : sxm_iqs_)
            for (auto& iq : ports) count += iq.take_notify() ? 1 : 0;
        return count;
    }

    template <typename QueueArray>
    static void reset_all(QueueArray& queues)
    {
        for (auto& queue : queues) queue.reset();
    }

    static void check_vxm_queue(std::size_t index)
    {
        if (index >= kVxmQueues) throw std::out_of_range("ICU VXM queue index");
    }
    static void check_mem_queue(std::size_t index)
    {
        if (index >= kMemQueues) throw std::out_of_range("ICU MEM queue index");
    }
    static void check_mxm_queue(std::size_t index)
    {
        if (index >= kMxmUnitCount) throw std::out_of_range("ICU MXM queue index");
    }
    static void check_sxm_queue(std::size_t index)
    {
        if (index >= kSxmQueues) throw std::out_of_range("ICU SXM queue index");
    }

    static SxmIcuPort sxm_port(const SxmInstruction& instruction)
    {
        switch (instruction.opcode) {
        case SxmOpcode::Transpose: return SxmIcuPort::Transpose;
        case SxmOpcode::Permute: return SxmIcuPort::Permute;
        case SxmOpcode::Distribute:
        case SxmOpcode::ShiftSelect: return SxmIcuPort::Stream;
        }
        throw std::logic_error("unknown SXM opcode for ICU port");
    }

    template <typename QueueArray>
    static std::size_t queued_instruction_count(const QueueArray& queues)
    {
        std::size_t count = 0;
        for (const auto& queue : queues) count += queue.queued_count();
        return count;
    }

    std::size_t queued_mxm_instruction_count() const
    {
        std::size_t count = 0;
        for (const auto& ports : mxm_iqs_)
            count += queued_instruction_count(ports);
        return count;
    }

    void log_cycle_header(std::ostream* os) const
    {
        if (os == nullptr) return;
        *os << "icu cycle " << cycle_ << '\n'
            << "  queues: vxm=" << queued_instruction_count(vxm_iqs_)
            << " mem=" << queued_instruction_count(mem_iqs_)
            << " mxm=" << queued_mxm_instruction_count()
            << " sxm=" << queued_sxm_instruction_count() << '\n';
    }
    static void log_dispatch_idle(std::ostream* os, bool any)
    {
        if (os != nullptr && !any) *os << "  ICU dispatch idle\n";
    }

    std::size_t queued_sxm_instruction_count() const
    {
        std::size_t count = 0;
        for (const auto& ports : sxm_iqs_)
            count += queued_instruction_count(ports);
        return count;
    }

    static std::string describe_mem(const MemInstruction& instruction)
    {
        std::ostringstream os;
        const char* opcode = "?";
        switch (instruction.opcode) {
        case MemOpcode::Read: opcode = "Read"; break;
        case MemOpcode::Write: opcode = "Write"; break;
        case MemOpcode::ReadWrite: opcode = "ReadWrite"; break;
        case MemOpcode::Gather: opcode = "Gather"; break;
        case MemOpcode::Scatter: opcode = "Scatter"; break;
        }
        os << opcode << " addr=b" << instruction.address.bank()
           << ":w" << instruction.address.word()
           << " stream=" << instruction.stream;
        return os.str();
    }

    static std::string describe_mxm(const MxmControlInstruction& instruction)
    {
        std::ostringstream os;
        if (instruction.opcode == MxmControlOpcode::IW) {
            os << "IW b" << instruction.weight_buffer
               << " load_mode=" << static_cast<int>(instruction.weight_load_mode);
        } else if (instruction.opcode == MxmControlOpcode::LoadScales) {
            os << "LoadScales b" << instruction.weight_buffer;
        } else if (
            instruction.opcode
            == MxmControlOpcode::ActivationDequantize) {
            os << "ActivationDequantize";
        } else {
            os << "Compute b" << instruction.weight_buffer
               << " out=" << instruction.stream_base
               << " partial=" << instruction.partial_stream_base
               << " acc_mode=" << static_cast<int>(instruction.accumulator_mode)
               << " pair=" << static_cast<int>(instruction.pair_mode);
        }
        return os.str();
    }

    static std::string describe_vxm(
        std::size_t alu, const VxmCompactInstruction& packet)
    {
        const auto instruction =
            VxmCompactInstructionCodec::decode(alu, packet).instruction;
        std::ostringstream os;
        os << VxmLane::operation_name(instruction.operation);
        if (instruction.output_stream.has_value())
            os << " out_stream=" << *instruction.output_stream;
        return os.str();
    }

    std::array<VxmIcu, kVxmQueues> vxm_iqs_{};
    std::array<MemIcu, kMemQueues> mem_iqs_{};
    std::array<std::array<MxmIcu, kMxmIcusPerUnit>, kMxmUnitCount> mxm_iqs_{};
    std::array<
        std::array<SxmIcu, kSxmIcusPerHemisphere>,
        kSxmQueues> sxm_iqs_{};
    std::size_t barrier_latency_cycles_{hw::kIcuBarrierLatencyCycles};
    std::deque<std::size_t> barrier_events_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
