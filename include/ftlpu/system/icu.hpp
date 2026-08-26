#pragma once

#include "ftlpu/c2c/dma.hpp"
#include "ftlpu/c2c/slice.hpp"
#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/hemisphere.hpp"
#include "ftlpu/icu/distributed_queue.hpp"
#include "ftlpu/icu/location.hpp"
#include "ftlpu/mem/tile_array.hpp"
#include "ftlpu/mxm/mxm.hpp"
#include "ftlpu/sxm/slice.hpp"
#include "ftlpu/vxm/compact_instruction.hpp"
#include "ftlpu/vxm/slice.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <ostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ftlpu {

class InstructionControlUnit {
public:
    static constexpr std::size_t kVxmQueues = VxmSlice::kAluQueues;
    static constexpr std::size_t kMemQueuesPerHemisphere =
        hw::kMemSliceColumns * hw::kMemBanksPerSlice;
    static constexpr std::size_t kMxmQueuesPerHemisphere =
        hw::kMxmsPerHemisphere;
    static constexpr std::size_t kSxmQueuesPerHemisphere = 2;
    static constexpr std::size_t kMemQueues = hw::kHemispheres * kMemQueuesPerHemisphere;
    static constexpr std::size_t kMxmQueues = hw::kMxmCount;
    static constexpr std::size_t kSxmQueues = hw::kHemispheres * kSxmQueuesPerHemisphere;

    static constexpr std::size_t mem_queue(
        Hemisphere hemisphere,
        std::size_t mem_slice,
        std::size_t bank)
    {
        return hemisphere_index(hemisphere) * kMemQueuesPerHemisphere
            + mem_slice * hw::kMemBanksPerSlice + bank;
    }

    static constexpr std::size_t mem_queue(
        Hemisphere hemisphere,
        std::size_t mem_slice)
    {
        return mem_queue(hemisphere, mem_slice, 0);
    }

    static constexpr std::size_t mxm_queue(Hemisphere hemisphere, std::size_t local_mxm)
    {
        return hemisphere_index(hemisphere) * kMxmQueuesPerHemisphere + local_mxm;
    }

    using Repeat = IcuRepeat;
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
    using C2cIcu = DistributedIcuQueue<
        C2cInstruction,
        hw::kIcuC2cInstructionBits,
        hw::kIcuC2cImemDepth,
        hw::kIcuC2cIqDepth,
        hw::kIcuFetchLatencyCycles>;
    using C2cDmaIcu = DistributedIcuQueue<
        C2cDmaInstruction,
        hw::kIcuC2cDmaInstructionBits,
        hw::kIcuC2cImemDepth,
        hw::kIcuC2cIqDepth,
        hw::kIcuFetchLatencyCycles>;

    explicit InstructionControlUnit(
        std::size_t barrier_latency_cycles = hw::kIcuBarrierLatencyCycles)
        : barrier_latency_cycles_(barrier_latency_cycles)
    {
    }
    void reset()
    {
        for (auto& queue : vxm_queues_) queue.reset();
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
        for (auto& queue : c2c_tx_queues_) queue.reset();
        for (auto& queue : c2c_dma_queues_) queue.reset();
        for (auto& queue : c2c_rx_queues_) queue.reset();
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
        for (auto& queue : c2c_tx_queues_) queue.push_nop(cycles);
        for (auto& queue : c2c_dma_queues_) queue.push_nop(cycles);
        for (auto& queue : c2c_rx_queues_) queue.push_nop(cycles);
    }

    void enqueue_control(
        IcuLocation location,
        IcuControlInstruction instruction)
    {
        switch (location.kind) {
        case IcuLocationKind::Mem:
            mem_queues_[mem_queue(
                static_cast<Hemisphere>(location.unit),
                location.index,
                location.bank)].append_control(instruction);
            return;
        case IcuLocationKind::Vxm:
            check_vxm_queue(location.index);
            vxm_queues_[location.index].append_control(instruction);
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
        case IcuLocationKind::C2cTx:
            if (location.unit >= hw::kHemispheres) {
                throw std::out_of_range(
                    "ICU C2C TX hemisphere is outside the chip");
            }
            c2c_tx_queues_[location.unit].append_control(instruction);
            return;
        case IcuLocationKind::C2cRx:
            if (location.unit >= hw::kHemispheres) {
                throw std::out_of_range(
                    "ICU C2C RX hemisphere is outside the chip");
            }
            c2c_rx_queues_[location.unit].append_control(instruction);
            return;
        case IcuLocationKind::C2cDma:
            if (location.unit >= hw::kHemispheres) {
                throw std::out_of_range(
                    "ICU C2C DMA hemisphere is outside the chip");
            }
            c2c_dma_queues_[location.unit].append_control(instruction);
            return;
        }
        throw std::logic_error("unknown ICU location kind");
    }
    void enqueue_vxm(
        std::size_t alu,
        VxmCompactInstruction instruction)
    {
        check_vxm_queue(alu);
        vxm_queues_[alu].push_instruction(std::move(instruction));
    }

    void enqueue_vxm(
        std::size_t alu,
        VxmChainDepth depth,
        const VxmLaneAluInstruction& instruction)
    {
        enqueue_vxm(
            alu,
            VxmCompactInstructionCodec::encode(
                alu, depth, instruction));
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

    void enqueue_mem_macro(std::size_t column,
        IcuMacroSchedule schedule, MemInstruction instruction)
    {
        check_mem_queue(column);
        mem_queues_[column].push_macro(schedule, std::move(instruction));
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

    void enqueue_mxm_load_macro(std::size_t mxm,
        IcuMacroSchedule schedule, MxmControlInstruction instruction)
    {
        check_mxm_queue(mxm);
        if (instruction.opcode != MxmControlOpcode::IW
            && !(instruction.opcode == MxmControlOpcode::Decode
                && instruction.decode_operation
                    == MxmDecodeOperation::LoadActivation))
            throw std::invalid_argument(
                "MXM load macro requires an IW/load-activation instruction");
        mxm_load_queues_[mxm].push_macro(
            schedule, std::move(instruction));
    }

    void enqueue_mxm_dequant_macro(std::size_t mxm,
        IcuMacroSchedule schedule, MxmDequantInstruction instruction)
    {
        check_mxm_queue(mxm);
        mxm_dequant_queues_[mxm].push_macro(
            schedule, std::move(instruction));
    }

    void enqueue_mxm_compute_macro(std::size_t mxm,
        IcuMacroSchedule schedule, MxmControlInstruction instruction)
    {
        check_mxm_queue(mxm);
        if (instruction.opcode == MxmControlOpcode::IW
            || (instruction.opcode == MxmControlOpcode::Decode
                && instruction.decode_operation
                    == MxmDecodeOperation::LoadActivation))
            throw std::invalid_argument(
                "MXM compute macro cannot carry a load instruction");
        mxm_compute_queues_[mxm].push_macro(
            schedule, std::move(instruction));
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

    void enqueue_c2c(
        Hemisphere endpoint_hemisphere,
        C2cInstruction instruction)
    {
        const auto endpoint = hemisphere_index(endpoint_hemisphere);
        if (instruction.opcode == C2cOpcode::Send) {
            c2c_tx_queues_[endpoint].push_instruction(std::move(instruction));
        } else {
            c2c_rx_queues_[endpoint].push_instruction(std::move(instruction));
        }
    }

    void enqueue_c2c_send(
        Hemisphere endpoint_hemisphere,
        std::size_t stream_index)
    {
        enqueue_c2c(
            endpoint_hemisphere,
            C2cInstruction::Send(stream_index));
    }

    void enqueue_c2c_receive(
        Hemisphere endpoint_hemisphere,
        std::size_t stream_index,
        Hemisphere consumer_hemisphere,
        std::size_t consumer_mem_slice,
        std::size_t consumer_mem_bank = 0,
        bool notify_mem = true,
        std::size_t base_row = 0,
        std::size_t vector_count = 1,
        std::size_t row_stride = 1,
        std::size_t fabric_stream_index =
            std::numeric_limits<std::size_t>::max())
    {
        enqueue_c2c(endpoint_hemisphere, C2cInstruction::Receive(
            stream_index, consumer_hemisphere, consumer_mem_slice,
            consumer_mem_bank, notify_mem, base_row, vector_count,
            row_stride, fabric_stream_index));
    }

    void enqueue_c2c_tx_nop(
        Hemisphere endpoint_hemisphere,
        std::size_t cycles)
    {
        c2c_tx_queues_[hemisphere_index(endpoint_hemisphere)].push_nop(cycles);
    }

    void enqueue_c2c_rx_nop(
        Hemisphere endpoint_hemisphere,
        std::size_t cycles)
    {
        c2c_rx_queues_[hemisphere_index(endpoint_hemisphere)].push_nop(cycles);
    }

    void enqueue_c2c_dma(
        Hemisphere endpoint_hemisphere,
        C2cDmaInstruction instruction)
    {
        c2c_dma_queues_[hemisphere_index(endpoint_hemisphere)]
            .push_instruction(std::move(instruction));
    }

    void enqueue_c2c_dma_nop(
        Hemisphere endpoint_hemisphere,
        std::size_t cycles)
    {
        c2c_dma_queues_[hemisphere_index(endpoint_hemisphere)]
            .push_nop(cycles);
    }

    void notify(IcuLocation location)
    {
        switch (location.kind) {
        case IcuLocationKind::Mem:
            mem_iq(mem_queue(
                static_cast<Hemisphere>(location.unit),
                location.index,
                location.bank)).notify();
            return;
        case IcuLocationKind::Vxm:
            vxm_iq(location.index).notify();
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
        case IcuLocationKind::C2cTx:
            if (location.unit >= hw::kHemispheres) {
                throw std::out_of_range(
                    "ICU C2C TX hemisphere is outside the chip");
            }
            c2c_tx_queues_[location.unit].notify();
            return;
        case IcuLocationKind::C2cRx:
            if (location.unit >= hw::kHemispheres) {
                throw std::out_of_range(
                    "ICU C2C RX hemisphere is outside the chip");
            }
            c2c_rx_queues_[location.unit].notify();
            return;
        case IcuLocationKind::C2cDma:
            if (location.unit >= hw::kHemispheres) {
                throw std::out_of_range(
                    "ICU C2C DMA hemisphere is outside the chip");
            }
            c2c_dma_queues_[location.unit].notify();
            return;
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
        for (auto& queue : mem_queues_) queue.notify();
        for (auto& queue : mxm_load_queues_) queue.notify();
        for (auto& queue : mxm_dequant_queues_) queue.notify();
        for (auto& queue : mxm_compute_queues_) queue.notify();
        for (auto& queue : sxm_transpose_queues_) queue.notify();
        for (auto& queue : sxm_permute_queues_) queue.notify();
        for (auto& queue : c2c_tx_queues_) queue.notify();
        for (auto& queue : c2c_rx_queues_) queue.notify();
        for (auto& queue : c2c_dma_queues_) queue.notify();
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
            if (os != nullptr) {
                *os << "  ICU -> VXM.q" << alu << ' '
                    << describe_vxm(alu, *instruction) << '\n';
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
        std::ostream* os = nullptr,
        std::array<C2cEndpoint*, hw::kHemispheres> c2cs = {},
        std::array<C2cDmaEngine*, hw::kHemispheres> c2c_dmas = {})
    {
        log_cycle_header(os);

        bool any = false;
        for (std::size_t hemisphere = 0;
             hemisphere < hw::kHemispheres;
             ++hemisphere) {
            const auto c2c_tx = c2c_tx_queues_[hemisphere].dispatch_next();
            if (c2c_tx.has_value()) {
                if (c2cs[hemisphere] == nullptr) {
                    throw std::logic_error(
                        "ICU issued C2C TX without an attached hemisphere endpoint");
                }
                c2cs[hemisphere]->tx().issue(*c2c_tx);
                any = true;
                if (os != nullptr) {
                    *os << "  ICU -> C2C."
                        << hemisphere_short_name(
                               static_cast<Hemisphere>(hemisphere))
                        << ".tx Send stream="
                        << c2c_tx->stream_index << '\n';
                }
            }

            const auto c2c_rx = c2c_rx_queues_[hemisphere].dispatch_next();
            if (c2c_rx.has_value()) {
                if (c2cs[hemisphere] == nullptr) {
                    throw std::logic_error(
                        "ICU issued C2C RX without an attached hemisphere endpoint");
                }
                c2cs[hemisphere]->rx().issue(*c2c_rx);
                any = true;
                if (os != nullptr) {
                    *os << "  ICU -> C2C."
                        << hemisphere_short_name(
                               static_cast<Hemisphere>(hemisphere))
                        << ".rx Receive stream="
                        << c2c_rx->stream_index << " consumer=MEM."
                        << hemisphere_short_name(c2c_rx->consumer.hemisphere)
                        << '.' << c2c_rx->consumer.mem_slice << '\n';
                }
            }

            const auto c2c_dma =
                c2c_dma_queues_[hemisphere].dispatch_next();
            if (c2c_dma.has_value()) {
                if (c2c_dmas[hemisphere] == nullptr) {
                    throw std::logic_error(
                        "ICU issued C2C DMA without an attached DMA engine");
                }
                c2c_dmas[hemisphere]->issue(*c2c_dma);
                any = true;
                if (os != nullptr) {
                    *os << "  ICU -> C2C."
                        << hemisphere_short_name(
                               static_cast<Hemisphere>(hemisphere))
                        << ".dma "
                        << (c2c_dma->direction
                                    == C2cDmaDirection::Ddr4ToC2c
                                ? "Load"
                                : "Store")
                        << " ddr4=" << c2c_dma->ddr4_address
                        << " vectors=" << c2c_dma->vector_count << std::endl;
                }
            }
        }
        for (std::size_t alu = 0; alu < kVxmQueues; ++alu) {
            const auto instruction = vxm_queues_[alu].dispatch_next();
            if (!instruction.has_value()) continue;
            vxm.issue_south(alu, *instruction);
            any = true;
            if (os != nullptr) {
                *os << "  ICU -> VXM.q" << alu << ' '
                    << describe_vxm(alu, *instruction) << '\n';
            }
        }

        for (std::size_t queue = 0; queue < kMemQueues; ++queue) {
            const auto instruction = mem_queues_[queue].dispatch_next();
            if (!instruction.has_value()) {
                continue;
            }
            const auto hemisphere = queue / kMemQueuesPerHemisphere;
            const auto local_queue = queue % kMemQueuesPerHemisphere;
            const auto mem_slice = local_queue / hw::kMemBanksPerSlice;
            const auto bank = local_queue % hw::kMemBanksPerSlice;
            mems[hemisphere].enqueue_instruction(mem_slice, bank, *instruction);
            any = true;
            if (os != nullptr) {
                *os << "  ICU -> MEM." << hemisphere_short_name(static_cast<Hemisphere>(hemisphere))
                    << ".c" << mem_slice << ".b" << bank << ' '
                    << describe_mem(*instruction) << '\n';
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

    C2cIcu& c2c_tx_iq(Hemisphere hemisphere) noexcept
    {
        return c2c_tx_queues_[hemisphere_index(hemisphere)];
    }
    C2cIcu& c2c_rx_iq(Hemisphere hemisphere) noexcept
    {
        return c2c_rx_queues_[hemisphere_index(hemisphere)];
    }

    C2cDmaIcu& c2c_dma_iq(Hemisphere hemisphere) noexcept
    {
        return c2c_dma_queues_[hemisphere_index(hemisphere)];
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
        collect(mem_queues_);
        collect(mxm_load_queues_);
        collect(mxm_dequant_queues_);
        collect(mxm_compute_queues_);
        collect(sxm_transpose_queues_);
        collect(sxm_permute_queues_);
        collect(c2c_tx_queues_);
        collect(c2c_rx_queues_);
        collect(c2c_dma_queues_);
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
        if (alu >= kVxmQueues) {
            throw std::out_of_range(
                "ICU VXM queue is outside the 8 compact control queues");
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
            << " c2c_tx=" << queued_instruction_count(c2c_tx_queues_)
            << " c2c_rx=" << queued_instruction_count(c2c_rx_queues_)
            << " c2c_dma="
            << queued_instruction_count(c2c_dma_queues_)
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
               << " col=" << instruction.weight_column
               << " stream=" << instruction.weight_stream_base;
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
               << " out=" << instruction.stream_base;
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
               << " out=" << instruction.stream_base;
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

    static std::string describe_vxm(
        std::size_t stage,
        const VxmCompactInstruction& packet)
    {
        const auto decoded =
            VxmCompactInstructionCodec::decode(stage, packet);
        const auto& instruction = decoded.instruction;
        std::ostringstream os;
        os << VxmLane::operation_name(instruction.operation)
           << " depth=" << static_cast<std::size_t>(decoded.chain_depth);
        if (instruction.output_stream.has_value()) {
            os << " output=" << *instruction.output_stream;
        }
        return os.str();
    }

    std::array<VxmIcu, kVxmQueues> vxm_queues_{};
    std::array<MemIcu, kMemQueues> mem_queues_{};
    std::array<MxmIcu, kMxmQueues> mxm_load_queues_{};
    std::array<MxmDequantIcu, kMxmQueues>
        mxm_dequant_queues_{};
    std::array<MxmIcu, kMxmQueues> mxm_compute_queues_{};
    std::array<SxmIcu, hw::kHemispheres> sxm_transpose_queues_{};
    std::array<SxmIcu, hw::kHemispheres> sxm_permute_queues_{};
    std::array<C2cIcu, hw::kHemispheres> c2c_tx_queues_{};
    std::array<C2cDmaIcu, hw::kHemispheres> c2c_dma_queues_{};
    std::array<C2cIcu, hw::kHemispheres> c2c_rx_queues_{};
    std::size_t barrier_latency_cycles_{hw::kIcuBarrierLatencyCycles};
    std::deque<std::size_t> barrier_events_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
