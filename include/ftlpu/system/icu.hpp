#pragma once

#include "ftlpu/c2c/dma.hpp"
#include "ftlpu/c2c/icu_instruction.hpp"
#include "ftlpu/c2c/slice.hpp"
#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/hemisphere.hpp"
#include "ftlpu/icu/distributed_queue.hpp"
#include "ftlpu/icu/common_barrier.hpp"
#include "ftlpu/icu/fu_3d_instruction.hpp"
#include "ftlpu/icu/location.hpp"
#include "ftlpu/mem/mem_array.hpp"
#include "ftlpu/mxm/mxm.hpp"
#include "ftlpu/sxm/slice.hpp"
#include "ftlpu/vxm/compact_instruction.hpp"
#include "ftlpu/vxm/slice.hpp"

#include <algorithm>
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
#include <vector>

namespace ftlpu {

struct IcuFrontendStatistics {
    std::size_t imem_entries{0};
    std::size_t fetched_entries{0};
    std::size_t issued_instructions{0};
    std::size_t macro_queues{0};
    std::size_t peak_macro_contexts_per_queue{0};
    std::size_t decoded_macro_contexts{0};
    std::size_t macro_decoder_bit_wait_cycles{0};
    std::size_t macro_decoder_context_stall_cycles{0};
    std::size_t peak_macro_reservoir_bits{0};
    std::size_t macro_frontend_cycles{0};
    std::size_t macro_fetched_words{0};
    std::size_t macro_payload_bits_consumed{0};
    std::size_t macro_reservoir_empty_cycles{0};
    std::size_t macro_reservoir_full_cycles{0};
    std::size_t macro_reservoir_occupancy_bit_samples{0};
    std::size_t macro_descriptor_count{0};
    std::size_t macro_mem_descriptor_count{0};
    std::size_t macro_mxm_descriptor_count{0};
    std::size_t macro_descriptor_decode_cycles{0};
    std::size_t macro_max_descriptor_decode_cycles{0};
    std::size_t macro_decoder_active_cycles{0};
    std::size_t macro_decoder_starvation_cycles{0};
    std::size_t macro_decoder_ddb_stall_cycles{0};
    std::size_t macro_ddb_entries{0};
    std::size_t peak_macro_ddb_occupancy{0};
    std::size_t macro_ddb_full_cycles{0};
    std::size_t macro_ddb_residency_cycles{0};
    std::size_t macro_ddb_occupancy_samples{0};
    std::size_t macro_max_ddb_residency_cycles{0};
    std::size_t macro_contexts_generated{0};
    std::size_t macro_timing_window_block_cycles{0};
    std::size_t macro_active_ram_full_cycles{0};
    std::size_t macro_capacity_feasible_wait_cycles{0};
    std::size_t macro_admission_count{0};
    std::size_t macro_admission_stall_cycles{0};
    std::size_t macro_admission_to_start_cycles{0};
    std::size_t macro_max_admission_to_start_cycles{0};
    std::size_t macro_active_occupancy_samples{0};
    std::size_t macro_context_active_lifetime_cycles{0};
    std::size_t macro_max_context_active_lifetime_cycles{0};
};

struct IcuMacroQueueFrontendStatistics {
    IcuMacroQueueKind kind{IcuMacroQueueKind::Mem};
    std::size_t queue_index{0};
    std::size_t imem_entries{0};
    std::size_t fetched_entries{0};
    std::size_t issued_instructions{0};
    std::size_t peak_active_contexts{0};
    std::size_t active_context_capacity{0};
    std::size_t active_context_bits{0};
    std::size_t instruction_bits{0};
    std::size_t imem_read_latency{0};
    std::size_t imem_request_initiation_interval{0};
    std::size_t reservoir_words{0};
    std::size_t decode_window_bits{0};
    std::size_t ddb_depth{0};
    std::size_t ddb_run_capacity{0};
    std::size_t ddb_entry_bits{0};
    std::size_t admission_lookahead{0};
    std::size_t context_expand_width{0};
    IcuMacroDecoderStatistics decoder{};
};

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
        hw::kIcuFetchLatencyCycles,
        hw::kIcuVxmMacroContextDepth,
        IcuQueueRole::Vxm>;
    using MemIcu = DistributedIcuQueue<
        MemInstruction,
        hw::kIcuMemInstructionBits,
        hw::kIcuMemImemDepth,
        hw::kIcuMemIqDepth,
        hw::kIcuFetchLatencyCycles,
        hw::kIcuMemMacroContextDepth,
        IcuQueueRole::Mem>;
    using MxmLoadIcu = DistributedIcuQueue<
        MxmControlInstruction,
        hw::kIcuMxmInstructionBits,
        hw::kIcuMxmImemDepth,
        hw::kIcuMxmIqDepth,
        hw::kIcuFetchLatencyCycles,
        hw::kIcuMxmMacroContextDepth,
        IcuQueueRole::MxmLoad>;
    using MxmDequantIcu = DistributedIcuQueue<
        MxmDequantInstruction,
        hw::kIcuMxmInstructionBits,
        hw::kIcuMxmImemDepth,
        hw::kIcuMxmIqDepth,
        hw::kIcuFetchLatencyCycles,
        hw::kIcuMxmMacroContextDepth,
        IcuQueueRole::MxmDequant>;
    using MxmComputeIcu = DistributedIcuQueue<
        MxmControlInstruction,
        hw::kIcuMxmInstructionBits,
        hw::kIcuMxmImemDepth,
        hw::kIcuMxmIqDepth,
        hw::kIcuFetchLatencyCycles,
        hw::kIcuMxmMacroContextDepth,
        IcuQueueRole::MxmCompute>;
    using SxmTransposeIcu = DistributedIcuQueue<
        SxmInstruction,
        hw::kIcuSxmInstructionBits,
        hw::kIcuSxmImemDepth,
        hw::kIcuSxmIqDepth,
        hw::kIcuFetchLatencyCycles,
        hw::kIcuSxmMacroContextDepth,
        IcuQueueRole::SxmTranspose>;
    using SxmPermuteIcu = DistributedIcuQueue<
        SxmInstruction,
        hw::kIcuSxmInstructionBits,
        hw::kIcuSxmImemDepth,
        hw::kIcuSxmIqDepth,
        hw::kIcuFetchLatencyCycles,
        hw::kIcuSxmMacroContextDepth,
        IcuQueueRole::SxmPermute>;
    using C2cTxIcu = DistributedIcuQueue<
        C2cInstruction,
        hw::kIcuC2cInstructionBits,
        hw::kIcuC2cImemDepth,
        hw::kIcuC2cIqDepth,
        hw::kIcuFetchLatencyCycles,
        hw::kIcuC2cMacroContextDepth,
        IcuQueueRole::C2cTx>;
    using C2cRxIcu = DistributedIcuQueue<
        C2cInstruction,
        hw::kIcuC2cInstructionBits,
        hw::kIcuC2cImemDepth,
        hw::kIcuC2cIqDepth,
        hw::kIcuFetchLatencyCycles,
        hw::kIcuC2cMacroContextDepth,
        IcuQueueRole::C2cRx>;
    using C2cDmaIcu = DistributedIcuQueue<
        C2cDmaInstruction,
        hw::kIcuC2cInstructionBits,
        hw::kIcuC2cImemDepth,
        hw::kIcuC2cIqDepth,
        hw::kIcuFetchLatencyCycles,
        hw::kIcuC2cMacroContextDepth,
        IcuQueueRole::C2cDma>;
    using MxmIcu = MxmLoadIcu;
    using MemRawImemWord = MemIcu::RawImemWord;
    using MxmRawImemWord = MxmLoadIcu::RawImemWord;

    static_assert(MemIcu::macro_context_bits
        == hw::kIcuMemMacroContextBits);
    static_assert(MxmLoadIcu::macro_context_bits
        == hw::kIcuMxmMacroContextBits);
    static_assert(MxmDequantIcu::macro_context_bits
        == hw::kIcuMxmMacroContextBits);

    explicit InstructionControlUnit(
        std::size_t barrier_latency_cycles = hw::kIcuBarrierLatencyCycles)
        : barrier_latency_cycles_(barrier_latency_cycles)
    {
    }

    void write_mem_raw_macro_imem(
        std::size_t queue, std::size_t address, MemRawImemWord word)
    {
        mem_iq(queue).write_raw_imem(address, std::move(word));
    }

    void configure_mem_raw_macro_imem(
        std::size_t queue, std::size_t word_count)
    {
        mem_iq(queue).configure_raw_macro(
            IcuMacroQueueKind::Mem, word_count);
    }

    void write_mxm_load_raw_macro_imem(
        std::size_t mxm, std::size_t address, MxmRawImemWord word)
    {
        mxm_load_iq(mxm).write_raw_imem(address, std::move(word));
    }

    void configure_mxm_load_raw_macro_imem(
        std::size_t mxm, std::size_t word_count)
    {
        mxm_load_iq(mxm).configure_raw_macro(
            IcuMacroQueueKind::MxmLoad, word_count);
    }

    void write_mxm_compute_raw_macro_imem(
        std::size_t mxm, std::size_t address, MxmRawImemWord word)
    {
        mxm_compute_iq(mxm).write_raw_imem(address, std::move(word));
    }

    void configure_mxm_compute_raw_macro_imem(
        std::size_t mxm, std::size_t word_count)
    {
        mxm_compute_iq(mxm).configure_raw_macro(
            IcuMacroQueueKind::MxmCompute, word_count);
    }

    void write_mxm_dequant_raw_macro_imem(
        std::size_t mxm, std::size_t address, MxmRawImemWord word)
    {
        mxm_dequant_iq(mxm).write_raw_imem(address, std::move(word));
    }

    void configure_mxm_dequant_raw_macro_imem(
        std::size_t mxm, std::size_t word_count)
    {
        mxm_dequant_iq(mxm).configure_raw_macro(
            IcuMacroQueueKind::MxmDequant, word_count);
    }

    // Runs all raw Macro frontends in parallel while the program cycle remains
    // stopped. Runtime can call this after writing/committing all raw images
    // and before enabling the global program issue clock.
    std::size_t prime_raw_macro_frontends()
    {
        std::size_t cycles = 0;
        for (;;) {
            bool pending = false;
            const auto prime_queue = [&](auto& queue) {
                if (!queue.raw_macro_mode() || queue.raw_macro_ready()) return;
                queue.prefetch_raw_macro();
                pending = true;
            };
            for (auto& queue : mem_queues_) prime_queue(queue);
            for (auto& queue : mxm_load_queues_) prime_queue(queue);
            for (auto& queue : mxm_dequant_queues_) prime_queue(queue);
            for (auto& queue : mxm_compute_queues_) prime_queue(queue);
            if (!pending) return cycles;
            ++cycles;
        }
    }
    void reset()
    {
        for (auto& queue : vxm_queues_) queue.reset();
        for (auto& queue : mem_queues_) {
            queue.reset();
        }
        for (auto& queue : mem_transport_queues_) queue.reset();
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
        pending_c2c_dma_raw_words_.fill(std::nullopt);
        barrier_events_.clear();
        program_issue_enabled_ = true;
        common_barriers_.reset();
        compute_enabled_ = true;
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
            check_mem_location(location);
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
        vxm_queues_[alu].push_vxm_run_2d(
            VxmIcuRun2DInstruction::Run2D(
                0,
                {1, 1}, {1, 1}, std::move(instruction)));
    }

    void enqueue_vxm_run_2d(
        std::size_t alu, VxmIcuRun2DInstruction instruction)
    {
        check_vxm_queue(alu);
        vxm_queues_[alu].push_vxm_run_2d(std::move(instruction));
    }

    void enqueue_vxm(
        std::size_t alu,
        VxmChainDepth depth,
        const VxmLaneAluInstruction& instruction)
    {
        enqueue_vxm(
            alu,
            VxmCompactInstructionCodec::encode(alu, depth, instruction));
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

    void enqueue_vxm_stream_nd(std::size_t alu,
        IcuVxmStreamNdSchedule schedule,
        VxmCompactInstruction instruction)
    {
        check_vxm_queue(alu);
        vxm_queues_[alu].push_vxm_stream_nd(
            schedule, std::move(instruction));
    }

    void enqueue_mem(std::size_t column, MemInstruction instruction)
    {
        check_mem_queue(column);
        mem_queues_[column].push_instruction(instruction);
    }

    void enqueue_mem_3d(
        std::size_t column,
        MemIcuInstruction instruction)
    {
        check_mem_queue(column);
        mem_queues_[column].push_mem_3d(std::move(instruction));
    }

    void enqueue_mem_write_read_2d(
        std::size_t column,
        MemIcuWriteRead2DInstruction instruction)
    {
        check_mem_queue(column);
        mem_queues_[column].push_mem_write_read_2d(
            std::move(instruction));
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

    void enqueue_mem_stream_nd(std::size_t column,
        IcuMemStreamNdSchedule schedule, MemInstruction instruction)
    {
        check_mem_queue(column);
        mem_queues_[column].push_mem_stream_nd(
            schedule, std::move(instruction));
    }

    void enqueue_mem_slice_program(
        std::size_t column, IcuMemSliceProgram program)
    {
        check_mem_queue(column);
        mem_queues_[column].push_mem_slice_program(std::move(program));
    }

    void enqueue_mem_synchronized_write(std::size_t queue,
        MemInstruction instruction, std::size_t count,
        std::size_t transport_delay, std::int64_t address_stride = 1,
        std::size_t reservation_cycles = 1)
    {
        check_mem_queue(queue);
        check_mem_synchronized_write(queue, instruction);
        mem_queues_[queue].push_synchronized(count,
            instruction.stream_id().index(), transport_delay,
            address_stride, std::move(instruction), reservation_cycles);
    }

    void enqueue_mem_synchronized_read(std::size_t queue,
        MemInstruction instruction, std::size_t count,
        std::size_t synchronization_tag, std::size_t transport_delay = 0,
        std::int64_t address_stride = 1,
        std::size_t reservation_cycles = 1)
    {
        check_mem_queue(queue);
        if (instruction.opcode != MemOpcode::Read)
            throw std::invalid_argument(
                "MEM_READ_SYNC must carry a read template");
        mem_queues_[queue].push_synchronized(count,
            synchronization_tag, transport_delay, address_stride,
            std::move(instruction), reservation_cycles);
    }

    static EncodedMemIcuSynchronizedPacket
    encode_mem_synchronized_packet(std::size_t count,
        std::size_t synchronization_tag, std::size_t transport_delay,
        std::int64_t address_stride, const MemInstruction& instruction,
        std::size_t reservation_cycles = 1)
    {
        if (instruction.opcode != MemOpcode::Write
            && instruction.opcode != MemOpcode::Read)
            throw std::invalid_argument(
                "MEM synchronized instruction must carry a read or write template");
        const auto encoded = MemIcu::encode_synchronized_raw_packet(
            count, synchronization_tag, transport_delay, address_stride,
            instruction, reservation_cycles);
        return EncodedMemIcuSynchronizedPacket {{encoded[0], encoded[1]}};
    }

    void push_mem_synchronized_raw(std::size_t queue,
        const EncodedMemIcuSynchronizedPacket& packet)
    {
        check_mem_queue(queue);
        mem_queues_[queue].push_encoded_synchronized_packet(
            MemIcu::EncodedSynchronizedPacket {
                packet.words[0], packet.words[1]});
    }

    MemIcu& mem_transport_iq(std::size_t queue)
    {
        check_mem_queue(queue);
        if (mem_transport_queues_.empty())
            mem_transport_queues_ = std::vector<MemIcu>(kMemQueues);
        return mem_transport_queues_[queue];
    }

    const MemIcu& mem_transport_iq(std::size_t queue) const
    {
        check_mem_queue(queue);
        if (mem_transport_queues_.empty()) {
            static const MemIcu empty;
            return empty;
        }
        return mem_transport_queues_[queue];
    }

    void push_mem_transport_synchronized_raw(std::size_t queue,
        const EncodedMemIcuSynchronizedPacket& packet)
    {
        mem_transport_iq(queue).push_encoded_synchronized_packet(
            MemIcu::EncodedSynchronizedPacket{packet.words[0], packet.words[1]});
    }

    void insert_mem_synchronized_raw(std::size_t queue,
        std::size_t address,
        const EncodedMemIcuSynchronizedPacket& packet)
    {
        check_mem_queue(queue);
        mem_queues_[queue].insert_encoded_synchronized_packet(address,
            MemIcu::EncodedSynchronizedPacket {
                packet.words[0], packet.words[1]});
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

    void enqueue_mxm_load_3d(
        std::size_t mxm,
        MxmLoadIcuInstruction instruction)
    {
        check_mxm_queue(mxm);
        mxm_load_queues_[mxm].push_mxm_load_3d(
            std::move(instruction));
    }

    void enqueue_mxm_dequant(
        std::size_t mxm,
        MxmDequantInstruction instruction)
    {
        check_mxm_queue(mxm);
        mxm_dequant_queues_[mxm].push_instruction(instruction);
    }

    void enqueue_mxm_dequant_3d(
        std::size_t mxm,
        MxmDequantIcuInstruction instruction)
    {
        check_mxm_queue(mxm);
        mxm_dequant_queues_[mxm].push_mxm_dequant_3d(
            std::move(instruction));
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

    void enqueue_mxm_load_stream_nd(std::size_t mxm,
        IcuMxmStreamNdSchedule schedule,
        MxmControlInstruction instruction)
    {
        check_mxm_queue(mxm);
        if (instruction.opcode != MxmControlOpcode::IW
            && !(instruction.opcode == MxmControlOpcode::Decode
                && instruction.decode_operation
                    == MxmDecodeOperation::LoadActivation))
            throw std::invalid_argument(
                "MXM load STREAM_ND requires IW/load-activation");
        mxm_load_queues_[mxm].push_mxm_stream_nd(
            schedule, std::move(instruction));
    }

    void enqueue_mxm_dequant_macro(std::size_t mxm,
        IcuMacroSchedule schedule, MxmDequantInstruction instruction)
    {
        check_mxm_queue(mxm);
        mxm_dequant_queues_[mxm].push_macro(
            schedule, std::move(instruction));
    }

    void enqueue_mxm_dequant_stream_nd(std::size_t mxm,
        IcuMxmStreamNdSchedule schedule,
        MxmDequantInstruction instruction)
    {
        check_mxm_queue(mxm);
        mxm_dequant_queues_[mxm].push_mxm_stream_nd(
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

    void enqueue_mxm_compute_stream_nd(std::size_t mxm,
        IcuMxmStreamNdSchedule schedule,
        MxmControlInstruction instruction)
    {
        check_mxm_queue(mxm);
        if (instruction.opcode == MxmControlOpcode::IW
            || (instruction.opcode == MxmControlOpcode::Decode
                && instruction.decode_operation
                    == MxmDecodeOperation::LoadActivation))
            throw std::invalid_argument(
                "MXM compute STREAM_ND cannot carry a load instruction");
        mxm_compute_queues_[mxm].push_mxm_stream_nd(
            schedule, std::move(instruction));
    }

    void enqueue_mxm_compute_3d(
        std::size_t mxm,
        MxmComputeIcuInstruction instruction)
    {
        check_mxm_queue(mxm);
        mxm_compute_queues_[mxm].push_mxm_compute_3d(
            std::move(instruction));
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
        const auto side = hemisphere_index(hemisphere);
        sxm_transpose_queues_[side].push_sxm_run_2d(
            SxmIcuRun2DInstruction::Run2D(
                0,
                {1, 1}, {1, 1}, std::move(instruction)));
    }

    void enqueue_sxm_transpose_tile_program(Hemisphere hemisphere,
        IcuSxmTileProgramSchedule schedule, SxmInstruction instruction)
    {
        if (instruction.opcode != SxmOpcode::Transpose)
            throw std::invalid_argument(
                "ICU SXM transpose tile program requires Transpose");
        const auto side = hemisphere_index(hemisphere);
        sxm_transpose_queues_[side].push_sxm_tile_program(
            schedule, std::move(instruction));
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
        const auto side = hemisphere_index(hemisphere);
        sxm_permute_queues_[side].push_sxm_run_2d(
            SxmIcuRun2DInstruction::Run2D(
                0,
                {1, 1}, {1, 1}, std::move(instruction)));
    }

    void enqueue_sxm_permute_tile_program(Hemisphere hemisphere,
        IcuSxmTileProgramSchedule schedule, SxmInstruction instruction)
    {
        if (instruction.opcode != SxmOpcode::Permute)
            throw std::invalid_argument(
                "ICU SXM permute tile program requires Permute");
        const auto side = hemisphere_index(hemisphere);
        sxm_permute_queues_[side].push_sxm_tile_program(
            schedule, std::move(instruction));
    }

    void enqueue_sxm_transpose_nop(std::size_t cycles)
    {
        enqueue_sxm_transpose_nop(Hemisphere::East, cycles);
    }

    void enqueue_sxm_transpose_nop(Hemisphere hemisphere, std::size_t cycles)
    {
        const auto side = hemisphere_index(hemisphere);
        sxm_transpose_queues_[side].push_nop(cycles);
    }

    void enqueue_sxm_permute_nop(std::size_t cycles)
    {
        enqueue_sxm_permute_nop(Hemisphere::East, cycles);
    }

    void enqueue_sxm_permute_nop(Hemisphere hemisphere, std::size_t cycles)
    {
        const auto side = hemisphere_index(hemisphere);
        sxm_permute_queues_[side].push_nop(cycles);
    }

    void enqueue_sxm_transpose_repeat(std::size_t count, std::size_t interval = 1)
    {
        enqueue_sxm_transpose_repeat(Hemisphere::East, count, interval);
    }

    void enqueue_sxm_transpose_repeat(Hemisphere hemisphere, std::size_t count, std::size_t interval = 1)
    {
        const auto side = hemisphere_index(hemisphere);
        sxm_transpose_queues_[side].push_repeat(Repeat {count, interval, 0});
    }

    void enqueue_sxm_permute_repeat(std::size_t count, std::size_t interval = 1)
    {
        enqueue_sxm_permute_repeat(Hemisphere::East, count, interval);
    }

    void enqueue_sxm_permute_repeat(Hemisphere hemisphere, std::size_t count, std::size_t interval = 1)
    {
        const auto side = hemisphere_index(hemisphere);
        sxm_permute_queues_[side].push_repeat(Repeat {count, interval, 0});
    }

    void enqueue_c2c(
        Hemisphere endpoint_hemisphere,
        C2cInstruction instruction)
    {
        const auto endpoint = hemisphere_index(endpoint_hemisphere);
        if (instruction.opcode == C2cOpcode::Send) {
            c2c_tx_queues_[endpoint].push_encoded_c2c_endpoint_packet(
                C2cIcuPacketCodec::encode(
                    C2cTxIcuInstruction::FromLegacy(
                        endpoint_hemisphere, instruction)));
        } else {
            c2c_rx_queues_[endpoint].push_encoded_c2c_endpoint_packet(
                C2cIcuPacketCodec::encode(
                    C2cRxIcuInstruction::FromLegacy(
                        endpoint_hemisphere, instruction)));
        }
    }

    void enqueue_c2c_send(
        Hemisphere endpoint_hemisphere,
        std::size_t stream_index,
        std::size_t vector_count = 1,
        std::size_t fabric_stream_index =
            std::numeric_limits<std::size_t>::max())
    {
        enqueue_c2c(
            endpoint_hemisphere,
            C2cInstruction::Send(
                stream_index, vector_count, fabric_stream_index));
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
            .push_encoded_c2c_dma_packet(C2cIcuPacketCodec::encode(
                C2cDmaIcuInstruction::FromLegacy(
                    endpoint_hemisphere, instruction)));
    }

    // Raw local-iMEM ingress used by software paging. Packets are decoded at
    // the queue boundary; the C2C engines only ever observe engine commands.
    void enqueue_c2c_tx_raw(Hemisphere endpoint_hemisphere,
        const C2cEndpointIcuPacket& packet)
    {
        const auto instruction = C2cIcuPacketCodec::decode_tx(packet);
        require_c2c_packet_hemisphere(endpoint_hemisphere,
            instruction.endpoint_hemisphere);
        c2c_tx_queues_[hemisphere_index(endpoint_hemisphere)]
            .push_encoded_c2c_endpoint_packet(packet);
    }

    void enqueue_c2c_rx_raw(Hemisphere endpoint_hemisphere,
        const C2cEndpointIcuPacket& packet)
    {
        const auto instruction = C2cIcuPacketCodec::decode_rx(packet);
        require_c2c_packet_hemisphere(endpoint_hemisphere,
            instruction.endpoint_hemisphere);
        c2c_rx_queues_[hemisphere_index(endpoint_hemisphere)]
            .push_encoded_c2c_endpoint_packet(packet);
    }

    void enqueue_c2c_dma_raw(Hemisphere endpoint_hemisphere,
        const C2cDmaIcuPacket& packet)
    {
        const auto instruction = C2cIcuPacketCodec::decode_dma(packet);
        require_c2c_packet_hemisphere(endpoint_hemisphere,
            instruction.endpoint_hemisphere);
        c2c_dma_queues_[hemisphere_index(endpoint_hemisphere)]
            .push_encoded_c2c_dma_packet(packet);
    }

    // Short aliases keep raw queue loading ergonomic for pager clients.
    void push_c2c_tx_raw(Hemisphere endpoint_hemisphere,
        const C2cEndpointIcuPacket& packet)
    {
        enqueue_c2c_tx_raw(endpoint_hemisphere, packet);
    }

    void push_c2c_rx_raw(Hemisphere endpoint_hemisphere,
        const C2cEndpointIcuPacket& packet)
    {
        enqueue_c2c_rx_raw(endpoint_hemisphere, packet);
    }

    void push_c2c_dma_raw(Hemisphere endpoint_hemisphere,
        const C2cDmaIcuPacket& packet)
    {
        enqueue_c2c_dma_raw(endpoint_hemisphere, packet);
    }

    static constexpr std::size_t c2c_endpoint_raw_word_count = 1;
    static constexpr std::size_t c2c_dma_raw_word_count = 2;

    void append_c2c_tx_raw_word(Hemisphere endpoint_hemisphere,
        const C2cEndpointIcuPacket& word)
    {
        enqueue_c2c_tx_raw(endpoint_hemisphere, word);
    }

    void append_c2c_rx_raw_word(Hemisphere endpoint_hemisphere,
        const C2cEndpointIcuPacket& word)
    {
        enqueue_c2c_rx_raw(endpoint_hemisphere, word);
    }

    // DMA words are assembled exactly as they appear in i-MEM. A dangling
    // first word is discarded by reset and can be queried by pager software.
    void append_c2c_dma_raw_word(Hemisphere endpoint_hemisphere,
        const C2cEndpointIcuPacket& word)
    {
        const auto endpoint = hemisphere_index(endpoint_hemisphere);
        if (!pending_c2c_dma_raw_words_[endpoint].has_value()) {
            c2c_dma_queues_[endpoint].push_encoded_c2c_dma_word(word);
            pending_c2c_dma_raw_words_[endpoint] = word;
            return;
        }
        auto packet = C2cDmaIcuPacket {};
        packet.words[0] = *pending_c2c_dma_raw_words_[endpoint];
        packet.words[1] = word;
        const auto instruction = C2cIcuPacketCodec::decode_dma(packet);
        require_c2c_packet_hemisphere(endpoint_hemisphere,
            instruction.endpoint_hemisphere);
        c2c_dma_queues_[endpoint].push_encoded_c2c_dma_word(word);
        pending_c2c_dma_raw_words_[endpoint].reset();
    }

    bool c2c_dma_raw_word_pending(Hemisphere endpoint_hemisphere) const noexcept
    {
        return pending_c2c_dma_raw_words_[hemisphere_index(endpoint_hemisphere)]
            .has_value();
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
            check_mem_location(location);
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

    void notify_tagged(IcuLocation location, std::size_t event_tag)
    {
        switch (location.kind) {
        case IcuLocationKind::Mem:
            check_mem_location(location);
            mem_iq(mem_queue(static_cast<Hemisphere>(location.unit),
                location.index, location.bank)).notify(event_tag);
            return;
        case IcuLocationKind::Vxm:
            vxm_iq(location.index).notify(event_tag);
            return;
        case IcuLocationKind::MxmLoad:
            mxm_load_iq(location.unit).notify(event_tag);
            return;
        case IcuLocationKind::MxmCompute:
            mxm_compute_iq(location.unit).notify(event_tag);
            return;
        case IcuLocationKind::MxmDequant:
            mxm_dequant_iq(location.unit).notify(event_tag);
            return;
        case IcuLocationKind::Sxm:
            if (location.unit >= hw::kHemispheres || location.index > 1)
                throw std::out_of_range(
                    "ICU tagged SXM location is outside the chip");
            if (location.index == 0)
                sxm_transpose_queues_[location.unit].notify(event_tag);
            else
                sxm_permute_queues_[location.unit].notify(event_tag);
            return;
        case IcuLocationKind::C2cTx:
            c2c_tx_queues_.at(location.unit).notify(event_tag);
            return;
        case IcuLocationKind::C2cRx:
            c2c_rx_queues_.at(location.unit).notify(event_tag);
            return;
        case IcuLocationKind::C2cDma:
            c2c_dma_queues_.at(location.unit).notify(event_tag);
            return;
        }
        throw std::logic_error("unknown tagged ICU location kind");
    }

    bool blocked_on_tagged_event(
        IcuLocation location, std::size_t event_tag)
    {
        switch (location.kind) {
        case IcuLocationKind::Mem:
            check_mem_location(location);
            return mem_iq(mem_queue(static_cast<Hemisphere>(location.unit),
                location.index, location.bank)).blocked_on_event(event_tag);
        case IcuLocationKind::Vxm:
            return vxm_iq(location.index).blocked_on_event(event_tag);
        case IcuLocationKind::MxmLoad:
            return mxm_load_iq(location.unit).blocked_on_event(event_tag);
        case IcuLocationKind::MxmCompute:
            return mxm_compute_iq(location.unit).blocked_on_event(event_tag);
        case IcuLocationKind::MxmDequant:
            return mxm_dequant_iq(location.unit).blocked_on_event(event_tag);
        case IcuLocationKind::Sxm:
            if (location.unit >= hw::kHemispheres || location.index > 1)
                throw std::out_of_range(
                    "ICU tagged SXM location is outside the chip");
            return location.index == 0
                ? sxm_transpose_queues_[location.unit]
                      .blocked_on_event(event_tag)
                : sxm_permute_queues_[location.unit]
                      .blocked_on_event(event_tag);
        case IcuLocationKind::C2cTx:
            return c2c_tx_queues_.at(location.unit)
                .blocked_on_event(event_tag);
        case IcuLocationKind::C2cRx:
            return c2c_rx_queues_.at(location.unit)
                .blocked_on_event(event_tag);
        case IcuLocationKind::C2cDma:
            return c2c_dma_queues_.at(location.unit)
                .blocked_on_event(event_tag);
        }
        throw std::logic_error("unknown tagged ICU location kind");
    }

    void set_program_issue_enabled(bool enabled) noexcept
    {
        program_issue_enabled_ = enabled;
    }

    bool program_issue_enabled() const noexcept
    {
        return program_issue_enabled_;
    }

    void configure_common_barriers(std::uint64_t generation,
        std::vector<IcuCommonBarrierEntry> entries,
        const std::vector<IcuCommonEventBinding>& bindings)
    {
        if (cycle_ != 0)
            throw StaticScheduleError("common barrier RAM must be prepared before execution");
        common_barriers_.configure(generation, std::move(entries), bindings);
    }

    void notify_common_mem_commit(std::uint64_t generation,
        std::uint32_t completion_instance)
    {
        common_barriers_.notify_mem_commit(generation, completion_instance);
    }

    std::uint64_t logical_cycle() const noexcept { return common_barriers_.logical_cycle(); }
    std::uint64_t physical_cycle() const noexcept { return common_barriers_.physical_cycle(); }
    std::uint64_t common_barrier_wait_cycles() const noexcept { return common_barriers_.wait_cycles(); }
    bool compute_enabled() const noexcept { return compute_enabled_; }
    bool common_barrier_holding() const noexcept { return common_barriers_.holding(); }
    bool common_barriers_done() const noexcept { return common_barriers_.done(); }
    bool common_barrier_due() const noexcept { return common_barriers_.barrier_due(); }

    void notify_mem_synchronized(IcuLocation location,
        std::size_t synchronization_tag)
    {
        if (location.kind != IcuLocationKind::Mem)
            throw std::invalid_argument(
                "C2C RX can only notify a target MEM ICU");
        check_mem_location(location);
        const auto queue = mem_queue(
            static_cast<Hemisphere>(location.unit),
            location.index, location.bank);
        if (!mem_transport_queues_.empty()
            && mem_transport_queues_[queue].accepts_synchronized_notification(synchronization_tag))
            mem_transport_queues_[queue].notify(synchronization_tag);
        else
            mem_iq(queue).notify(synchronization_tag);
    }

    bool mem_accepts_synchronized_notification(IcuLocation location,
        std::size_t synchronization_tag) const
    {
        if (location.kind != IcuLocationKind::Mem)
            throw std::invalid_argument(
                "C2C RX readiness requires a target MEM ICU");
        check_mem_location(location);
        const auto queue = mem_queue(
            static_cast<Hemisphere>(location.unit),
            location.index, location.bank);
        return (!mem_transport_queues_.empty()
                && mem_transport_queues_[queue].accepts_synchronized_notification(synchronization_tag))
            || mem_iq(queue).accepts_synchronized_notification(synchronization_tag);
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

    // Fan out one tagged hardware event to every local ICU.  Page-ready
    // synchronization uses this path: each ordinary queue carries the same
    // WAIT_EVENT immediately before its first command at or after the page's
    // consumer boundary, while C2C transport queues may keep running.
    void broadcast_tagged_notification(std::size_t event_tag)
    {
        for (auto& queue : vxm_queues_) queue.notify(event_tag);
        for (auto& queue : mem_queues_) queue.notify(event_tag);
        for (auto& queue : mxm_load_queues_) queue.notify(event_tag);
        for (auto& queue : mxm_dequant_queues_) queue.notify(event_tag);
        for (auto& queue : mxm_compute_queues_) queue.notify(event_tag);
        for (auto& queue : sxm_transpose_queues_) queue.notify(event_tag);
        for (auto& queue : sxm_permute_queues_) queue.notify(event_tag);
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
        if (common_barriers_.barrier_due()
            && !vxm.idle())
            throw StaticScheduleError("unsafe common barrier: VXM has in-flight computation");
        compute_enabled_ = common_barriers_.begin_cycle(program_issue_enabled_);
        bool any = false;
        if (compute_enabled_) {
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
        }
        else for (auto& queue : vxm_queues_) queue.hold_compute_cycle();
        log_dispatch_idle(os, any);
        common_barriers_.end_cycle();
        ++cycle_;
    }
    void dispatch(
        std::array<MemArrayModel, hw::kHemispheres>& mems,
        VxmSlice& vxm,
        std::array<SxmSlice, hw::kHemispheres>& sxms,
        std::array<Mxm, kMxmQueues>& mxms,
        std::ostream* os = nullptr,
        std::array<C2cEndpoint*, hw::kHemispheres> c2cs = {},
        std::array<C2cDmaEngine*, hw::kHemispheres> c2c_dmas = {})
    {
        log_cycle_header(os);

        compute_enabled_ = common_barriers_.begin_cycle(program_issue_enabled_);
        if (!compute_enabled_) {
            for (auto& queue : vxm_queues_) queue.hold_compute_cycle();
            for (auto& queue : mxm_load_queues_) queue.hold_compute_cycle();
            for (auto& queue : mxm_dequant_queues_) queue.hold_compute_cycle();
            for (auto& queue : mxm_compute_queues_) queue.hold_compute_cycle();
            for (auto& queue : sxm_transpose_queues_) queue.hold_compute_cycle();
            for (auto& queue : sxm_permute_queues_) queue.hold_compute_cycle();
        }

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
                if (c2c_tx->consumer.notify_mem) {
                    auto& target = mem_iq(mem_queue(
                        c2c_tx->consumer.hemisphere,
                        c2c_tx->consumer.mem_slice,
                        c2c_tx->consumer.mem_bank));
                    target.launch();
                    if (!mem_transport_queues_.empty())
                        mem_transport_queues_[mem_queue(
                            c2c_tx->consumer.hemisphere,
                            c2c_tx->consumer.mem_slice,
                            c2c_tx->consumer.mem_bank)].launch();
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
        if (compute_enabled_) {
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
        }

        for (std::size_t queue = 0; queue < kMemQueues; ++queue) {
            auto instruction = std::optional<MemInstruction>{};
            if (compute_enabled_)
                instruction = mem_queues_[queue].dispatch_next();
            else if (common_barriers_.configured())
                mem_queues_[queue].hold_compute_cycle();
            else
                instruction = mem_queues_[queue].tick_transport_only();
            const auto transport = mem_transport_queues_.empty()
                ? std::optional<MemInstruction>{}
                : mem_transport_queues_[queue].dispatch_next();
            if (instruction && transport)
                throw StaticScheduleError("MEM transport and compute contend for the shared bank issue port");
            if (transport) instruction = transport;
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
                    << describe_mem(*instruction)
                    << (mem_queues_[queue].last_trace().action
                                == IcuQueueAction::SynchronizedIssue
                            ? " source=synchronized" : "")
                    << '\n';
            }
        }

        if (compute_enabled_)
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

        if (compute_enabled_)
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

        if (compute_enabled_)
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

        if (compute_enabled_)
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
        common_barriers_.end_cycle();
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

    const MemIcu& mem_iq(std::size_t queue) const
    {
        check_mem_queue(queue);
        return mem_queues_[queue];
    }

    MxmLoadIcu& mxm_load_iq(std::size_t mxm)
    {
        check_mxm_queue(mxm);
        return mxm_load_queues_[mxm];
    }

    MxmDequantIcu& mxm_dequant_iq(std::size_t mxm)
    {
        check_mxm_queue(mxm);
        return mxm_dequant_queues_[mxm];
    }

    MxmComputeIcu& mxm_compute_iq(std::size_t mxm)
    {
        check_mxm_queue(mxm);
        return mxm_compute_queues_[mxm];
    }

    SxmTransposeIcu& sxm_transpose_iq(Hemisphere hemisphere)
    {
        return sxm_transpose_queues_[hemisphere_index(hemisphere)];
    }

    SxmPermuteIcu& sxm_permute_iq(Hemisphere hemisphere)
    {
        return sxm_permute_queues_[hemisphere_index(hemisphere)];
    }

    C2cTxIcu& c2c_tx_iq(Hemisphere hemisphere) noexcept
    {
        return c2c_tx_queues_[hemisphere_index(hemisphere)];
    }
    C2cRxIcu& c2c_rx_iq(Hemisphere hemisphere) noexcept
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

    std::vector<IcuMacroQueueFrontendStatistics>
    macro_queue_frontend_statistics() const
    {
        std::vector<IcuMacroQueueFrontendStatistics> result;
        result.reserve(kMemQueues + 3 * kMxmQueues);
        const auto append = [&](IcuMacroQueueKind kind,
                                std::size_t index,
                                const auto& queue) {
            if (!queue.raw_macro_mode()) return;
            result.push_back(IcuMacroQueueFrontendStatistics{
                kind,
                index,
                queue.imem_occupancy(),
                queue.fetched_count(),
                queue.issued_count(),
                queue.peak_active_macros(),
                queue.macro_context_depth,
                queue.macro_context_bits,
                queue.instruction_bits,
                queue.macro_imem_read_latency,
                queue.macro_imem_request_initiation_interval,
                queue.macro_reservoir_words,
                queue.macro_decode_window_bits,
                queue.macro_ddb_depth,
                queue.macro_ddb_run_capacity,
                queue.macro_ddb_layout().entry_bits,
                queue.macro_admission_lookahead,
                queue.macro_context_expand_width,
                queue.macro_decoder_statistics()});
        };
        for (std::size_t index = 0; index < mem_queues_.size(); ++index)
            append(IcuMacroQueueKind::Mem, index, mem_queues_[index]);
        for (std::size_t index = 0; index < mxm_load_queues_.size(); ++index)
            append(IcuMacroQueueKind::MxmLoad, index,
                mxm_load_queues_[index]);
        for (std::size_t index = 0; index < mxm_dequant_queues_.size(); ++index)
            append(IcuMacroQueueKind::MxmDequant, index,
                mxm_dequant_queues_[index]);
        for (std::size_t index = 0; index < mxm_compute_queues_.size(); ++index)
            append(IcuMacroQueueKind::MxmCompute, index,
                mxm_compute_queues_[index]);
        return result;
    }

    IcuFrontendStatistics frontend_statistics() const noexcept
    {
        IcuFrontendStatistics statistics;
        const auto collect_queue = [&](const auto& queue) {
            statistics.imem_entries += queue.imem_occupancy();
            statistics.fetched_entries += queue.fetched_count();
            statistics.issued_instructions += queue.issued_count();
            const auto& decoder = queue.macro_decoder_statistics();
            statistics.decoded_macro_contexts += decoder.decoded_contexts;
            statistics.macro_decoder_bit_wait_cycles +=
                decoder.bit_wait_cycles;
            statistics.macro_decoder_context_stall_cycles +=
                decoder.context_stall_cycles;
            statistics.peak_macro_reservoir_bits = std::max(
                statistics.peak_macro_reservoir_bits,
                decoder.peak_reservoir_bits);
            statistics.macro_frontend_cycles += decoder.cycles;
            statistics.macro_fetched_words += decoder.fetched_words;
            statistics.macro_payload_bits_consumed +=
                decoder.payload_bits_consumed;
            statistics.macro_reservoir_empty_cycles +=
                decoder.reservoir_empty_cycles;
            statistics.macro_reservoir_full_cycles +=
                decoder.reservoir_full_cycles;
            statistics.macro_reservoir_occupancy_bit_samples +=
                decoder.reservoir_occupancy_bit_samples;
            statistics.macro_descriptor_count += decoder.descriptor_count;
            statistics.macro_mem_descriptor_count +=
                decoder.mem_descriptor_count;
            statistics.macro_mxm_descriptor_count +=
                decoder.mxm_descriptor_count;
            statistics.macro_descriptor_decode_cycles +=
                decoder.descriptor_decode_cycles;
            statistics.macro_max_descriptor_decode_cycles = std::max(
                statistics.macro_max_descriptor_decode_cycles,
                decoder.max_descriptor_decode_cycles);
            statistics.macro_decoder_active_cycles +=
                decoder.decoder_active_cycles;
            statistics.macro_decoder_starvation_cycles +=
                decoder.decoder_starvation_cycles;
            statistics.macro_decoder_ddb_stall_cycles +=
                decoder.decoder_ddb_stall_cycles;
            statistics.macro_ddb_entries += decoder.ddb_entries_committed;
            statistics.peak_macro_ddb_occupancy = std::max(
                statistics.peak_macro_ddb_occupancy,
                decoder.peak_ddb_occupancy);
            statistics.macro_ddb_full_cycles += decoder.ddb_full_cycles;
            statistics.macro_ddb_residency_cycles +=
                decoder.ddb_residency_cycles;
            statistics.macro_ddb_occupancy_samples +=
                decoder.ddb_occupancy_samples;
            statistics.macro_max_ddb_residency_cycles = std::max(
                statistics.macro_max_ddb_residency_cycles,
                decoder.max_ddb_residency_cycles);
            statistics.macro_contexts_generated +=
                decoder.contexts_generated;
            statistics.macro_timing_window_block_cycles +=
                decoder.timing_window_block_cycles;
            statistics.macro_active_ram_full_cycles +=
                decoder.active_ram_full_cycles;
            statistics.macro_capacity_feasible_wait_cycles +=
                decoder.capacity_feasible_wait_cycles;
            statistics.macro_admission_count += decoder.admission_count;
            statistics.macro_admission_stall_cycles +=
                decoder.admission_stall_cycles;
            statistics.macro_admission_to_start_cycles +=
                decoder.admission_to_start_cycles;
            statistics.macro_max_admission_to_start_cycles = std::max(
                statistics.macro_max_admission_to_start_cycles,
                decoder.max_admission_to_start_cycles);
            statistics.macro_active_occupancy_samples +=
                decoder.active_occupancy_samples;
            statistics.macro_context_active_lifetime_cycles +=
                decoder.context_active_lifetime_cycles;
            statistics.macro_max_context_active_lifetime_cycles = std::max(
                statistics.macro_max_context_active_lifetime_cycles,
                decoder.max_context_active_lifetime_cycles);
            const auto peak = queue.peak_active_macros();
            statistics.macro_queues +=
                queue.raw_macro_mode() || peak != 0 ? 1 : 0;
            if (peak > statistics.peak_macro_contexts_per_queue)
                statistics.peak_macro_contexts_per_queue = peak;
        };
        const auto collect_queues = [&](const auto& queues) {
            for (const auto& queue : queues) collect_queue(queue);
        };

        collect_queues(vxm_queues_);
        collect_queues(mem_queues_);
        collect_queues(mxm_load_queues_);
        collect_queues(mxm_dequant_queues_);
        collect_queues(mxm_compute_queues_);
        collect_queues(sxm_transpose_queues_);
        collect_queues(sxm_permute_queues_);
        collect_queues(c2c_tx_queues_);
        collect_queues(c2c_dma_queues_);
        collect_queues(c2c_rx_queues_);
        return statistics;
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
        collect(mem_transport_queues_);
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

    static void check_mem_location(const IcuLocation& location)
    {
        if (location.unit >= hw::kHemispheres
            || location.index >= hw::kMemSliceColumns
            || location.bank >= hw::kMemBanksPerSlice)
            throw std::out_of_range(
                "ICU MEM location is outside the configured chip topology");
    }

    static void check_mem_synchronized_write(
        std::size_t, const MemInstruction& instruction)
    {
        if (instruction.opcode != MemOpcode::Write)
            throw std::invalid_argument(
                "MEM synchronized instruction must carry a write template");
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
    static void require_c2c_packet_hemisphere(Hemisphere queue_hemisphere,
        Hemisphere packet_hemisphere)
    {
        if (queue_hemisphere != packet_hemisphere)
            throw std::invalid_argument(
                "C2C raw packet endpoint does not match its ICU queue");
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
    // Legacy VXM/SXM builders still describe positions in a compiler-side
    // program timeline. These cursors exist only while translating that API:
    // gaps become local-iMEM NOPs and every hardware RUN_2D packet starts at
    // relative cycle zero. They are never consulted by the runtime ICU.
    std::array<MemIcu, kMemQueues> mem_queues_{};
    // Separate bounded native transport control RAM. It shares the physical
    // bank issue port, never the raw Macro decoder or its logical clock.
    std::vector<MemIcu> mem_transport_queues_{};
    std::array<MxmLoadIcu, kMxmQueues> mxm_load_queues_{};
    std::array<MxmDequantIcu, kMxmQueues>
        mxm_dequant_queues_{};
    std::array<MxmComputeIcu, kMxmQueues> mxm_compute_queues_{};
    std::array<SxmTransposeIcu, hw::kHemispheres> sxm_transpose_queues_{};
    std::array<SxmPermuteIcu, hw::kHemispheres> sxm_permute_queues_{};
    std::array<C2cTxIcu, hw::kHemispheres> c2c_tx_queues_{};
    std::array<C2cDmaIcu, hw::kHemispheres> c2c_dma_queues_{};
    std::array<C2cRxIcu, hw::kHemispheres> c2c_rx_queues_{};
    std::array<std::optional<C2cEndpointIcuPacket>, hw::kHemispheres>
        pending_c2c_dma_raw_words_{};
    std::size_t barrier_latency_cycles_{hw::kIcuBarrierLatencyCycles};
    std::deque<std::size_t> barrier_events_{};
    bool program_issue_enabled_{true};
    bool compute_enabled_{true};
    IcuCommonBarrierController common_barriers_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
