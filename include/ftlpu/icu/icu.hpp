#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/stream_port.hpp"
#include "ftlpu/icu/fetch.hpp"
#include "ftlpu/icu/instruction.hpp"
#include "ftlpu/mem/tile_array.hpp"
#include "ftlpu/mxm/mxm.hpp"
#include "ftlpu/vxm/slice.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace ftlpu {

template <typename FuncInstruction>
using IqEntry = std::variant<IcuControlInstruction, FuncInstruction>;

namespace detail {

template <typename FuncInstruction>
FuncInstruction apply_icu_repeat_stride(
    FuncInstruction instruction,
    std::int64_t,
    std::size_t)
{
    return instruction;
}

inline MemInstruction apply_icu_repeat_stride(
    MemInstruction instruction,
    std::int64_t address_stride,
    std::size_t repeat_index)
{
    const auto delta = address_stride * static_cast<std::int64_t>(repeat_index);
    const auto address = static_cast<std::int64_t>(instruction.address.encoded());
    if (delta < 0 && address < -delta) {
        throw std::out_of_range("ICU MEM Repeat address stride underflow");
    }
    instruction.address = MemLocalWordAddress13(
        static_cast<std::size_t>(address + delta));
    return instruction;
}

template <typename FuncInstruction>
IqEntry<FuncInstruction> decode_icu_fetch_packet(
    const isa::EncodedInstructionPacket& packet)
{
    if (isa::packet_kind(packet) == isa::InstructionPacketKind::IcuControl) {
        return IqEntry<FuncInstruction> {
            std::in_place_type<IcuControlInstruction>,
            isa::decode_icu_packet(packet)};
    }
    if constexpr (std::is_same_v<FuncInstruction, MemInstruction>) {
        return IqEntry<FuncInstruction> {
            std::in_place_type<FuncInstruction>, isa::decode_mem_packet(packet)};
    } else if constexpr (std::is_same_v<FuncInstruction, MxmControlInstruction>) {
        return IqEntry<FuncInstruction> {
            std::in_place_type<FuncInstruction>, isa::decode_mxm_packet(packet)};
    } else if constexpr (std::is_same_v<FuncInstruction, VxmLaneAluInstruction>) {
        return IqEntry<FuncInstruction> {
            std::in_place_type<FuncInstruction>, isa::decode_vxm_packet(packet)};
    } else {
        throw std::logic_error("ICU Ifetch decoder does not support this functional instruction type");
    }
}

} // namespace detail

// One program-ordered instruction queue for one functional slice.  ICU
// controls and functional instructions occupy the same FIFO, so a control at
// the head always orders all following functional dispatches.
template <typename FuncInstruction>
class SliceIcu {
public:
    using Entry = IqEntry<FuncInstruction>;

    void reset()
    {
        iq_.clear();
        pending_fetch_entries_.clear();
        fetch_state_.reset();
        clear_runtime_state();
        cycle_ = 0;
    }

    void load(std::vector<Entry> program)
    {
        if (program.size() * hw::kEncodedInstructionPacketBytes
            > hw::kIcuIqCapacityBytes) {
            throw StaticScheduleError("ICU IQ program exceeds modeled capacity");
        }
        iq_.assign(
            std::make_move_iterator(program.begin()),
            std::make_move_iterator(program.end()));
        clear_runtime_state();
        cycle_ = 0;
    }

    void enqueue(Entry entry)
    {
        require_capacity_for(hw::kEncodedInstructionPacketBytes);
        iq_.push_back(std::move(entry));
    }

    void enqueue(FuncInstruction instruction)
    {
        enqueue(Entry {std::in_place_type<FuncInstruction>, std::move(instruction)});
    }

    void enqueue_control(IcuControlInstruction instruction)
    {
        enqueue(Entry {std::in_place_type<IcuControlInstruction>, instruction});
    }

    // Dispatch consumes only the current IQ state.  Instructions decoded by
    // evaluate_fetch remain invisible until commit_fetch is called.
    std::optional<FuncInstruction> dispatch()
    {
        ++cycle_;
        notify_emitted_ = false;

        if (nop_remaining_ > 0) {
            --nop_remaining_;
            return std::nullopt;
        }
        if (repeat_remaining_ > 0) {
            return tick_repeat();
        }
        if (iq_.empty()) {
            return std::nullopt;
        }

        if (auto* instruction = std::get_if<FuncInstruction>(&iq_.front())) {
            auto result = std::move(*instruction);
            iq_.pop_front();
            last_dispatched_ = result;
            return result;
        }

        const auto control = std::get<IcuControlInstruction>(iq_.front());
        return execute_control(control);
    }

    std::optional<FuncInstruction> tick()
    {
        return dispatch();
    }

    void evaluate_fetch(StreamRegisterFabric& fabric, std::size_t column)
    {
        if (!fetch_state_.has_value() || !pending_fetch_entries_.empty()) {
            return;
        }
        if (!fabric.cycle_open()) {
            throw std::logic_error("ICU evaluate_fetch requires an open SR cycle");
        }
        if (column >= fabric.column_count()) {
            throw std::out_of_range("ICU fetch port maps outside the stream fabric");
        }

        const auto stream = fetch_state_->source_stream;
        StreamInputPort input(fabric, column, stream.direction(), "ICU Ifetch");
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            if (!input.segment_valid(tile, stream.index())) {
                continue;
            }
            fetch_state_->buffer.accept_segment(
                tile,
                input.consume_segment(tile, stream.index()));
        }
        if (!fetch_state_->buffer.complete()) {
            return;
        }

        stage_fetched_packets(fetch_state_->buffer.packets());
    }

    // Starts a non-stream packet transfer, currently used only by a MEM ICU
    // reading the SRAM physically local to its functional slice.
    void begin_local_fetch()
    {
        begin_fetch_reservation();
    }

    void stage_fetched_packets(const IcuFetchBuffer::PacketArray& packets)
    {
        if (!fetch_reservation_active_) {
            throw std::logic_error("ICU packet fetch has no active IQ reservation");
        }
        if (!pending_fetch_entries_.empty()) {
            throw std::logic_error("ICU packet fetch already has a pending completion");
        }
        pending_fetch_entries_.reserve(packets.size());
        for (const auto& packet : packets) {
            pending_fetch_entries_.push_back(
                detail::decode_icu_fetch_packet<FuncInstruction>(packet));
        }
    }

    void commit_fetch()
    {
        if (pending_fetch_entries_.empty()) {
            return;
        }
        if (!fetch_reservation_active_) {
            throw std::logic_error("ICU fetch completion has no active IQ reservation");
        }
        if (pending_fetch_entries_.size() != hw::kIcuFetchPackets) {
            throw std::logic_error("ICU fetch completion must contain exactly 40 packets");
        }
        release_fetch_reservation();
        for (auto& entry : pending_fetch_entries_) {
            // Reservation guaranteed this space at Fetch issue time.
            iq_.push_back(std::move(entry));
        }
        pending_fetch_entries_.clear();
        fetch_state_.reset();
        fetch_reservation_active_ = false;
    }

    bool fetch_active() const noexcept { return fetch_reservation_active_; }
    bool fetch_complete_pending_commit() const noexcept
    {
        return !pending_fetch_entries_.empty();
    }
    const IcuFetchState* fetch_state() const noexcept
    {
        return fetch_state_.has_value() ? &*fetch_state_ : nullptr;
    }

    // Supplies one incoming synchronization notification. Tokens are retained
    // until a Sync reaches the queue head.
    void notify()
    {
        ++notification_tokens_;
    }

    // Notify is an outgoing ICU control event. The system may route it to a
    // peer SliceIcu by calling notify() on that peer.
    bool take_notify()
    {
        const auto emitted = notify_emitted_;
        notify_emitted_ = false;
        return emitted;
    }

    bool blocked_on_sync() const
    {
        if (nop_remaining_ != 0 || repeat_remaining_ != 0 || iq_.empty()) {
            return false;
        }
        const auto* control = std::get_if<IcuControlInstruction>(&iq_.front());
        return control != nullptr
            && control->opcode == IcuControlOpcode::Sync
            && notification_tokens_ == 0;
    }

    bool synced() const { return blocked_on_sync(); }

    bool done() const
    {
        return iq_.empty() && nop_remaining_ == 0 && repeat_remaining_ == 0
            && !fetch_reservation_active_ && pending_fetch_entries_.empty();
    }

    bool running() const { return !done(); }
    std::size_t cycle() const noexcept { return cycle_; }
    std::size_t fetch_count() const noexcept { return fetch_count_; }
    std::size_t iq_occupancy() const noexcept { return iq_.size(); }
    std::size_t occupancy_bytes() const noexcept
    {
        return iq_.size() * hw::kEncodedInstructionPacketBytes;
    }
    std::size_t reserved_bytes() const noexcept { return reserved_bytes_; }
    std::size_t free_bytes() const noexcept
    {
        return hw::kIcuIqCapacityBytes - occupancy_bytes() - reserved_bytes_;
    }
    std::size_t pending_issue_cycles() const noexcept
    {
        if (repeat_remaining_ == 0) {
            return nop_remaining_;
        }
        return nop_remaining_ + repeat_cooldown_ + 1
            + (repeat_remaining_ - 1) * repeat_interval_;
    }

    void reserve_fetch()
    {
        reserve_bytes(hw::kIcuFetchBufferBytes);
    }

    void release_fetch_reservation()
    {
        if (reserved_bytes_ < hw::kIcuFetchBufferBytes) {
            throw std::logic_error("ICU has no 640-byte Ifetch reservation to release");
        }
        reserved_bytes_ -= hw::kIcuFetchBufferBytes;
    }
    std::size_t queued_count() const noexcept
    {
        return iq_.size() + repeat_remaining_ + nop_remaining_;
    }

private:
    void require_capacity_for(std::size_t bytes) const
    {
        if (bytes > free_bytes()) {
            throw StaticScheduleError("ICU IQ capacity exceeded by static schedule");
        }
    }

    void reserve_bytes(std::size_t bytes)
    {
        require_capacity_for(bytes);
        reserved_bytes_ += bytes;
    }

    void clear_runtime_state()
    {
        last_dispatched_.reset();
        repeat_instruction_.reset();
        nop_remaining_ = 0;
        repeat_remaining_ = 0;
        repeat_interval_ = 1;
        repeat_cooldown_ = 0;
        repeat_address_stride_ = 0;
        repeat_index_ = 0;
        notification_tokens_ = 0;
        notify_emitted_ = false;
        fetch_count_ = 0;
        reserved_bytes_ = 0;
        pending_fetch_entries_.clear();
        fetch_state_.reset();
        fetch_reservation_active_ = false;
    }

    void begin_fetch_reservation()
    {
        if (fetch_reservation_active_) {
            throw StaticScheduleError(
                "one SliceIcu cannot have two active instruction fetches");
        }
        reserve_fetch();
        fetch_reservation_active_ = true;
        ++fetch_count_;
    }

    std::optional<FuncInstruction> execute_control(
        const IcuControlInstruction& control)
    {
        switch (control.opcode) {
        case IcuControlOpcode::Fetch:
            begin_fetch_reservation();
            fetch_state_ = IcuFetchState {control.source_stream, {}, true};
            fetch_state_->buffer.reset();
            iq_.pop_front();
            return std::nullopt;
        case IcuControlOpcode::Nop:
            iq_.pop_front();
            nop_remaining_ = control.count;
            if (nop_remaining_ > 0) {
                --nop_remaining_;
            }
            return std::nullopt;
        case IcuControlOpcode::Repeat:
            return begin_repeat(control);
        case IcuControlOpcode::Sync:
            if (notification_tokens_ == 0) {
                return std::nullopt;
            }
            --notification_tokens_;
            iq_.pop_front();
            return std::nullopt;
        case IcuControlOpcode::Notify:
            iq_.pop_front();
            notify_emitted_ = true;
            return std::nullopt;
        }
        throw std::logic_error("unknown ICU control opcode");
    }

    std::optional<FuncInstruction> begin_repeat(
        const IcuControlInstruction& control)
    {
        if (control.interval == 0) {
            throw std::invalid_argument("ICU Repeat interval must be at least one cycle");
        }
        if (!last_dispatched_.has_value()) {
            throw std::logic_error("ICU Repeat needs a prior functional instruction in the same IQ");
        }

        iq_.pop_front();
        if (control.count == 0) {
            return std::nullopt;
        }
        repeat_remaining_ = control.count;
        repeat_instruction_ = *last_dispatched_;
        repeat_interval_ = control.interval;
        repeat_cooldown_ = control.interval - 1;
        repeat_address_stride_ = control.address_stride;
        repeat_index_ = 1;
        return tick_repeat();
    }

    std::optional<FuncInstruction> tick_repeat()
    {
        if (repeat_cooldown_ > 0) {
            --repeat_cooldown_;
            return std::nullopt;
        }

        auto result = detail::apply_icu_repeat_stride(
            *repeat_instruction_, repeat_address_stride_, repeat_index_);
        --repeat_remaining_;
        ++repeat_index_;
        if (repeat_remaining_ > 0) {
            repeat_cooldown_ = repeat_interval_ - 1;
        }
        last_dispatched_ = result;
        return result;
    }

    std::deque<Entry> iq_{};
    std::optional<FuncInstruction> last_dispatched_{};
    std::optional<FuncInstruction> repeat_instruction_{};
    std::size_t nop_remaining_{0};
    std::size_t repeat_remaining_{0};
    std::size_t repeat_interval_{1};
    std::size_t repeat_cooldown_{0};
    std::int64_t repeat_address_stride_{0};
    std::size_t repeat_index_{0};
    std::size_t notification_tokens_{0};
    bool notify_emitted_{false};
    std::size_t fetch_count_{0};
    std::size_t reserved_bytes_{0};
    std::size_t cycle_{0};
    std::optional<IcuFetchState> fetch_state_{};
    bool fetch_reservation_active_{false};
    std::vector<Entry> pending_fetch_entries_{};
};

// Whole-system ICU wiring. Each array element is one SliceIcu and therefore
// owns exactly one program-ordered IQ for its target functional slice.
class InstructionControlUnit {
public:
    using Repeat = IcuRepeat;
    enum class MxmIcuPort : std::size_t {
        Load = 0,
        Compute = 1,
    };

    static constexpr std::size_t kVxmQueues = VxmSlice::kAluQueues;
    static constexpr std::size_t kMemQueues = hw::kSliceColumns;
    static constexpr std::size_t kMxmUnitCount = 2;
    static constexpr std::size_t kMxmIcusPerUnit = 2;

    explicit InstructionControlUnit(
        std::size_t barrier_latency_cycles = hw::kIcuBarrierLatencyCycles)
        : barrier_latency_cycles_(barrier_latency_cycles)
    {
    }

    void reset()
    {
        reset_all(vxm_iqs_);
        reset_all(mem_iqs_);
        for (auto& fetch : mem_local_fetches_) fetch.reset();
        for (auto& ports : mxm_iqs_) reset_all(ports);
        barrier_events_.clear();
        cycle_ = 0;
    }

    // Location is topology/control-plane state and is deliberately not
    // encoded in Fetch itself.
    void enqueue_control(
        IcuLocation location,
        IcuControlInstruction instruction)
    {
        switch (location.kind) {
        case IcuLocationKind::Mem:
            enqueue_mem_control(location.index, instruction);
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
        }
        throw std::logic_error("unknown ICU location kind");
    }

    void enqueue_nop(std::size_t cycles)
    {
        if (cycles == 0) return;
        for (auto& iq : vxm_iqs_) iq.enqueue_control(IcuControlInstruction::Nop(cycles));
        for (auto& iq : mem_iqs_) iq.enqueue_control(IcuControlInstruction::Nop(cycles));
        for (auto& ports : mxm_iqs_) {
            for (auto& iq : ports) iq.enqueue_control(IcuControlInstruction::Nop(cycles));
        }
    }

    void enqueue_vxm(std::size_t alu, VxmLaneAluInstruction instruction)
    {
        vxm_iq(alu).enqueue(std::move(instruction));
    }

    void enqueue_vxm_control(std::size_t alu, IcuControlInstruction instruction)
    {
        vxm_iq(alu).enqueue_control(instruction);
    }

    void enqueue_vxm_nop(std::size_t alu, std::size_t cycles)
    {
        if (cycles == 0) return;
        enqueue_vxm_control(alu, IcuControlInstruction::Nop(cycles));
    }

    void enqueue_vxm_repeat(std::size_t alu, std::size_t count, std::size_t interval = 1)
    {
        if (count == 0) return;
        enqueue_vxm_control(alu, IcuControlInstruction::Repeat(count, interval));
    }

    void enqueue_mem(std::size_t column, MemInstruction instruction)
    {
        mem_iq(column).enqueue(std::move(instruction));
    }

    void enqueue_mem_control(std::size_t column, IcuControlInstruction instruction)
    {
        mem_iq(column).enqueue_control(instruction);
    }

    void enqueue_mem_nop(std::size_t column, std::size_t cycles)
    {
        if (cycles == 0) return;
        enqueue_mem_control(column, IcuControlInstruction::Nop(cycles));
    }

    void enqueue_mem_repeat(
        std::size_t column,
        std::size_t count,
        std::size_t interval = 1,
        std::int64_t address_stride = 0)
    {
        if (count == 0) return;
        enqueue_mem_control(
            column,
            IcuControlInstruction::Repeat(count, interval, address_stride));
    }

    void enqueue_mxm(std::size_t mxm, MxmControlInstruction instruction)
    {
        const auto port = instruction.opcode == MxmControlOpcode::Compute
            ? MxmIcuPort::Compute
            : MxmIcuPort::Load;
        mxm_iq(mxm, port).enqueue(std::move(instruction));
    }

    void enqueue_mxm_control(
        std::size_t mxm,
        MxmIcuPort port,
        IcuControlInstruction instruction)
    {
        mxm_iq(mxm, port).enqueue_control(instruction);
    }

    void enqueue_mxm_nop(std::size_t mxm, std::size_t cycles)
    {
        if (cycles == 0) return;
        enqueue_mxm_control(mxm, MxmIcuPort::Load, IcuControlInstruction::Nop(cycles));
        enqueue_mxm_control(mxm, MxmIcuPort::Compute, IcuControlInstruction::Nop(cycles));
    }

    void enqueue_mxm_load_nop(std::size_t mxm, std::size_t cycles)
    {
        if (cycles == 0) return;
        enqueue_mxm_control(mxm, MxmIcuPort::Load, IcuControlInstruction::Nop(cycles));
    }

    void enqueue_mxm_compute_nop(std::size_t mxm, std::size_t cycles)
    {
        if (cycles == 0) return;
        enqueue_mxm_control(mxm, MxmIcuPort::Compute, IcuControlInstruction::Nop(cycles));
    }

    void enqueue_mxm_repeat(
        std::size_t mxm,
        MxmIcuPort port,
        std::size_t count,
        std::size_t interval = 1)
    {
        if (count == 0) return;
        enqueue_mxm_control(mxm, port, IcuControlInstruction::Repeat(count, interval));
    }

    void enqueue_mxm_load_repeat(std::size_t mxm, std::size_t count, std::size_t interval = 1)
    {
        enqueue_mxm_repeat(mxm, MxmIcuPort::Load, count, interval);
    }

    void enqueue_mxm_compute_repeat(std::size_t mxm, std::size_t count, std::size_t interval = 1)
    {
        enqueue_mxm_repeat(mxm, MxmIcuPort::Compute, count, interval);
    }

    void notify_vxm(std::size_t alu) { vxm_iq(alu).notify(); }
    void notify_mem(std::size_t column) { mem_iq(column).notify(); }
    void notify_mxm(std::size_t mxm, MxmIcuPort port) { mxm_iq(mxm, port).notify(); }

    // Called once after this cycle's dispatch/evaluate phase.  Notify is a
    // chip-wide event: after the modeled barrier latency, every SliceIcu gets
    // one token and any Sync at a queue head can retire on its next dispatch.
    void advance_barrier_events()
    {
        for (auto& remaining : barrier_events_) {
            if (remaining > 0) {
                --remaining;
            }
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
        for (auto& ports : mxm_iqs_) {
            for (auto& iq : ports) iq.notify();
        }
    }

    std::size_t barrier_latency_cycles() const noexcept
    {
        return barrier_latency_cycles_;
    }

    std::size_t pending_barrier_event_count() const noexcept
    {
        return barrier_events_.size();
    }

    SliceIcu<VxmLaneAluInstruction>& vxm_iq(std::size_t alu)
    {
        check_vxm_queue(alu);
        return vxm_iqs_[alu];
    }

    const SliceIcu<VxmLaneAluInstruction>& vxm_iq(std::size_t alu) const
    {
        check_vxm_queue(alu);
        return vxm_iqs_[alu];
    }

    SliceIcu<MemInstruction>& mem_iq(std::size_t column)
    {
        check_mem_queue(column);
        return mem_iqs_[column];
    }

    const SliceIcu<MemInstruction>& mem_iq(std::size_t column) const
    {
        check_mem_queue(column);
        return mem_iqs_[column];
    }

    SliceIcu<MxmControlInstruction>& mxm_iq(std::size_t mxm, MxmIcuPort port)
    {
        check_mxm_queue(mxm);
        return mxm_iqs_[mxm][static_cast<std::size_t>(port)];
    }

    const SliceIcu<MxmControlInstruction>& mxm_iq(
        std::size_t mxm,
        MxmIcuPort port) const
    {
        check_mxm_queue(mxm);
        return mxm_iqs_[mxm][static_cast<std::size_t>(port)];
    }

    // Fetch frontend is intentionally separate from dispatch so a system can
    // read current SR state before functional slices write next SR state.
    void evaluate_fetches(
        StreamRegisterFabric& fabric,
        const IcuFetchPortMap& ports)
    {
        for (std::size_t alu = 0; alu < kVxmQueues; ++alu) {
            vxm_iqs_[alu].evaluate_fetch(fabric, ports.column(IcuLocation::Vxm(alu)));
        }
        for (std::size_t column = 0; column < kMemQueues; ++column) {
            mem_iqs_[column].evaluate_fetch(fabric, ports.column(IcuLocation::Mem(column)));
        }
        for (std::size_t mxm = 0; mxm < kMxmUnitCount; ++mxm) {
            mxm_iqs_[mxm][static_cast<std::size_t>(MxmIcuPort::Load)].evaluate_fetch(
                fabric, ports.column(IcuLocation::MxmLoad(mxm)));
            mxm_iqs_[mxm][static_cast<std::size_t>(MxmIcuPort::Compute)].evaluate_fetch(
                fabric, ports.column(IcuLocation::MxmCompute(mxm)));
        }
    }

    // Minimal reset/bootstrap state for a MEM ICU: the address points into
    // that same MEM slice's SRAM. No MEM Read instruction is injected by C++.
    void bootstrap_mem_icu_from_local_sram(
        std::size_t mem_slice,
        MemLocalWordAddress13 base_address)
    {
        check_mem_queue(mem_slice);
        if (base_address.word() + hw::kIcuFetchVectorCount
            > hw::kSramWordsPerBank) {
            throw StaticScheduleError(
                "MEM ICU local bootstrap block cannot cross an SRAM bank");
        }
        if (mem_local_fetches_[mem_slice].has_value()) {
            throw StaticScheduleError(
                "MEM ICU already has an active local SRAM bootstrap fetch");
        }
        mem_iqs_[mem_slice].begin_local_fetch();
        mem_local_fetches_[mem_slice] = MemIcuLocalFetchState {
            base_address, 0, {}, false};
    }

    void evaluate_mem_local_fetches(const MemArrayModel& memory)
    {
        for (std::size_t mem_slice = 0; mem_slice < kMemQueues; ++mem_slice) {
            auto& state_slot = mem_local_fetches_[mem_slice];
            if (!state_slot.has_value() || state_slot->completion_staged) {
                continue;
            }
            auto& state = *state_slot;
            const auto vector = memory.read_sram_vector(
                mem_slice,
                state.base_address.advance_words(state.next_vector));
            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
                auto& packet = state.packets[
                    state.next_vector * hw::kTileRows + tile];
                for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                    packet.bytes[lane] = vector[tile][lane];
                }
            }
            ++state.next_vector;
            if (state.next_vector == hw::kIcuFetchVectorCount) {
                mem_iqs_[mem_slice].stage_fetched_packets(state.packets);
                state.completion_staged = true;
            }
        }
    }

    bool mem_local_fetch_active(std::size_t mem_slice) const
    {
        check_mem_queue(mem_slice);
        return mem_local_fetches_[mem_slice].has_value();
    }

    const MemIcuLocalFetchState* mem_local_fetch_state(
        std::size_t mem_slice) const
    {
        check_mem_queue(mem_slice);
        const auto& state = mem_local_fetches_[mem_slice];
        return state.has_value() ? &*state : nullptr;
    }

    void commit_fetches()
    {
        for (auto& iq : vxm_iqs_) iq.commit_fetch();
        for (std::size_t mem_slice = 0; mem_slice < kMemQueues; ++mem_slice) {
            const auto local_completion = mem_local_fetches_[mem_slice].has_value()
                && mem_local_fetches_[mem_slice]->completion_staged;
            mem_iqs_[mem_slice].commit_fetch();
            if (local_completion) {
                mem_local_fetches_[mem_slice].reset();
            }
        }
        for (auto& ports : mxm_iqs_) {
            for (auto& iq : ports) iq.commit_fetch();
        }
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
                    << describe_vxm(*instruction) << '\n';
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
        log_cycle_header(os);
        bool any = false;

        for (std::size_t alu = 0; alu < kVxmQueues; ++alu) {
            const auto instruction = vxm_iqs_[alu].tick();
            if (!instruction.has_value()) continue;
            vxm.issue_south(alu, *instruction);
            any = true;
            if (os != nullptr) *os << "  ICU -> VXM alu" << alu << ' ' << describe_vxm(*instruction) << '\n';
        }
        for (std::size_t column = 0; column < kMemQueues; ++column) {
            const auto instruction = mem_iqs_[column].tick();
            if (!instruction.has_value()) continue;
            mem.enqueue_instruction(column, *instruction);
            any = true;
            if (os != nullptr) *os << "  ICU -> MEM q" << column << ' ' << describe_mem(*instruction) << '\n';
        }
        for (std::size_t mxm = 0; mxm < kMxmUnitCount; ++mxm) {
            for (std::size_t port = 0; port < kMxmIcusPerUnit; ++port) {
                const auto instruction = mxm_iqs_[mxm][port].tick();
                if (!instruction.has_value()) continue;
                mxms[mxm].control().issue_south(*instruction);
                any = true;
                if (os != nullptr) {
                    *os << "  ICU -> MXM" << mxm
                        << (port == static_cast<std::size_t>(MxmIcuPort::Load) ? ".load " : ".compute ")
                        << describe_mxm(*instruction) << '\n';
                }
            }
        }

        log_dispatch_idle(os, any);
        ++cycle_;
    }

    std::size_t cycle() const noexcept { return cycle_; }

private:
    std::size_t take_emitted_notifications()
    {
        std::size_t count = 0;
        for (auto& iq : vxm_iqs_) count += iq.take_notify() ? 1 : 0;
        for (auto& iq : mem_iqs_) count += iq.take_notify() ? 1 : 0;
        for (auto& ports : mxm_iqs_) {
            for (auto& iq : ports) count += iq.take_notify() ? 1 : 0;
        }
        return count;
    }

    template <typename QueueArray>
    static void reset_all(QueueArray& queues)
    {
        for (auto& queue : queues) queue.reset();
    }

    static void check_vxm_queue(std::size_t alu)
    {
        if (alu >= kVxmQueues) throw std::out_of_range("ICU VXM queue is outside the 16 ALU queues");
    }

    static void check_mem_queue(std::size_t column)
    {
        if (column >= kMemQueues) throw std::out_of_range("ICU MEM queue is outside the 44 MEM queues");
    }

    static void check_mxm_queue(std::size_t mxm)
    {
        if (mxm >= kMxmUnitCount) throw std::out_of_range("ICU MXM unit index is outside the two MXM units");
    }

    template <typename QueueArray>
    static std::size_t queued_instruction_count(const QueueArray& queues)
    {
        std::size_t count = 0;
        for (const auto& queue : queues) count += queue.queued_count();
        return count;
    }

    void log_cycle_header(std::ostream* os) const
    {
        if (os == nullptr) return;
        *os << "icu cycle " << cycle_ << '\n'
            << "  queues: vxm=" << queued_instruction_count(vxm_iqs_)
            << " mem=" << queued_instruction_count(mem_iqs_)
            << " mxm=" << queued_mxm_instruction_count() << '\n';
    }

    static void log_dispatch_idle(std::ostream* os, bool any)
    {
        if (os != nullptr && !any) *os << "  ICU dispatch idle\n";
    }

    std::size_t queued_mxm_instruction_count() const
    {
        std::size_t count = 0;
        for (const auto& ports : mxm_iqs_) {
            count += queued_instruction_count(ports);
        }
        return count;
    }

    static std::string describe_mem(const MemInstruction& instruction)
    {
        std::ostringstream os;
        const char* opcode = "?";
        switch (instruction.opcode) {
        case MemOpcode::Read: opcode = "Read"; break;
        case MemOpcode::Write: opcode = "Write"; break;
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
            os << "IW b" << instruction.weight_buffer;
        } else {
            os << "Compute b" << instruction.weight_buffer
               << " acc=" << instruction.accumulator_bank
               << (instruction.accumulate ? "+=" : "=")
               << " stream=" << instruction.activation_stream_base
               << " out=" << instruction.stream_base
               << (instruction.reduce ? " reduce" : " retain");
        }
        return os.str();
    }

    static std::string describe_vxm(const VxmLaneAluInstruction& instruction)
    {
        std::ostringstream os;
        os << VxmLane::opcode_name(instruction.opcode);
        if (instruction.output_stream.has_value()) {
            os << " out_stream=" << *instruction.output_stream;
        }
        return os.str();
    }

    std::array<SliceIcu<VxmLaneAluInstruction>, kVxmQueues> vxm_iqs_{};
    std::array<SliceIcu<MemInstruction>, kMemQueues> mem_iqs_{};
    std::array<std::optional<MemIcuLocalFetchState>, kMemQueues>
        mem_local_fetches_{};
    std::array<
        std::array<SliceIcu<MxmControlInstruction>, kMxmIcusPerUnit>,
        kMxmUnitCount> mxm_iqs_{};
    std::size_t barrier_latency_cycles_{hw::kIcuBarrierLatencyCycles};
    std::deque<std::size_t> barrier_events_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
