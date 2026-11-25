#pragma once

#include "ftlpu/icu/qctl_hardware_interface.hpp"
#include "ftlpu/icu/qctl_distributed_program_store.hpp"
#include "ftlpu/icu/queue_control_pool.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace ftlpu {

inline constexpr IcuQueueControl qctl_zero_record() noexcept
{
    return {0, static_cast<IcuControlOpcode>(0), 0};
}

// Physical 96-bit QCTL storage and its single read port.  This object owns
// only SRAM/directory/arbitration state.  Per-Queue fetch and retirement
// offsets deliberately live in IcuQctlCache.
class IcuQctlPool {
public:
    struct Partition {
        std::size_t base{0};
        std::size_t count{0};
        std::uint64_t generation{0};
    };
    struct Response {
        IcuQueueControl record{};
        bool error{false};
        bool initialized{true};
    };
    struct Statistics {
        std::uint64_t cycles{0};
        std::uint64_t reads{0};
        std::size_t peak_pending{0};
        std::size_t maximum_arbitration_wait{0};
    };

    void configure(std::size_t depth, std::size_t queue_count,
        std::size_t read_latency, std::size_t initiation_interval = 1)
    {
        if (depth == 0 || queue_count == 0 || read_latency == 0
            || initiation_interval == 0)
            throw StaticScheduleError("invalid QCTL Pool geometry");
        depth_ = depth;
        latency_ = read_latency;
        initiation_interval_ = initiation_interval;
        records_.assign(depth, std::nullopt);
        partitions_.assign(queue_count, std::nullopt);
        requests_.assign(queue_count, std::nullopt);
        grants_.assign(queue_count, false);
        responses_.assign(queue_count, std::nullopt);
        pending_.clear();
        round_robin_ = 0;
        next_request_cycle_ = 0;
        cycle_ = 0;
        open_ = false;
        statistics_ = {};
        last_grant_valid_ = false;
        last_response_valid_ = false;
        fault_ = false;
        fault_code_ = 0;
    }

    void reset()
    {
        for (auto& partition : partitions_) partition.reset();
        for (auto& request : requests_) request.reset();
        std::fill(grants_.begin(), grants_.end(), false);
        for (auto& response : responses_) response.reset();
        pending_.clear();
        round_robin_ = 0;
        next_request_cycle_ = 0;
        cycle_ = 0;
        open_ = false;
        statistics_ = {};
        last_grant_valid_ = false;
        last_response_valid_ = false;
        fault_ = false;
        fault_code_ = 0;
    }

    void configure_partition(std::size_t queue, std::size_t base,
        std::size_t count, std::uint64_t generation)
    {
        check_queue(queue);
        if (generation == 0 || base > depth_ || count > depth_ - base
            || partitions_[queue].has_value())
            throw StaticScheduleError("invalid QCTL Pool partition");
        for (std::size_t other = 0; other < partitions_.size(); ++other) {
            if (other == queue || !partitions_[other] || count == 0
                || partitions_[other]->count == 0)
                continue;
            const auto& value = *partitions_[other];
            if (base < value.base + value.count && value.base < base + count)
                throw StaticScheduleError("overlapping QCTL Pool partitions");
        }
        partitions_[queue] = Partition{base, count, generation};
    }

    void load(std::size_t address, IcuQueueControl record)
    {
        if (address >= records_.size())
            throw StaticScheduleError("QCTL Pool write outside SRAM");
        records_[address] = record;
    }

    const Partition& partition(std::size_t queue) const
    {
        check_queue(queue);
        if (!partitions_[queue])
            throw StaticScheduleError("QCTL Pool Queue has no partition");
        return *partitions_[queue];
    }

    std::size_t depth() const noexcept { return depth_; }
    const std::optional<IcuQueueControl>& committed_record(std::size_t address) const
    { return records_.at(address); }

    void begin_cycle()
    {
        if (open_) throw StaticScheduleError("QCTL Pool cycle opened twice");
        open_ = true;
        last_response_valid_ = false;
        while (!pending_.empty() && pending_.front().ready_cycle <= cycle_) {
            auto pending = std::move(pending_.front());
            pending_.pop_front();
            if (responses_[pending.queue])
                throw StaticScheduleError("QCTL Pool response was not consumed");
            responses_[pending.queue] = Response{
                pending.record, pending.error, pending.initialized};
            last_response_valid_ = true;
            last_response_queue_ = pending.queue;
            last_response_ = *responses_[pending.queue];
        }
    }

    void request(std::size_t queue, std::size_t offset,
        std::uint64_t generation)
    {
        if (!open_) throw StaticScheduleError("QCTL Pool request outside cycle");
        const auto& descriptor = partition(queue);
        if (generation != descriptor.generation || offset >= descriptor.count)
            throw StaticScheduleError("invalid QCTL Pool indexed read");
        const Request request{offset, generation, cycle_};
        if (requests_[queue] && (requests_[queue]->offset != offset
                || requests_[queue]->generation != generation))
            throw StaticScheduleError("QCTL Pool request changed while stalled");
        if (!requests_[queue]) requests_[queue] = request;
    }

    bool take_grant(std::size_t queue)
    {
        check_queue(queue);
        const bool granted = grants_[queue];
        grants_[queue] = false;
        return granted;
    }

    std::optional<Response> take_response(std::size_t queue)
    {
        check_queue(queue);
        return std::exchange(responses_[queue], std::nullopt);
    }

    void end_cycle(bool allow_read = true)
    {
        if (!open_) throw StaticScheduleError("QCTL Pool cycle has no begin");
        last_grant_valid_ = false;
        if (allow_read && !fault_ && cycle_ >= next_request_cycle_ && !requests_.empty()) {
            for (std::size_t step = 0; step < requests_.size(); ++step) {
                const auto queue = (round_robin_ + step) % requests_.size();
                if (!requests_[queue]) continue;
                const auto request = *requests_[queue];
                const auto& descriptor = partition(queue);
                const auto address = descriptor.base + request.offset;
                const bool error = request.generation != descriptor.generation
                    || request.offset >= descriptor.count || address >= records_.size();
                const bool initialized = error || records_[address].has_value();
                if (error && !fault_) {
                    fault_ = true;
                    fault_code_ = 4;
                }
                statistics_.maximum_arbitration_wait = std::max(
                    statistics_.maximum_arbitration_wait,
                    static_cast<std::size_t>(cycle_ - request.first_cycle));
                // RTL stage zero captures the synchronous SRAM read and the
                // configured READ_LATENCY stages follow it.  The response is
                // therefore observable READ_LATENCY+1 edges after a grant.
                pending_.push_back(Pending{cycle_ + latency_ + 1, queue,
                    error || !initialized ? qctl_zero_record() : *records_[address],
                    error, initialized});
                requests_[queue].reset();
                grants_[queue] = true;
                last_grant_valid_ = true;
                last_grant_queue_ = queue;
                last_grant_index_ = request.offset;
                last_grant_generation_ = request.generation;
                round_robin_ = (queue + 1) % requests_.size();
                next_request_cycle_ = cycle_ + initiation_interval_;
                ++statistics_.reads;
                statistics_.peak_pending = std::max(
                    statistics_.peak_pending, pending_.size());
                break;
            }
        }
        ++cycle_;
        ++statistics_.cycles;
        open_ = false;
    }

    std::uint64_t cycle() const noexcept { return cycle_; }
    const Statistics& statistics() const noexcept { return statistics_; }
    bool last_grant_valid() const noexcept { return last_grant_valid_; }
    std::size_t last_grant_queue() const noexcept { return last_grant_queue_; }
    std::size_t last_grant_index() const noexcept { return last_grant_index_; }
    std::uint64_t last_grant_generation() const noexcept
    { return last_grant_generation_; }
    bool last_response_valid() const noexcept { return last_response_valid_; }
    std::size_t last_response_queue() const noexcept
    { return last_response_queue_; }
    const Response& last_response() const noexcept { return last_response_; }
    bool fault() const noexcept { return fault_; }
    std::uint8_t fault_code() const noexcept { return fault_code_; }

    // Combinational view of the single physical read port.  Cache requests
    // are presented before POSEDGE; end_cycle() commits exactly this
    // round-robin choice.
    std::optional<std::size_t> preview_grant(
        const std::vector<bool>& request_valid) const
    {
        if (request_valid.size() != requests_.size())
            throw StaticScheduleError("QCTL Pool request vector width mismatch");
        if (fault_ || cycle_ < next_request_cycle_ || request_valid.empty())
            return std::nullopt;
        for (std::size_t step = 0; step < request_valid.size(); ++step) {
            const auto queue = (round_robin_ + step) % request_valid.size();
            if (request_valid[queue]) return queue;
        }
        return std::nullopt;
    }

private:
    struct Request {
        std::size_t offset{0};
        std::uint64_t generation{0};
        std::uint64_t first_cycle{0};
    };
    struct Pending {
        std::uint64_t ready_cycle{0};
        std::size_t queue{0};
        IcuQueueControl record{};
        bool error{false};
        bool initialized{true};
    };

    void check_queue(std::size_t queue) const
    {
        if (queue >= partitions_.size())
            throw StaticScheduleError("QCTL Pool Queue index is out of range");
    }

    std::size_t depth_{0};
    std::size_t latency_{0};
    std::size_t initiation_interval_{1};
    std::vector<std::optional<IcuQueueControl>> records_{};
    std::vector<std::optional<Partition>> partitions_{};
    std::vector<std::optional<Request>> requests_{};
    std::vector<bool> grants_{};
    std::vector<std::optional<Response>> responses_{};
    std::deque<Pending> pending_{};
    std::size_t round_robin_{0};
    std::uint64_t next_request_cycle_{0};
    std::uint64_t cycle_{0};
    bool open_{false};
    Statistics statistics_{};
    bool last_grant_valid_{false};
    std::size_t last_grant_queue_{0};
    std::size_t last_grant_index_{0};
    std::uint64_t last_grant_generation_{0};
    bool last_response_valid_{false};
    std::size_t last_response_queue_{0};
    Response last_response_{};
    bool fault_{false};
    std::uint8_t fault_code_{0};
};

// Per-Queue fetch/cache state.  It is intentionally separate from both the
// physical Pool and the semantic Controller.
class IcuQctlCache {
public:
    void configure(IcuQctlPool& pool, std::size_t queue,
        std::size_t cache_depth)
    {
        if (cache_depth == 0)
            throw StaticScheduleError("QCTL Cache depth must be non-zero");
        pool_ = &pool;
        queue_ = queue;
        cache_depth_ = cache_depth;
        generation_ = pool.partition(queue).generation;
        count_ = pool.partition(queue).count;
        configured_ = true;
        reset_runtime();
    }

    void rebind_pool(IcuQctlPool& pool) noexcept { pool_ = &pool; }

    void reset_runtime()
    {
        cache_.clear();
        fetch_offset_ = 0;
        retire_offset_ = 0;
        in_flight_ = 0;
        request_pending_ = false;
        primed_ = count_ == 0;
        fault_ = false;
        fault_code_ = 0;
    }

    void begin_cycle()
    {
        ensure_configured();
        if (pool_->take_grant(queue_)) {
            if (!request_pending_ || fetch_offset_ >= count_) {
                if (!fault_) { fault_ = true; fault_code_ = 2; }
            } else {
                request_pending_ = false;
                ++fetch_offset_;
                ++in_flight_;
            }
        }
        if (auto response = pool_->take_response(queue_)) {
            if (in_flight_ == 0 || response->error) {
                if (!fault_) { fault_ = true; fault_code_ = 2; }
            } else if (cache_.size() >= cache_depth_) {
                if (!fault_) { fault_ = true; fault_code_ = 3; }
            } else {
                --in_flight_;
                cache_.push_back(response->initialized
                    ?std::optional{response->record}:std::nullopt);
            }
        }
        if (cache_.size() == std::min(cache_depth_, count_)) primed_ = true;
    }

    void end_cycle()
    {
        ensure_configured();
        if (!fault_ && !request_pending_ && fetch_offset_ < count_
            && cache_.size() + in_flight_ < cache_depth_) {
            pool_->request(queue_, fetch_offset_, generation_);
            request_pending_ = true;
        }
    }

    const IcuQueueControl* head() const noexcept
    { return cache_.empty() || !cache_.front() ? nullptr : &*cache_.front(); }
    bool undefined_head() const noexcept
    { return !cache_.empty() && !cache_.front(); }

    void retire()
    {
        if (cache_.empty())
        {
            if (!fault_) { fault_ = true; fault_code_ = 4; }
            return;
        }
        cache_.pop_front();
        ++retire_offset_;
    }

    bool primed() const noexcept { return primed_; }
    bool done() const noexcept { return retire_offset_ == count_; }
    std::size_t fetch_offset() const noexcept { return fetch_offset_; }
    std::size_t retire_offset() const noexcept { return retire_offset_; }
    std::size_t in_flight_reads() const noexcept { return in_flight_; }
    std::size_t level() const noexcept { return cache_.size(); }
    std::size_t count() const noexcept { return count_; }
    bool request_pending() const noexcept { return request_pending_; }
    bool configured() const noexcept { return configured_; }
    bool fault() const noexcept { return fault_; }
    std::uint8_t fault_code() const noexcept { return fault_code_; }
    std::uint64_t generation() const noexcept { return generation_; }
    bool wants_request() const noexcept
    {
        return pool_ && !fault_ && !request_pending_ && fetch_offset_ < count_
            && cache_.size() + in_flight_ < cache_depth_;
    }
    bool request_valid() const noexcept
    {
        return request_pending_ || wants_request();
    }

private:
    void ensure_configured() const
    {
        if (!pool_) throw StaticScheduleError("QCTL Cache is not configured");
    }

    IcuQctlPool* pool_{nullptr};
    std::size_t queue_{0};
    std::size_t cache_depth_{0};
    std::size_t count_{0};
    std::uint64_t generation_{0};
    std::deque<std::optional<IcuQueueControl>> cache_{};
    std::size_t fetch_offset_{0};
    std::size_t retire_offset_{0};
    std::size_t in_flight_{0};
    bool request_pending_{false};
    bool primed_{false};
    bool configured_{false};
    bool fault_{false};
    std::uint8_t fault_code_{0};
};

enum class IcuQctlControllerState : std::uint8_t {
    IDLE,
    SYNC_WAIT,
    EVENT_WAIT,
    NOTIFY_WAIT,
    MEMPACKET_WAIT,
};

struct IcuQctlControllerInputs {
    std::uint32_t scheduler_cycle{0};
    const IcuQueueControl* head{nullptr};
    bool event_valid{false};
    std::uint32_t event_tag{0};
    bool sync_release{false};
    bool notify_ready{false};
    bool mem_packet_retired{false};
    bool run{true};
    bool configured{true};
    bool primed{true};
    bool done{false};
};

struct IcuQctlControllerOutputs {
    bool scheduler_counter_ce{false};
    bool retire_head{false};
    bool sync_arrival_valid{false};
    bool event_wait{false};
    bool event_ready{false};
    bool event_fire{false};
    std::uint32_t wait_event_tag{0};
    bool notify_valid{false};
    bool notify_fire{false};
    bool mem_packet_valid{false};
    std::uint32_t mem_packet_ordinal{0};
    bool fault{false};
    std::uint8_t fault_code{0};
};

// Queue-local semantic FSM.  It has no local cycle counter, Pool address,
// cache FIFO, fetch offset, retire offset or in-flight read ownership.
class IcuQctlController {
public:
    void reset() noexcept
    {
        state_ = IcuQctlControllerState::IDLE;
        held_operand_ = 0;
        fault_ = false;
        fault_code_ = 0;
    }

    IcuQctlControllerOutputs evaluate(
        const IcuQctlControllerInputs& inputs) const
    {
        IcuQctlControllerOutputs outputs{};
        outputs.fault = fault_;
        outputs.fault_code = fault_code_;
        if (fault_ || !inputs.run || !inputs.configured) return outputs;
        if (!inputs.primed) return outputs;
        if (state_ == IcuQctlControllerState::IDLE) {
            if (inputs.done) {
                outputs.scheduler_counter_ce = true;
                return outputs;
            }
            // Empty-but-not-done Cache state is a safe stall.  The Scheduler
            // may advance only after the next absolute boundary is visible.
            if (!inputs.head) return outputs;
            if (inputs.head->cycle > inputs.scheduler_cycle) {
                outputs.scheduler_counter_ce = true;
                return outputs;
            }
            if (inputs.head->cycle < inputs.scheduler_cycle) return outputs;
            outputs.scheduler_counter_ce = false;
            return drive(inputs.head->opcode, inputs.head->operand,
                inputs, outputs, true);
        }
        outputs.scheduler_counter_ce = false;
        switch (state_) {
        case IcuQctlControllerState::SYNC_WAIT:
            return drive(IcuControlOpcode::Sync, 0, inputs, outputs, false);
        case IcuQctlControllerState::EVENT_WAIT:
            return drive(IcuControlOpcode::WaitEvent, held_operand_, inputs, outputs, false);
        case IcuQctlControllerState::NOTIFY_WAIT:
            return drive(IcuControlOpcode::Notify, 0, inputs, outputs, false);
        case IcuQctlControllerState::MEMPACKET_WAIT:
            return drive(kQueueControlMemPacket, held_operand_, inputs, outputs, false);
        case IcuQctlControllerState::IDLE:
            break;
        }
        return outputs;
    }

    void commit(const IcuQctlControllerInputs& inputs,
        const IcuQctlControllerOutputs& outputs)
    {
        if (!fault_) {
            if (inputs.run && inputs.configured && !inputs.primed) {
                fault_ = true;
                fault_code_ = 1;
            } else if (state_ == IcuQctlControllerState::IDLE
                && inputs.run && inputs.configured && inputs.primed
                && inputs.head
                && inputs.head->cycle < inputs.scheduler_cycle) {
                fault_ = true;
                fault_code_ = 2;
            } else if (state_ == IcuQctlControllerState::IDLE
                && inputs.run && inputs.configured && inputs.primed
                && inputs.head
                && inputs.head->cycle == inputs.scheduler_cycle
                && !valid_record(*inputs.head)) {
                fault_ = true;
                fault_code_ = 3;
            }
        }
        if (outputs.retire_head) {
            state_ = IcuQctlControllerState::IDLE;
            held_operand_ = 0;
            return;
        }
        if (fault_ || !inputs.run || !inputs.configured || !inputs.primed
            || state_ != IcuQctlControllerState::IDLE || !inputs.head
            || inputs.head->cycle != inputs.scheduler_cycle)
            return;
        held_operand_ = inputs.head->operand;
        switch (inputs.head->opcode) {
        case IcuControlOpcode::Sync:
            state_ = IcuQctlControllerState::SYNC_WAIT;
            break;
        case IcuControlOpcode::WaitEvent:
            state_ = IcuQctlControllerState::EVENT_WAIT;
            break;
        case IcuControlOpcode::Notify:
            if (!outputs.retire_head)
                state_ = IcuQctlControllerState::NOTIFY_WAIT;
            break;
        default:
            if (!outputs.retire_head)
                state_ = IcuQctlControllerState::MEMPACKET_WAIT;
            break;
        }
    }

    IcuQctlControllerState state() const noexcept { return state_; }
    std::uint32_t held_operand() const noexcept { return held_operand_; }
    bool fault() const noexcept { return fault_; }
    std::uint8_t fault_code() const noexcept { return fault_code_; }

private:
    static IcuQctlControllerOutputs drive(IcuControlOpcode opcode,
        std::uint32_t operand, const IcuQctlControllerInputs& inputs,
        IcuQctlControllerOutputs outputs, bool entering)
    {
        switch (opcode) {
        case IcuControlOpcode::Sync:
            if (operand != 0) return outputs;
            outputs.sync_arrival_valid = true;
            // RTL always registers SYNC_WAIT on the boundary-hit edge.  A
            // release is consumed only by the already-waiting state.
            outputs.retire_head = !entering && inputs.sync_release;
            return outputs;
        case IcuControlOpcode::WaitEvent:
            outputs.wait_event_tag = operand;
            outputs.event_wait = true;
            outputs.event_ready = true; // compatibility alias for event_wait
            outputs.event_fire = inputs.event_valid
                && inputs.event_tag == operand;
            outputs.retire_head = outputs.event_fire;
            return outputs;
        case IcuControlOpcode::Notify:
            if (operand != 0) return outputs;
            outputs.notify_valid = true;
            outputs.notify_fire = inputs.notify_ready;
            outputs.retire_head = outputs.notify_fire;
            return outputs;
        default:
            if (opcode != kQueueControlMemPacket || operand == 0)
                return outputs;
            outputs.mem_packet_valid = true;
            outputs.mem_packet_ordinal = operand;
            outputs.retire_head = inputs.mem_packet_retired;
            return outputs;
        }
    }

    static bool valid_record(const IcuQueueControl& record) noexcept
    {
        return ((record.opcode == IcuControlOpcode::Sync
                    || record.opcode == IcuControlOpcode::Notify)
                   && record.operand == 0)
            || record.opcode == IcuControlOpcode::WaitEvent
            || (record.opcode == kQueueControlMemPacket
                && record.operand != 0);
    }

    IcuQctlControllerState state_{IcuQctlControllerState::IDLE};
    std::uint32_t held_operand_{0};
    bool fault_{false};
    std::uint8_t fault_code_{0};
};

// Shared arrival/release barrier.  SYNC arrivals are idempotent bits, not
// historical tokens.  A NOTIFY request may arrive early and must remain valid
// at its Controller until all configured participants have arrived.
class IcuQctlArrivalReleaseBarrier {
public:
    struct Outputs {
        std::vector<bool> release{};
        std::vector<bool> notify_ready{};
        bool completed{false};
        std::size_t notifier{std::numeric_limits<std::size_t>::max()};
    };

    void configure(std::vector<bool> participants)
    {
        if (participants.empty())
            throw StaticScheduleError("QCTL Barrier width is zero");
        participants_ = std::move(participants);
        arrived_.assign(participants_.size(), false);
        release_.assign(participants_.size(), false);
        round_robin_ = 0;
        fault_ = false;
        fault_code_ = 0;
    }

    void reset() noexcept
    {
        std::fill(arrived_.begin(), arrived_.end(), false);
        std::fill(release_.begin(), release_.end(), false);
        round_robin_ = 0;
        fault_ = false;
        fault_code_ = 0;
    }

    Outputs evaluate(const std::vector<bool>& sync_arrival,
        const std::vector<bool>& notify_valid) const
    {
        if (sync_arrival.size() != participants_.size()
            || notify_valid.size() != participants_.size())
            throw StaticScheduleError("QCTL Barrier vector width mismatch");
        auto combined = arrived_;
        for (std::size_t queue = 0; queue < combined.size(); ++queue)
            combined[queue] = combined[queue]
                || (participants_[queue] && sync_arrival[queue]);
        const bool all_arrived = [&] {
            for (std::size_t queue = 0; queue < combined.size(); ++queue)
                if (participants_[queue] && !combined[queue]) return false;
            return true;
        }();
        Outputs outputs{release_,
            std::vector<bool>(combined.size(), false)};
        if (!all_arrived || std::ranges::any_of(release_, [](bool value) {
                return value;
            })) return outputs;
        for (std::size_t step = 0; step < notify_valid.size(); ++step) {
            const auto queue = (round_robin_ + step) % notify_valid.size();
            if (!notify_valid[queue]) continue;
            outputs.notifier = queue;
            if (participants_[queue]) return outputs;
            outputs.completed = true;
            outputs.notify_ready[queue] = true;
            break;
        }
        return outputs;
    }

    void commit(const std::vector<bool>& sync_arrival,
        const Outputs& outputs)
    {
        if (sync_arrival.size() != participants_.size())
            throw StaticScheduleError("QCTL Barrier vector width mismatch");
        for (std::size_t queue = 0; queue < participants_.size(); ++queue) {
            if (!fault_ && participants_[queue] && outputs.notifier == queue) {
                fault_ = true;
                fault_code_ = 1;
            }
        }
        if (outputs.completed) {
            std::fill(arrived_.begin(), arrived_.end(), false);
            release_ = participants_;
            round_robin_ = (outputs.notifier + 1) % participants_.size();
            return;
        }
        if (std::ranges::any_of(release_, [](bool value) { return value; })) {
            std::fill(arrived_.begin(), arrived_.end(), false);
            std::fill(release_.begin(), release_.end(), false);
            return;
        }
        for (std::size_t queue = 0; queue < arrived_.size(); ++queue)
            arrived_[queue] = arrived_[queue]
                || (participants_[queue] && sync_arrival[queue]);
    }

    bool arrived(std::size_t queue) const
    {
        if (queue >= arrived_.size())
            throw StaticScheduleError("QCTL Barrier Queue index is out of range");
        return arrived_[queue];
    }
    bool fault() const noexcept { return fault_; }
    std::uint8_t fault_code() const noexcept { return fault_code_; }
    const std::vector<bool>& participants() const noexcept
    { return participants_; }

private:
    std::vector<bool> participants_{};
    std::vector<bool> arrived_{};
    std::vector<bool> release_{};
    std::size_t round_robin_{0};
    bool fault_{false};
    std::uint8_t fault_code_{0};
};

// DMA/Event-side holding register.  Early events live here, not in any QCTL
// Controller.  One 32-bit TAG is broadcast until every selected consumer has
// accepted it, after which the next queued event becomes visible.
class IcuQctlEventManager {
public:
    struct Event {
        std::uint32_t tag{0};
        std::vector<bool> consumers{};
        std::vector<bool> accepted{};
    };
    struct Selection {
        bool found{false};
        std::size_t index{0};
        std::uint32_t tag{0};
        std::vector<bool> visible{};
    };

    void configure(std::size_t queue_count, std::size_t depth)
    {
        if (queue_count == 0 || depth == 0)
            throw StaticScheduleError("invalid QCTL Event Manager geometry");
        queue_count_ = queue_count;
        depth_ = depth;
        events_.clear();
        fault_ = false;
        fault_code_ = 0;
    }

    void reset() noexcept
    {
        events_.clear();
        fault_ = false;
        fault_code_ = 0;
    }

    void push(std::uint32_t tag, std::vector<bool> consumers)
    {
        if (consumers.size() != queue_count_)
            throw StaticScheduleError("invalid QCTL Event consumer mask width");
        if (std::ranges::none_of(consumers, [](bool value) { return value; })) {
            if (!fault_) {
                fault_ = true;
                fault_code_ = 1;
            }
            return;
        }
        if (events_.size() >= depth_)
            throw StaticScheduleError("QCTL Event Manager overflow");
        events_.push_back(Event{tag, std::move(consumers),
            std::vector<bool>(queue_count_, false)});
    }

    bool valid() const noexcept { return !events_.empty(); }
    std::uint32_t tag() const
    {
        if (events_.empty())
            throw StaticScheduleError("QCTL Event TAG requested while empty");
        return events_.front().tag;
    }

    bool targets(std::size_t queue) const
    {
        if (events_.empty() || queue >= queue_count_)
            return false;
        return events_.front().consumers[queue]
            && !events_.front().accepted[queue];
    }

    Selection select(const std::vector<bool>& event_wait) const
    {
        if (event_wait.size() != queue_count_)
            throw StaticScheduleError("QCTL Event wait vector width mismatch");
        Selection result{};
        result.visible.assign(queue_count_, false);
        for (std::size_t index = 0; index < events_.size(); ++index) {
            for (std::size_t queue = 0; queue < queue_count_; ++queue) {
                result.visible[queue] = events_[index].consumers[queue]
                    && !events_[index].accepted[queue] && event_wait[queue];
            }
            if (std::ranges::any_of(result.visible,
                    [](bool value) { return value; })) {
                result.found = true;
                result.index = index;
                result.tag = events_[index].tag;
                return result;
            }
        }
        std::fill(result.visible.begin(), result.visible.end(), false);
        return result;
    }

    void commit(const std::vector<bool>& event_fire)
    {
        if (events_.empty()) {
            if (std::ranges::any_of(event_fire, [](bool value) { return value; }))
                throw StaticScheduleError("QCTL Event accepted while manager empty");
            return;
        }
        if (event_fire.size() != queue_count_)
            throw StaticScheduleError("QCTL Event fire vector width mismatch");
        auto& event = events_.front();
        for (std::size_t queue = 0; queue < queue_count_; ++queue) {
            if (event_fire[queue] && !event.consumers[queue])
                throw StaticScheduleError("non-consumer accepted QCTL Event");
            event.accepted[queue] = event.accepted[queue] || event_fire[queue];
        }
        for (std::size_t queue = 0; queue < queue_count_; ++queue)
            if (event.consumers[queue] && !event.accepted[queue]) return;
        events_.pop_front();
    }

    void commit(const Selection& selected,
        const std::vector<bool>& event_fire)
    {
        if (event_fire.size() != queue_count_)
            throw StaticScheduleError("QCTL Event fire vector width mismatch");
        for (std::size_t queue = 0; queue < queue_count_; ++queue) {
            if (event_fire[queue]
                && (!selected.found || !selected.visible[queue])) {
                if (!fault_) {
                    fault_ = true;
                    fault_code_ = 2;
                }
                return;
            }
        }
        if (!selected.found) return;
        auto& event = events_.at(selected.index);
        for (std::size_t queue = 0; queue < queue_count_; ++queue)
            event.accepted[queue] = event.accepted[queue] || event_fire[queue];
        for (std::size_t queue = 0; queue < queue_count_; ++queue)
            if (event.consumers[queue] && !event.accepted[queue]) return;
        events_.erase(events_.begin()
            + static_cast<std::ptrdiff_t>(selected.index));
    }

    std::size_t level() const noexcept { return events_.size(); }
    std::size_t capacity() const noexcept { return depth_; }
    bool selection_will_retire(const Selection& selected,
        const std::vector<bool>& event_fire) const
    {
        if (!selected.found) return false;
        const auto& event = events_.at(selected.index);
        for (std::size_t queue = 0; queue < queue_count_; ++queue)
            if (event.consumers[queue]
                && !event.accepted[queue] && !event_fire[queue]) return false;
        return true;
    }
    bool fault() const noexcept { return fault_; }
    std::uint8_t fault_code() const noexcept { return fault_code_; }

private:
    std::size_t queue_count_{0};
    std::size_t depth_{0};
    std::deque<Event> events_{};
    bool fault_{false};
    std::uint8_t fault_code_{0};
};

// Complete standalone CModel QCTL subsystem.  Scheduler counters remain
// external inputs; this runtime only returns each Queue's CE and handshakes.
class IcuQctlRuntime {
public:
    struct QueueOutputs : IcuQctlControllerOutputs {
        IcuQctlControllerState state{IcuQctlControllerState::IDLE};
        std::size_t fetch_offset{0};
        std::size_t retire_offset{0};
        std::size_t in_flight_reads{0};
        std::size_t cache_level{0};
        bool primed{false};
        bool done{false};
    };
    struct LoadInputs {
        bool clear{false};
        bool load_mode{true};
        bool partition_valid{false};
        std::size_t partition_queue{0};
        std::size_t partition_base{0};
        std::size_t partition_count{0};
        std::uint64_t partition_generation{0};
        bool load_valid{false};
        std::size_t load_address{0};
        IcuQueueControl load_data{qctl_zero_record()};
    };

    IcuQctlRuntime() = default;
    IcuQctlRuntime(const IcuQctlRuntime& other)
        : pool_(other.pool_), caches_(other.caches_),
          controllers_(other.controllers_), configured_(other.configured_),
          distributed_store_(other.distributed_store_),
          endpoint_count_(other.endpoint_count_),
          endpoint_generation_(other.endpoint_generation_),
          endpoint_retire_offset_(other.endpoint_retire_offset_),
          endpoint_fault_(other.endpoint_fault_),
          endpoint_fault_code_(other.endpoint_fault_code_),
          distributed_mode_(other.distributed_mode_),
          barrier_(other.barrier_), event_manager_(other.event_manager_),
          cache_depth_(other.cache_depth_),
          barrier_configured_(other.barrier_configured_),
          hardware_view_(other.hardware_view_),
          pending_event_in_valid_(other.pending_event_in_valid_),
          pending_event_in_tag_(other.pending_event_in_tag_),
          pending_event_in_targets_(other.pending_event_in_targets_),
          pending_event_input_preaccepted_(
              other.pending_event_input_preaccepted_),
          pool_configuration_fault_(other.pool_configuration_fault_),
          pool_configuration_fault_code_(
              other.pool_configuration_fault_code_)
    {
        for (auto& cache : caches_) cache.rebind_pool(pool_);
        cycle_open_ = false;
        load_cycle_open_ = false;
    }
    IcuQctlRuntime& operator=(const IcuQctlRuntime& other)
    {
        if (this == &other) return *this;
        pool_ = other.pool_;
        caches_ = other.caches_;
        controllers_ = other.controllers_;
        configured_ = other.configured_;
        distributed_store_ = other.distributed_store_;
        endpoint_count_ = other.endpoint_count_;
        endpoint_generation_ = other.endpoint_generation_;
        endpoint_retire_offset_ = other.endpoint_retire_offset_;
        endpoint_fault_ = other.endpoint_fault_;
        endpoint_fault_code_ = other.endpoint_fault_code_;
        distributed_mode_ = other.distributed_mode_;
        barrier_ = other.barrier_;
        event_manager_ = other.event_manager_;
        cache_depth_ = other.cache_depth_;
        barrier_configured_ = other.barrier_configured_;
        hardware_view_ = other.hardware_view_;
        pending_event_in_valid_ = other.pending_event_in_valid_;
        pending_event_in_tag_ = other.pending_event_in_tag_;
        pending_event_in_targets_ = other.pending_event_in_targets_;
        pending_event_input_preaccepted_ =
            other.pending_event_input_preaccepted_;
        pool_configuration_fault_ = other.pool_configuration_fault_;
        pool_configuration_fault_code_ = other.pool_configuration_fault_code_;
        for (auto& cache : caches_) cache.rebind_pool(pool_);
        cycle_open_ = false;
        load_cycle_open_ = false;
        return *this;
    }

    void configure(std::size_t pool_depth, std::size_t queue_count,
        std::size_t cache_depth, std::size_t read_latency,
        std::size_t initiation_interval = 1,
        std::size_t event_depth = 8)
    {
        if (cache_depth == 0)
            throw StaticScheduleError("QCTL Runtime cache depth is zero");
        pool_.configure(pool_depth, queue_count, read_latency,
            initiation_interval);
        caches_.clear();
        caches_.resize(queue_count);
        controllers_.clear();
        controllers_.resize(queue_count);
        configured_.assign(queue_count, false);
        endpoint_count_.assign(queue_count, 0);
        endpoint_generation_.assign(queue_count, 0);
        endpoint_retire_offset_.assign(queue_count, 0);
        endpoint_fault_.assign(queue_count, false);
        endpoint_fault_code_.assign(queue_count, 0);
        pending_heads_.assign(queue_count, std::nullopt);
        pending_head_error_.assign(queue_count, false);
        distributed_mode_ = queue_count
            == IcuQctlDistributedProgramStore::kQueueCount;
        if (distributed_mode_)
            distributed_store_.configure(pool_depth, queue_count, 16, 64, 2);
        cache_depth_ = cache_depth;
        event_manager_.configure(queue_count, event_depth);
        barrier_.configure(std::vector<bool>(queue_count, false));
        barrier_configured_ = true;
        initialize_hardware_view(queue_count);
        pending_event_in_targets_.assign(queue_count, false);
        pending_event_in_valid_ = false;
        pending_event_input_preaccepted_ = false;
        pool_configuration_fault_ = false;
        pool_configuration_fault_code_ = 0;
        cycle_open_ = false;
        load_cycle_open_ = false;
    }

    void reset()
    {
        pool_.reset();
        for (auto& cache : caches_) cache.reset_runtime();
        for (auto& controller : controllers_) controller.reset();
        std::fill(configured_.begin(), configured_.end(), false);
        std::fill(endpoint_count_.begin(), endpoint_count_.end(), 0);
        std::fill(endpoint_generation_.begin(), endpoint_generation_.end(), 0);
        std::fill(endpoint_retire_offset_.begin(), endpoint_retire_offset_.end(), 0);
        std::fill(endpoint_fault_.begin(), endpoint_fault_.end(), false);
        std::fill(endpoint_fault_code_.begin(), endpoint_fault_code_.end(), 0);
        std::fill(pending_heads_.begin(), pending_heads_.end(), std::nullopt);
        std::fill(pending_head_error_.begin(), pending_head_error_.end(), false);
        if (distributed_mode_) distributed_store_.reset();
        barrier_.reset();
        event_manager_.reset();
        if (!configured_.empty())
            barrier_.configure(std::vector<bool>(configured_.size(), false));
        barrier_configured_ = !configured_.empty();
        initialize_hardware_view(configured_.size());
        pending_event_in_targets_.assign(configured_.size(), false);
        pending_event_in_valid_ = false;
        pending_event_input_preaccepted_ = false;
        pool_configuration_fault_ = false;
        pool_configuration_fault_code_ = 0;
        cycle_open_ = false;
        load_cycle_open_ = false;
    }

    void configure_partition(std::size_t queue, std::size_t base,
        std::vector<IcuQueueControl> records, std::uint64_t generation)
    {
        check_queue(queue);
        validate_queue_control_entries(records);
        LoadInputs partition{};
        partition.partition_valid = true;
        partition.partition_queue = queue;
        partition.partition_base = base;
        partition.partition_count = records.size();
        partition.partition_generation = generation;
        load_cycle(partition);
        for (std::size_t offset = 0; offset < records.size(); ++offset) {
            LoadInputs load{};
            load.load_valid = true;
            load.load_address = base + offset;
            load.load_data = records[offset];
            load_cycle(load);
        }
    }

    void observe_program_load_inputs(const LoadInputs& inputs)
    {
        if (distributed_mode_) {
            auto& manager = hardware_view_.load_manager;
            manager.rst_ni = true;
            manager.clear_i = inputs.clear;
            manager.load_mode_i = inputs.load_mode;
            manager.partition_valid_i = inputs.partition_valid;
            manager.partition_queue_i = static_cast<std::uint32_t>(
                inputs.partition_queue);
            manager.partition_base_i = static_cast<std::uint32_t>(
                inputs.partition_base);
            manager.partition_count_i = static_cast<std::uint32_t>(
                inputs.partition_count);
            manager.partition_generation_i = inputs.partition_generation;
            manager.load_valid_i = inputs.load_valid;
            manager.load_address_i = static_cast<std::uint32_t>(
                inputs.load_address);
            manager.load_data_i = inputs.load_data;
            hardware_view_.pool.rst_ni = true;
            hardware_view_.pool.clear_i = inputs.clear;
            hardware_view_.pool.load_mode_i = inputs.load_mode;
            hardware_view_.pool.partition_valid_i = inputs.partition_valid;
            hardware_view_.pool.load_valid_i = inputs.load_valid;
            sample_global_aliases();
            return;
        }
        auto& view = hardware_view_.pool;
        view.rst_ni = true;
        view.clear_i = inputs.clear;
        view.load_mode_i = inputs.load_mode;
        view.partition_valid_i = inputs.partition_valid;
        view.partition_ready_o = inputs.load_mode
            && !pool_configuration_fault_ && !pool_.fault();
        view.partition_queue_i = static_cast<std::uint32_t>(inputs.partition_queue);
        view.partition_base_i = static_cast<std::uint32_t>(inputs.partition_base);
        view.partition_count_i = static_cast<std::uint32_t>(inputs.partition_count);
        view.partition_generation_i = inputs.partition_generation;
        view.load_valid_i = inputs.load_valid;
        view.load_ready_o = view.partition_ready_o;
        view.load_address_i = static_cast<std::uint32_t>(inputs.load_address);
        view.load_data_i = inputs.load_data;
        sample_global_aliases();
    }

    void begin_load_cycle(const LoadInputs& inputs, bool clock_read_pipeline = false)
    {
        if (cycle_open_ || load_cycle_open_) throw StaticScheduleError(
            "QCTL load cycle opened while another cycle is active");
        if (distributed_mode_) {
            begin_distributed_load_cycle(inputs);
            return;
        }
        pending_load_inputs_ = inputs;
        load_clocks_read_pipeline_ = clock_read_pipeline && !inputs.clear;
        if (load_clocks_read_pipeline_) {
            pool_.begin_cycle();
            for (std::size_t queue=0; queue<caches_.size(); ++queue)
                if (configured_[queue]) caches_[queue].begin_cycle();
        }
        load_cycle_open_ = true;
        auto& view = hardware_view_.pool;
        view = {};
        view.request_valid_i.assign(caches_.size(), false);
        view.request_ready_o.assign(caches_.size(), false);
        view.request_index_i.assign(caches_.size(), 0);
        view.request_generation_i.assign(caches_.size(), 0);
        observe_program_load_inputs(inputs);
        std::fill(hardware_view_.endpoint_configure_valid_o.begin(),
            hardware_view_.endpoint_configure_valid_o.end(), false);
        std::fill(hardware_view_.endpoint_configure_count_o.begin(),
            hardware_view_.endpoint_configure_count_o.end(), 0);
        std::fill(hardware_view_.endpoint_configure_generation_o.begin(),
            hardware_view_.endpoint_configure_generation_o.end(), 0);
        const auto clear_bools = std::vector<bool>(caches_.size(), false);
        const auto clear_u32s = std::vector<std::uint32_t>(
            caches_.size(), 0);
        auto& event_view = hardware_view_.event_buffer;
        event_view.event_wait_i = clear_bools;
        event_view.event_consume_i = clear_bools;
        event_view.event_valid_o = clear_bools;
        event_view.event_tag_o = 0;
        auto& fabric_view = hardware_view_.fabric;
        fabric_view.endpoint_event_wait_i = clear_bools;
        fabric_view.endpoint_event_consume_i = clear_bools;
        fabric_view.event_wait_o = clear_bools;
        fabric_view.event_consume_o = clear_bools;
        fabric_view.event_buffer_valid_i = clear_bools;
        fabric_view.event_buffer_tag_i = 0;
        fabric_view.endpoint_event_valid_o = clear_bools;
        fabric_view.endpoint_event_tag_o = 0;
        fabric_view.endpoint_sync_arrive_valid_i = clear_bools;
        fabric_view.endpoint_notify_valid_i = clear_bools;
        fabric_view.endpoint_notify_ready_o = clear_bools;
        fabric_view.endpoint_sync_release_valid_o = clear_bools;
        fabric_view.endpoint_mem_packet_valid_i = clear_bools;
        fabric_view.endpoint_mem_packet_ordinal_i = clear_u32s;
        fabric_view.mem_packet_valid_o = clear_bools;
        fabric_view.mem_packet_ordinal_o = clear_u32s;
        fabric_view.mem_packet_retired_i = clear_bools;
        fabric_view.endpoint_mem_packet_retired_o = clear_bools;
        for (std::size_t queue = 0; queue < hardware_view_.endpoints.size();
             ++queue) {
            auto& endpoint = hardware_view_.endpoints[queue];
            endpoint.configure_valid_i = false;
            endpoint.scheduler_counter_ce_o = false;
            endpoint.event_valid_i = false;
            endpoint.event_tag_i = 0;
            endpoint.event_wait_o = false;
            endpoint.event_consume_o = false;
            endpoint.sync_arrive_valid_o = false;
            endpoint.sync_release_valid_i = false;
            endpoint.notify_valid_o = false;
            endpoint.notify_ready_i = false;
            endpoint.mem_packet_valid_o = false;
            endpoint.mem_packet_ordinal_o = 0;
            endpoint.mem_packet_retired_i = false;
            hardware_view_.caches[queue].configure_valid_i = false;
            auto& controller = hardware_view_.controllers[queue];
            controller.configure_valid_i = false;
            controller.scheduler_counter_ce_o = false;
            controller.event_valid_i = false;
            controller.event_tag_i = 0;
            controller.event_wait_o = false;
            controller.event_consume_o = false;
            controller.sync_arrive_valid_o = false;
            controller.sync_release_valid_i = false;
            controller.notify_valid_o = false;
            controller.notify_ready_i = false;
            controller.mem_packet_valid_o = false;
            controller.mem_packet_ordinal_o = 0;
            controller.mem_packet_retired_i = false;
        }
        if (inputs.partition_valid && view.partition_ready_o
            && inputs.partition_queue < configured_.size()
            && inputs.partition_count <= 16 && inputs.partition_generation != 0
            && inputs.partition_base <= pool_.depth()
            && inputs.partition_count <= pool_.depth() - inputs.partition_base) {
            hardware_view_.endpoint_configure_valid_o[
                inputs.partition_queue] = true;
            hardware_view_.endpoint_configure_count_o[
                inputs.partition_queue] = static_cast<std::uint32_t>(
                    inputs.partition_count);
            hardware_view_.endpoint_configure_generation_o[
                inputs.partition_queue] = inputs.partition_generation;
            auto& endpoint = hardware_view_.endpoints[inputs.partition_queue];
            endpoint.configure_valid_i = true;
            endpoint.configure_count_i = static_cast<std::uint32_t>(
                inputs.partition_count);
            endpoint.configure_generation_i = inputs.partition_generation;
            auto& cache = hardware_view_.caches[inputs.partition_queue];
            cache.configure_valid_i = true;
            cache.configure_count_i = endpoint.configure_count_i;
            cache.configure_generation_i = endpoint.configure_generation_i;
            hardware_view_.controllers[inputs.partition_queue]
                .configure_valid_i = true;
        }
        // LOAD mode disables Pool reads in RTL, but configured Cache request
        // intent remains observable at the endpoint/Pool boundary.  Sample
        // that intent without granting a read while the load port owns the
        // shared Pool.
        sample_pool_interface(false);
        auto& event = hardware_view_.event_buffer;
        event.event_in_valid_i = pending_event_in_valid_;
        event.event_in_ready_o = event_manager_.level()
            < event_manager_.capacity();
        event.event_in_tag_i = pending_event_in_tag_;
        event.event_in_target_mask_i = pending_event_in_targets_.empty()
            ? std::vector<bool>(caches_.size(), false)
            : pending_event_in_targets_;
        hardware_view_.fabric.event_in_valid_i = event.event_in_valid_i;
        hardware_view_.fabric.event_in_ready_i = event.event_in_ready_o;
        hardware_view_.fabric.event_backpressure_o = event.event_in_valid_i
            && !event.event_in_ready_o;
        sample_global_status_only();
        sample_global_aliases();
    }

    void complete_load_cycle()
    {
        if (!load_cycle_open_)
            throw StaticScheduleError("QCTL load cycle has no begin");
        if (distributed_mode_) {
            complete_distributed_load_cycle();
            return;
        }
        const auto inputs = pending_load_inputs_;
        auto& view = hardware_view_.pool;
        if (inputs.clear) {
            pool_.reset();
            // Directory clear is independent of the always-clocked SRAM
            // write port. A concurrent accepted in-range write still lands.
            if (inputs.load_valid && view.load_ready_o
                && inputs.load_address < pool_.depth())
                pool_.load(inputs.load_address,inputs.load_data);
            for (auto& cache : caches_) cache.reset_runtime();
            for (auto& controller : controllers_) controller.reset();
            std::fill(configured_.begin(), configured_.end(), false);
            barrier_.reset();
            event_manager_.reset();
            pool_configuration_fault_ = false;
            pool_configuration_fault_code_ = 0;
            load_cycle_open_ = false;
            return;
        }
        if (inputs.partition_valid && view.partition_ready_o) {
            try {
                if (inputs.partition_count > 16)
                    throw StaticScheduleError(
                        "QCTL partition entry limit exceeded");
                check_queue(inputs.partition_queue);
                pool_.configure_partition(inputs.partition_queue,
                    inputs.partition_base, inputs.partition_count,
                    inputs.partition_generation);
                caches_[inputs.partition_queue].configure(pool_,
                    inputs.partition_queue, cache_depth_);
                controllers_[inputs.partition_queue].reset();
                configured_[inputs.partition_queue] = true;
            } catch (const StaticScheduleError&) {
                if (!pool_configuration_fault_) {
                    bool overlap = false;
                    if (inputs.partition_queue < configured_.size()
                        && inputs.partition_generation != 0
                        && inputs.partition_count <= 16) {
                        for (std::size_t queue = 0;
                             queue < configured_.size(); ++queue) {
                            if (!configured_[queue]
                                || inputs.partition_count == 0) continue;
                            const auto& partition = pool_.partition(queue);
                            overlap = inputs.partition_base
                                    < partition.base + partition.count
                                && partition.base < inputs.partition_base
                                    + inputs.partition_count;
                            if (overlap) break;
                        }
                    }
                    pool_configuration_fault_ = true;
                    pool_configuration_fault_code_ = overlap ? 2 : 1;
                }
            }
        }
        if (inputs.load_valid && view.load_ready_o) {
            try {
                pool_.load(inputs.load_address, inputs.load_data);
            } catch (const StaticScheduleError&) {
                if (!pool_configuration_fault_) {
                    pool_configuration_fault_ = true;
                    pool_configuration_fault_code_ = 3;
                }
            }
        }
        view.fault_o = pool_configuration_fault_ || pool_.fault();
        view.fault_code_o = pool_configuration_fault_
            ? pool_configuration_fault_code_ : pool_.fault_code();
        if (load_clocks_read_pipeline_) pool_.end_cycle(false);
        load_clocks_read_pipeline_ = false;
        load_cycle_open_ = false;
    }

    void load_cycle(const LoadInputs& inputs)
    {
        begin_load_cycle(inputs);
        complete_load_cycle();
    }

    void configure_barrier(std::vector<bool> participants)
    {
        if (participants.size() != configured_.size())
            throw StaticScheduleError("QCTL Runtime barrier width mismatch");
        barrier_.configure(std::move(participants));
        barrier_configured_ = true;
    }

    void push_event(std::uint32_t tag, std::vector<bool> consumers)
    { event_manager_.push(tag, std::move(consumers)); }
    void drive_event_input(bool valid, std::uint32_t tag,
        std::vector<bool> targets)
    {
        if (targets.size() != caches_.size())
            throw StaticScheduleError("QCTL Event input mask width mismatch");
        pending_event_in_valid_ = valid;
        pending_event_in_tag_ = tag;
        pending_event_in_targets_ = std::move(targets);
        pending_event_input_preaccepted_ = false;
    }

    void prime_cycle(bool accept_current_pool_response = false)
    {
        if (distributed_mode_) {
            LoadInputs idle{};
            idle.load_mode = false;
            begin_distributed_load_cycle(idle);
            complete_distributed_load_cycle();
            return;
        }
        // All clocked owners observe the previous Current state at the same
        // edge.  Caches first consume grants/responses committed by the prior
        // Pool cycle; only then may the Pool publish this edge's new response.
        // Reversing this order would let a Cache consume a response in the
        // same edge that creates it and collapse one RTL pipeline stage.
        if (accept_current_pool_response) pool_.begin_cycle();
        for (std::size_t queue = 0; queue < caches_.size(); ++queue)
            if (configured_[queue]) caches_[queue].begin_cycle();
        if (!accept_current_pool_response) pool_.begin_cycle();
        clear_load_interface_view();
        sample_pool_interface();
        for (std::size_t queue = 0; queue < caches_.size(); ++queue)
            if (configured_[queue]) caches_[queue].end_cycle();
        pool_.end_cycle();
        sample_global_status_only();
        sample_global_aliases();
    }

    bool all_primed() const noexcept
    {
        if (distributed_mode_) {
            for (std::size_t queue = 0; queue < configured_.size(); ++queue)
                if (configured_[queue] && !distributed_store_.loaded(queue))
                    return false;
            return distributed_store_.load_complete();
        }
        for (std::size_t queue = 0; queue < caches_.size(); ++queue)
            if (configured_[queue] && !caches_[queue].primed()) return false;
        return true;
    }

    bool all_done() const noexcept
    {
        if (distributed_mode_) {
            for (std::size_t queue = 0; queue < configured_.size(); ++queue)
                if (configured_[queue]
                    && endpoint_retire_offset_[queue] != endpoint_count_[queue])
                    return false;
            return true;
        }
        for (std::size_t queue = 0; queue < caches_.size(); ++queue)
            if (configured_[queue] && !caches_[queue].done()) return false;
        return true;
    }

    const std::vector<QueueOutputs>& begin_cycle(
        const std::vector<std::uint32_t>& scheduler_cycles,
        const std::vector<bool>& run)
    {
        if (cycle_open_ || !barrier_configured_
            || scheduler_cycles.size() != caches_.size()
            || run.size() != caches_.size())
            throw StaticScheduleError("invalid QCTL Runtime begin-cycle inputs");
        if (distributed_mode_)
            return begin_distributed_runtime_cycle(scheduler_cycles, run);
        for (std::size_t queue = 0; queue < caches_.size(); ++queue)
            if (configured_[queue]) caches_[queue].begin_cycle();
        pool_.begin_cycle();
        clear_load_interface_view();

        pending_inputs_.assign(caches_.size(), {});
        pending_outputs_.assign(caches_.size(), {});
        pending_arrivals_.assign(caches_.size(), false);
        pending_notify_requests_.assign(caches_.size(), false);

        // First pass exposes EVENT_WAIT independently of the Event Buffer.
        std::vector<bool> event_wait(caches_.size(), false);
        for (std::size_t queue = 0; queue < caches_.size(); ++queue) {
            auto& input = pending_inputs_[queue];
            input.scheduler_cycle = scheduler_cycles[queue];
            input.head = configured_[queue] ? caches_[queue].head() : nullptr;
            if (run[queue] && caches_[queue].undefined_head())
                throw StaticScheduleError("cannot execute an unwritten QCTL Pool record");
            input.run = run[queue] && configured_[queue];
            input.configured = configured_[queue];
            input.primed = configured_[queue] && caches_[queue].primed();
            input.done = configured_[queue] && caches_[queue].done();
            const auto preview = controllers_[queue].evaluate(input);
            event_wait[queue] = preview.event_wait;
        }
        preaccept_pending_event_input();
        pending_event_selection_ = event_manager_.select(event_wait);

        for (std::size_t queue = 0; queue < caches_.size(); ++queue) {
            auto& input = pending_inputs_[queue];
            input.event_valid = pending_event_selection_.found
                && pending_event_selection_.visible[queue];
            input.event_tag = pending_event_selection_.found
                ? pending_event_selection_.tag : 0;
            const auto preview = controllers_[queue].evaluate(input);
            pending_arrivals_[queue] = preview.sync_arrival_valid;
            pending_notify_requests_[queue] = preview.notify_valid;
        }
        pending_barrier_outputs_ = barrier_.evaluate(
            pending_arrivals_, pending_notify_requests_);

        for (std::size_t queue = 0; queue < caches_.size(); ++queue) {
            auto& input = pending_inputs_[queue];
            input.sync_release = pending_barrier_outputs_.release[queue];
            input.notify_ready = pending_barrier_outputs_.notify_ready[queue];
            const auto raw = controllers_[queue].evaluate(input);
            static_cast<IcuQctlControllerOutputs&>(pending_outputs_[queue]) = raw;
            sample_queue_output(queue, pending_outputs_[queue]);
        }
        sample_hardware_view(event_wait);
        sample_pool_interface();
        sample_global_aliases();
        cycle_open_ = true;
        return pending_outputs_;
    }

    std::vector<QueueOutputs> complete_cycle(
        const std::vector<bool>& mem_packet_retired)
    {
        if (!cycle_open_ || mem_packet_retired.size() != caches_.size())
            throw StaticScheduleError("invalid QCTL Runtime complete-cycle inputs");
        if (distributed_mode_)
            return complete_distributed_runtime_cycle(mem_packet_retired);
        set_mem_packet_retired_inputs(mem_packet_retired);
        std::vector<bool> event_fires(caches_.size(), false);
        for (std::size_t queue = 0; queue < caches_.size(); ++queue) {
            auto& input = pending_inputs_[queue];
            const auto raw = controllers_[queue].evaluate(input);
            static_cast<IcuQctlControllerOutputs&>(pending_outputs_[queue]) = raw;
            if (configured_[queue] && raw.retire_head)
                caches_[queue].retire();
            controllers_[queue].commit(input, raw);
            event_fires[queue] = raw.event_fire;
            sample_queue_output(queue, pending_outputs_[queue]);
        }

        event_manager_.commit(pending_event_selection_, event_fires);
        if (pending_event_in_valid_ && !pending_event_input_preaccepted_
            && hardware_view_.event_buffer.event_in_ready_o)
            event_manager_.push(pending_event_in_tag_,
                pending_event_in_targets_);
        barrier_.commit(pending_arrivals_, pending_barrier_outputs_);
        for (std::size_t queue = 0; queue < caches_.size(); ++queue)
            if (configured_[queue]) caches_[queue].end_cycle();
        pool_.end_cycle();
        hardware_view_.event_buffer.event_consume_i = event_fires;
        hardware_view_.fabric.endpoint_event_consume_i = event_fires;
        hardware_view_.fabric.event_consume_o = event_fires;
        pending_event_in_valid_ = false;
        pending_event_input_preaccepted_ = false;
        cycle_open_ = false;
        return pending_outputs_;
    }

    const std::vector<QueueOutputs>& set_mem_packet_retired_inputs(
        const std::vector<bool>& mem_packet_retired)
    {
        if (!cycle_open_ || mem_packet_retired.size() != caches_.size())
            throw StaticScheduleError("invalid QCTL MEM_PACKET completion inputs");
        for (std::size_t queue = 0; queue < caches_.size(); ++queue) {
            auto& input = pending_inputs_[queue];
            input.mem_packet_retired = mem_packet_retired[queue];
            const auto raw = controllers_[queue].evaluate(input);
            static_cast<IcuQctlControllerOutputs&>(pending_outputs_[queue]) = raw;
            auto& controller = hardware_view_.controllers[queue];
            auto& endpoint = hardware_view_.endpoints[queue];
            controller.mem_packet_retired_i = mem_packet_retired[queue];
            controller.head_retire_o = raw.retire_head;
            controller.mem_packet_valid_o = raw.mem_packet_valid;
            controller.mem_packet_ordinal_o = raw.mem_packet_ordinal;
            endpoint.mem_packet_retired_i = mem_packet_retired[queue];
            endpoint.mem_packet_valid_o = raw.mem_packet_valid;
            endpoint.mem_packet_ordinal_o = raw.mem_packet_ordinal;
            hardware_view_.fabric.mem_packet_retired_i[queue] =
                mem_packet_retired[queue];
            hardware_view_.fabric.endpoint_mem_packet_retired_o[queue] =
                mem_packet_retired[queue];
        }
        sample_global_aliases();
        return pending_outputs_;
    }

    std::vector<QueueOutputs> tick(
        const std::vector<std::uint32_t>& scheduler_cycles,
        const std::vector<bool>& mem_packet_retired)
    {
        begin_cycle(scheduler_cycles,
            std::vector<bool>(scheduler_cycles.size(), true));
        return complete_cycle(mem_packet_retired);
    }

    const IcuQctlPool& pool() const noexcept { return pool_; }
    const IcuQctlEventManager& event_manager() const noexcept
    { return event_manager_; }
    const IcuQctlGlobalInterfaceView& hardware_interface_view() const noexcept
    { return hardware_view_; }
    static std::uint8_t encode_global_fault_code(bool pool_fault,
        std::uint8_t pool_code, bool event_fault, std::uint8_t event_code,
        bool fabric_fault, std::uint8_t fabric_code,
        const std::vector<bool>& local_fault,
        const std::vector<std::uint8_t>& local_code)
    {
        if (local_fault.size() != local_code.size())
            throw StaticScheduleError("QCTL local fault vector width mismatch");
        if (pool_fault) return pool_code;
        if (event_fault) return static_cast<std::uint8_t>(0x20 | event_code);
        if (fabric_fault) return static_cast<std::uint8_t>(0x30 | fabric_code);
        for (std::size_t queue = 0; queue < local_fault.size(); ++queue)
            if (local_fault[queue])
                return static_cast<std::uint8_t>(0x10 | local_code[queue]);
        return 0;
    }
    bool enabled() const noexcept
    { return std::ranges::any_of(configured_, [](bool value) { return value; }); }
    std::optional<IcuQueueControl> committed_program_record(std::size_t address) const
    { return distributed_mode_ ? distributed_store_.committed_record(address)
                               : pool_.committed_record(address); }
    bool undefined_program_read_response() const noexcept
    { return !distributed_mode_ && pool_.last_response_valid()
        && !pool_.last_response().initialized; }
    std::optional<IcuQctlPool::Partition> committed_program_partition(std::size_t queue) const
    {
        if (!queue_configured(queue)) return std::nullopt;
        if (distributed_mode_) {
            const auto value = distributed_store_.partition(queue);
            if (!value) return std::nullopt;
            return IcuQctlPool::Partition{value->base, value->count,
                value->generation};
        }
        return pool_.partition(queue);
    }
    bool queue_configured(std::size_t queue) const
    { check_queue(queue); return configured_[queue]; }
    bool queue_primed(std::size_t queue) const
    { check_queue(queue); return configured_[queue]
        && (distributed_mode_ ? distributed_store_.loaded(queue)
                              : caches_[queue].primed()); }
    bool queue_done(std::size_t queue) const
    { check_queue(queue); return configured_[queue]
        && (distributed_mode_
            ? endpoint_retire_offset_[queue] == endpoint_count_[queue]
            : caches_[queue].done()); }
    QueueOutputs queue_status(std::size_t queue) const
    {
        check_queue(queue);
        QueueOutputs output{};
        output.state = controllers_[queue].state();
        if (!configured_[queue]) return output;
        if (distributed_mode_) {
            output.fetch_offset = endpoint_retire_offset_[queue];
            output.retire_offset = endpoint_retire_offset_[queue];
            output.cache_level = distributed_store_.loaded(queue)
                && endpoint_retire_offset_[queue] < endpoint_count_[queue] ? 1 : 0;
            output.primed = distributed_store_.loaded(queue);
            output.done = endpoint_retire_offset_[queue] == endpoint_count_[queue];
            output.fault = endpoint_fault_[queue]
                || controllers_[queue].fault();
            output.fault_code = endpoint_fault_[queue]
                ? endpoint_fault_code_[queue] : controllers_[queue].fault_code();
            return output;
        }
        output.fetch_offset = caches_[queue].fetch_offset();
        output.retire_offset = caches_[queue].retire_offset();
        output.in_flight_reads = caches_[queue].in_flight_reads();
        output.cache_level = caches_[queue].level();
        output.primed = caches_[queue].primed();
        output.done = caches_[queue].done();
        output.fault = caches_[queue].fault() || controllers_[queue].fault();
        output.fault_code = caches_[queue].fault()
            ? caches_[queue].fault_code() : controllers_[queue].fault_code();
        return output;
    }

    QueueOutputs queue_cycle_status(std::size_t queue) const
    {
        check_queue(queue);
        if (pending_outputs_.size() != configured_.size())
            return queue_status(queue);
        return pending_outputs_[queue];
    }

private:
    static IcuQctlDistributedProgramStore::Inputs distributed_inputs(
        const LoadInputs& inputs)
    {
        IcuQctlDistributedProgramStore::Inputs result{};
        result.clear = inputs.clear;
        result.load_mode = inputs.load_mode;
        result.partition_valid = inputs.partition_valid;
        result.partition_queue = inputs.partition_queue;
        result.partition_base = inputs.partition_base;
        result.partition_count = inputs.partition_count;
        result.partition_generation = inputs.partition_generation;
        result.load_valid = inputs.load_valid;
        result.load_address = inputs.load_address;
        result.load_data = inputs.load_data;
        return result;
    }

    void begin_distributed_load_cycle(const LoadInputs& inputs)
    {
        pending_load_inputs_ = inputs;
        const auto& preview = distributed_store_.begin_cycle(
            distributed_inputs(inputs));
        load_cycle_open_ = true;
        load_clocks_read_pipeline_ = false;
        clear_distributed_cycle_view();
        sample_distributed_load_path(inputs, preview);
        apply_distributed_configure_view(preview);
        sample_distributed_endpoint_status();
        sample_global_aliases();
    }

    void complete_distributed_load_cycle()
    {
        const auto inputs = pending_load_inputs_;
        const auto preview = distributed_store_.preview();
        distributed_store_.complete_cycle();
        if (inputs.clear) {
            for (auto& controller : controllers_) controller.reset();
            std::fill(configured_.begin(), configured_.end(), false);
            std::fill(endpoint_count_.begin(), endpoint_count_.end(), 0);
            std::fill(endpoint_generation_.begin(), endpoint_generation_.end(), 0);
            std::fill(endpoint_retire_offset_.begin(), endpoint_retire_offset_.end(), 0);
            std::fill(endpoint_fault_.begin(), endpoint_fault_.end(), false);
            std::fill(endpoint_fault_code_.begin(), endpoint_fault_code_.end(), 0);
            barrier_.reset();
            event_manager_.reset();
        } else {
            commit_distributed_configure(preview);
        }
        load_cycle_open_ = false;
    }

    const std::vector<QueueOutputs>& begin_distributed_runtime_cycle(
        const std::vector<std::uint32_t>& scheduler_cycles,
        const std::vector<bool>& run)
    {
        LoadInputs idle{};
        idle.load_mode = false;
        const auto& load_preview = distributed_store_.begin_cycle(
            distributed_inputs(idle));
        clear_distributed_cycle_view();
        sample_distributed_load_path(idle, load_preview);
        apply_distributed_configure_view(load_preview);

        pending_inputs_.assign(configured_.size(), {});
        pending_outputs_.assign(configured_.size(), {});
        pending_arrivals_.assign(configured_.size(), false);
        pending_notify_requests_.assign(configured_.size(), false);
        pending_heads_.assign(configured_.size(), std::nullopt);
        pending_head_error_.assign(configured_.size(), false);

        std::vector<bool> event_wait(configured_.size(), false);
        for (std::size_t queue = 0; queue < configured_.size(); ++queue) {
            const auto head = distributed_store_.head(queue,
                endpoint_retire_offset_[queue], endpoint_generation_[queue]);
            if (head.valid) pending_heads_[queue] = head.record;
            pending_head_error_[queue] = head.error;
            auto& input = pending_inputs_[queue];
            input.scheduler_cycle = scheduler_cycles[queue];
            input.head = pending_heads_[queue] ? &*pending_heads_[queue] : nullptr;
            input.run = run[queue] && configured_[queue];
            input.configured = configured_[queue];
            input.primed = configured_[queue] && distributed_store_.loaded(queue);
            input.done = configured_[queue]
                && endpoint_retire_offset_[queue] == endpoint_count_[queue];
            const auto preview = controllers_[queue].evaluate(input);
            event_wait[queue] = preview.event_wait;
        }
        preaccept_pending_event_input();
        pending_event_selection_ = event_manager_.select(event_wait);

        for (std::size_t queue = 0; queue < configured_.size(); ++queue) {
            auto& input = pending_inputs_[queue];
            input.event_valid = pending_event_selection_.found
                && pending_event_selection_.visible[queue];
            input.event_tag = pending_event_selection_.found
                ? pending_event_selection_.tag : 0;
            const auto preview = controllers_[queue].evaluate(input);
            pending_arrivals_[queue] = preview.sync_arrival_valid;
            pending_notify_requests_[queue] = preview.notify_valid;
        }
        pending_barrier_outputs_ = barrier_.evaluate(
            pending_arrivals_, pending_notify_requests_);
        for (std::size_t queue = 0; queue < configured_.size(); ++queue) {
            auto& input = pending_inputs_[queue];
            input.sync_release = pending_barrier_outputs_.release[queue];
            input.notify_ready = pending_barrier_outputs_.notify_ready[queue];
            const auto raw = controllers_[queue].evaluate(input);
            static_cast<IcuQctlControllerOutputs&>(pending_outputs_[queue]) = raw;
            if (endpoint_fault_[queue]) {
                pending_outputs_[queue].fault = true;
                pending_outputs_[queue].fault_code = endpoint_fault_code_[queue];
            }
            sample_queue_output(queue, pending_outputs_[queue]);
        }
        sample_distributed_hardware_view(event_wait);
        cycle_open_ = true;
        return pending_outputs_;
    }

    std::vector<QueueOutputs> complete_distributed_runtime_cycle(
        const std::vector<bool>& mem_packet_retired)
    {
        set_mem_packet_retired_inputs(mem_packet_retired);
        std::vector<bool> event_fires(configured_.size(), false);
        for (std::size_t queue = 0; queue < configured_.size(); ++queue) {
            auto& input = pending_inputs_[queue];
            const auto raw = controllers_[queue].evaluate(input);
            static_cast<IcuQctlControllerOutputs&>(pending_outputs_[queue]) = raw;
            const bool head_valid = pending_heads_[queue].has_value()
                && !pending_head_error_[queue] && !endpoint_fault_[queue];
            const bool done = configured_[queue]
                && endpoint_retire_offset_[queue] == endpoint_count_[queue];
            if (!endpoint_fault_[queue] && input.run && configured_[queue]
                && !done && pending_head_error_[queue]) {
                endpoint_fault_[queue] = true;
                endpoint_fault_code_[queue] = 2;
            } else if (!endpoint_fault_[queue] && raw.retire_head
                && !head_valid) {
                endpoint_fault_[queue] = true;
                endpoint_fault_code_[queue] = 3;
            }
            if (configured_[queue] && raw.retire_head && head_valid)
                ++endpoint_retire_offset_[queue];
            controllers_[queue].commit(input, raw);
            event_fires[queue] = raw.event_fire;
            sample_queue_output(queue, pending_outputs_[queue]);
        }
        event_manager_.commit(pending_event_selection_, event_fires);
        if (pending_event_in_valid_ && !pending_event_input_preaccepted_
            && hardware_view_.event_buffer.event_in_ready_o)
            event_manager_.push(pending_event_in_tag_, pending_event_in_targets_);
        barrier_.commit(pending_arrivals_, pending_barrier_outputs_);
        const auto load_preview = distributed_store_.preview();
        distributed_store_.complete_cycle();
        commit_distributed_configure(load_preview);
        refresh_mem_packet_retire_postedge(mem_packet_retired);
        hardware_view_.event_buffer.event_consume_i = event_fires;
        hardware_view_.fabric.endpoint_event_consume_i = event_fires;
        hardware_view_.fabric.event_consume_o = event_fires;
        pending_event_in_valid_ = false;
        pending_event_input_preaccepted_ = false;
        cycle_open_ = false;
        return pending_outputs_;
    }

    // The RTL Event Buffer captures a ready input on the active edge and its
    // newly registered entry is immediately visible to the post-edge
    // combinational selection logic.  The CModel LOW/POSEDGE transaction must
    // therefore make a non-backpressured input available before Controller
    // evaluation for this tick.  A full Buffer is left for complete_cycle(),
    // where simultaneous retirement can create the required append slot.
    void preaccept_pending_event_input()
    {
        pending_event_input_preaccepted_ = false;
        if (!pending_event_in_valid_
            || event_manager_.level() >= event_manager_.capacity())
            return;
        event_manager_.push(pending_event_in_tag_, pending_event_in_targets_);
        pending_event_input_preaccepted_ = true;
    }

    // MEM_PACKET retirement is a direct Endpoint input, not a buffered Event.
    // When it fires on the active edge, RTL exposes the retired head and the
    // Controller's new IDLE/CE outputs immediately in POST_POSEDGE.  Refresh
    // only the affected Endpoint views and retained cycle outputs; no second
    // state transition is performed.
    void refresh_mem_packet_retire_postedge(
        const std::vector<bool>& mem_packet_retired)
    {
        bool any = false;
        for (std::size_t queue = 0; queue < configured_.size(); ++queue) {
            if (!mem_packet_retired[queue]) continue;
            any = true;
            const auto head = distributed_store_.head(queue,
                endpoint_retire_offset_[queue], endpoint_generation_[queue]);
            pending_heads_[queue] = head.valid
                ? std::optional<IcuQueueControl>{head.record} : std::nullopt;
            pending_head_error_[queue] = head.error;
            auto& input = pending_inputs_[queue];
            input.head = pending_heads_[queue] ? &*pending_heads_[queue] : nullptr;
            input.done = configured_[queue]
                && endpoint_retire_offset_[queue] == endpoint_count_[queue];
            const auto raw = controllers_[queue].evaluate(input);
            static_cast<IcuQctlControllerOutputs&>(pending_outputs_[queue]) = raw;
            sample_queue_output(queue, pending_outputs_[queue]);
        }
        if (!any) return;

        sample_distributed_endpoint_status();
        for (std::size_t queue = 0; queue < configured_.size(); ++queue) {
            if (!mem_packet_retired[queue]) continue;
            const auto& input = pending_inputs_[queue];
            const auto& output = pending_outputs_[queue];
            auto& controller = hardware_view_.controllers[queue];
            auto& endpoint = hardware_view_.endpoints[queue];
            controller.done_i = input.done;
            controller.scheduler_counter_ce_o = output.scheduler_counter_ce;
            controller.head_valid_i = pending_heads_[queue].has_value();
            controller.head_record_i = pending_heads_[queue]
                ? *pending_heads_[queue] : qctl_zero_record();
            controller.head_retire_o = output.retire_head;
            controller.mem_packet_valid_o = output.mem_packet_valid;
            controller.mem_packet_ordinal_o = output.mem_packet_ordinal;
            controller.mem_packet_retired_i = true;
            endpoint.scheduler_counter_ce_o = output.scheduler_counter_ce;
            endpoint.mem_packet_valid_o = output.mem_packet_valid;
            endpoint.mem_packet_ordinal_o = output.mem_packet_ordinal;
            endpoint.mem_packet_retired_i = true;
            hardware_view_.fabric.endpoint_mem_packet_valid_i[queue] =
                output.mem_packet_valid;
            hardware_view_.fabric.endpoint_mem_packet_ordinal_i[queue] =
                output.mem_packet_ordinal;
            hardware_view_.fabric.mem_packet_valid_o[queue] =
                output.mem_packet_valid;
            hardware_view_.fabric.mem_packet_ordinal_o[queue] =
                output.mem_packet_ordinal;
            hardware_view_.fabric.mem_packet_retired_i[queue] = true;
            hardware_view_.fabric.endpoint_mem_packet_retired_o[queue] = true;
        }
        sample_global_aliases();
    }

    void commit_distributed_configure(
        const IcuQctlDistributedProgramStore::Preview& preview)
    {
        for (std::size_t queue = 0; queue < configured_.size(); ++queue) {
            if (!preview.configure_valid[queue]) continue;
            configured_[queue] = true;
            endpoint_count_[queue] = preview.configure_count[queue];
            endpoint_generation_[queue] = preview.configure_generation[queue];
            endpoint_retire_offset_[queue] = 0;
            endpoint_fault_[queue] = false;
            endpoint_fault_code_[queue] = 0;
            controllers_[queue].reset();
        }
    }

    void clear_distributed_cycle_view()
    {
        clear_load_interface_view();
        std::fill(hardware_view_.endpoint_store_loaded_o.begin(),
            hardware_view_.endpoint_store_loaded_o.end(), false);
        std::fill(hardware_view_.endpoint_head_valid_o.begin(),
            hardware_view_.endpoint_head_valid_o.end(), false);
        std::fill(hardware_view_.endpoint_head_record_o.begin(),
            hardware_view_.endpoint_head_record_o.end(), qctl_zero_record());
        std::fill(hardware_view_.endpoint_head_error_o.begin(),
            hardware_view_.endpoint_head_error_o.end(), false);
    }

    void sample_distributed_load_path(const LoadInputs& inputs,
        const IcuQctlDistributedProgramStore::Preview& preview)
    {
        auto& manager = hardware_view_.load_manager;
        manager = {};
        manager.rst_ni = true;
        manager.clear_i = inputs.clear;
        manager.load_mode_i = inputs.load_mode;
        manager.partition_valid_i = inputs.partition_valid;
        manager.partition_ready_o = preview.partition_ready;
        manager.partition_queue_i = static_cast<std::uint32_t>(inputs.partition_queue);
        manager.partition_base_i = static_cast<std::uint32_t>(inputs.partition_base);
        manager.partition_count_i = static_cast<std::uint32_t>(inputs.partition_count);
        manager.partition_generation_i = inputs.partition_generation;
        manager.load_valid_i = inputs.load_valid;
        manager.load_ready_o = preview.load_ready;
        manager.load_address_i = static_cast<std::uint32_t>(inputs.load_address);
        manager.load_data_i = inputs.load_data;
        manager.relay_packet_valid_o = preview.relay_packet_valid;
        manager.relay_packet_ready_i = preview.relay_packet_ready;
        manager.relay_packet_kind_o = preview.relay_packet.kind
            == IcuQctlDistributedProgramStore::PacketKind::Record;
        manager.relay_packet_queue_o = static_cast<std::uint32_t>(
            preview.relay_packet.queue);
        manager.relay_packet_offset_o = static_cast<std::uint32_t>(
            preview.relay_packet.offset);
        manager.relay_packet_count_o = static_cast<std::uint32_t>(
            preview.relay_packet.count);
        manager.relay_packet_generation_o = preview.relay_packet.generation;
        manager.relay_packet_record_o = preview.relay_packet.record;
        manager.idle_o = preview.manager_idle;
        manager.fault_o = preview.manager_fault;
        manager.fault_code_o = preview.manager_fault_code;

        for (std::size_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
            for (std::size_t relay = 0; relay < 7; ++relay) {
                const auto index = hemisphere * 7 + relay;
                const auto& signals = preview.relays[hemisphere][relay];
                auto& view = hardware_view_.relays[index];
                view.chain_packet_valid_i = signals.input_valid;
                view.chain_packet_ready_o = signals.input_ready;
                view.chain_packet_fire = signals.input_fire;
                view.local_fire = signals.local_fire;
                view.next_packet_valid_o = signals.output_valid;
                view.next_packet_ready_i = signals.output_ready;
                view.fifo_level = static_cast<std::uint32_t>(signals.fifo_level);
                view.idle_o = signals.fifo_level == 0 && !signals.input_valid;
                view.fault_o = signals.fault;
                view.fault_code_o = signals.fault_code;
            }
        }

        auto& legacy = hardware_view_.pool;
        legacy.rst_ni = true;
        legacy.clear_i = inputs.clear;
        legacy.load_mode_i = inputs.load_mode;
        legacy.partition_valid_i = inputs.partition_valid;
        legacy.partition_ready_o = preview.partition_ready;
        legacy.partition_queue_i = manager.partition_queue_i;
        legacy.partition_base_i = manager.partition_base_i;
        legacy.partition_count_i = manager.partition_count_i;
        legacy.partition_generation_i = manager.partition_generation_i;
        legacy.load_valid_i = inputs.load_valid;
        legacy.load_ready_o = preview.load_ready;
        legacy.load_address_i = manager.load_address_i;
        legacy.load_data_i = inputs.load_data;
        legacy.request_valid_i.assign(configured_.size(), false);
        legacy.request_ready_o.assign(configured_.size(), false);
        legacy.request_index_i.assign(configured_.size(), 0);
        legacy.request_generation_i.assign(configured_.size(), 0);
        legacy.response_valid_o = false;
        legacy.response_error_o = false;
        legacy.fault_o = preview.fault;
        legacy.fault_code_o = preview.fault_code;
        hardware_view_.load_complete_o = preview.load_complete;
    }

    void apply_distributed_configure_view(
        const IcuQctlDistributedProgramStore::Preview& preview)
    {
        hardware_view_.endpoint_configure_valid_o = preview.configure_valid;
        hardware_view_.endpoint_configure_count_o = preview.configure_count;
        hardware_view_.endpoint_configure_generation_o =
            preview.configure_generation;
        for (std::size_t queue = 0; queue < configured_.size(); ++queue) {
            auto& endpoint = hardware_view_.endpoints[queue];
            auto& controller = hardware_view_.controllers[queue];
            endpoint.configure_valid_i = preview.configure_valid[queue];
            endpoint.configure_count_i = preview.configure_count[queue];
            endpoint.configure_generation_i = preview.configure_generation[queue];
            controller.configure_valid_i = preview.configure_valid[queue];
        }
    }

    void sample_distributed_endpoint_status()
    {
        hardware_view_.all_primed_o = hardware_view_.load_complete_o;
        hardware_view_.all_done_o = true;
        bool local_fault = false;
        std::uint8_t local_code = 0;
        for (std::size_t queue = 0; queue < configured_.size(); ++queue) {
            const auto head = distributed_store_.head(queue,
                endpoint_retire_offset_[queue], endpoint_generation_[queue]);
            auto& endpoint = hardware_view_.endpoints[queue];
            endpoint.store_head_index_o = static_cast<std::uint32_t>(
                endpoint_retire_offset_[queue]);
            endpoint.store_head_generation_o = endpoint_generation_[queue];
            endpoint.store_loaded_i = configured_[queue]
                && distributed_store_.loaded(queue);
            endpoint.store_head_valid_i = head.valid;
            endpoint.store_head_record_i = head.valid ? head.record
                                                       : qctl_zero_record();
            endpoint.store_head_error_i = head.error;
            endpoint.configured_o = configured_[queue];
            endpoint.primed_o = configured_[queue]
                && distributed_store_.loaded(queue);
            endpoint.done_o = configured_[queue]
                && endpoint_retire_offset_[queue] == endpoint_count_[queue];
            endpoint.fault_o = endpoint_fault_[queue]
                || controllers_[queue].fault();
            endpoint.fault_code_o = endpoint_fault_[queue]
                ? endpoint_fault_code_[queue] : controllers_[queue].fault_code();
            hardware_view_.endpoint_store_loaded_o[queue] =
                endpoint.store_loaded_i;
            hardware_view_.endpoint_head_valid_o[queue] =
                endpoint.store_head_valid_i;
            hardware_view_.endpoint_head_record_o[queue] =
                endpoint.store_head_record_i;
            hardware_view_.endpoint_head_error_o[queue] =
                endpoint.store_head_error_i;
            hardware_view_.endpoint_configured_i[queue] = endpoint.configured_o;
            hardware_view_.endpoint_primed_i[queue] = endpoint.primed_o;
            hardware_view_.endpoint_done_i[queue] = endpoint.done_o;
            hardware_view_.endpoint_fault_i[queue] = endpoint.fault_o;
            hardware_view_.endpoint_fault_code_i[queue] = endpoint.fault_code_o;
            hardware_view_.queue_active_o[queue] = configured_[queue];
            hardware_view_.all_primed_o = hardware_view_.all_primed_o
                && (!configured_[queue] || endpoint.primed_o);
            hardware_view_.all_done_o = hardware_view_.all_done_o
                && (!configured_[queue] || endpoint.done_o);
            if (!local_fault && endpoint.fault_o) {
                local_fault = true;
                local_code = endpoint.fault_code_o;
            }
        }
        hardware_view_.fault_o = distributed_store_.fault()
            || event_manager_.fault() || barrier_.fault() || local_fault;
        hardware_view_.fault_code_o = distributed_store_.fault()
            ? distributed_store_.fault_code()
            : (event_manager_.fault()
                ? static_cast<std::uint8_t>(0x20 | event_manager_.fault_code())
                : (barrier_.fault()
                    ? static_cast<std::uint8_t>(0x30 | barrier_.fault_code())
                    : (local_fault
                        ? static_cast<std::uint8_t>(0x60 | local_code) : 0)));
    }

    void sample_distributed_hardware_view(
        const std::vector<bool>& event_wait)
    {
        const auto count = configured_.size();
        hardware_view_.fabric.sync_participant_mask_i = barrier_.participants();
        hardware_view_.fabric.endpoint_sync_arrive_valid_i = pending_arrivals_;
        hardware_view_.fabric.endpoint_notify_valid_i = pending_notify_requests_;
        hardware_view_.fabric.endpoint_notify_ready_o =
            pending_barrier_outputs_.notify_ready;
        hardware_view_.fabric.endpoint_sync_release_valid_o =
            pending_barrier_outputs_.release;
        std::vector<bool> event_fires(count, false);
        for (std::size_t queue = 0; queue < count; ++queue)
            event_fires[queue] = pending_outputs_[queue].event_fire;
        hardware_view_.event_buffer.event_wait_i = event_wait;
        hardware_view_.event_buffer.event_consume_i = event_fires;
        hardware_view_.event_buffer.event_valid_o =
            pending_event_selection_.found ? pending_event_selection_.visible
                                           : std::vector<bool>(count, false);
        hardware_view_.event_buffer.event_tag_o = pending_event_selection_.found
            ? pending_event_selection_.tag : 0;
        hardware_view_.event_buffer.event_in_valid_i = pending_event_in_valid_;
        hardware_view_.event_buffer.event_in_tag_i = pending_event_in_tag_;
        hardware_view_.event_buffer.event_in_target_mask_i =
            pending_event_in_targets_.empty()
                ? std::vector<bool>(count, false) : pending_event_in_targets_;
        hardware_view_.event_buffer.event_in_ready_o =
            event_manager_.level() < event_manager_.capacity()
            || event_manager_.selection_will_retire(
                pending_event_selection_, event_fires);
        hardware_view_.event_buffer.fault_o = event_manager_.fault();
        hardware_view_.event_buffer.fault_code_o = event_manager_.fault_code();
        auto& fabric = hardware_view_.fabric;
        fabric.endpoint_event_wait_i = event_wait;
        fabric.endpoint_event_consume_i = event_fires;
        fabric.event_wait_o = event_wait;
        fabric.event_consume_o = event_fires;
        fabric.event_buffer_valid_i = hardware_view_.event_buffer.event_valid_o;
        fabric.event_buffer_tag_i = hardware_view_.event_buffer.event_tag_o;
        fabric.event_in_valid_i = pending_event_in_valid_;
        fabric.event_in_ready_i = hardware_view_.event_buffer.event_in_ready_o;
        fabric.event_backpressure_o = pending_event_in_valid_
            && !hardware_view_.event_buffer.event_in_ready_o;
        fabric.endpoint_event_valid_o = hardware_view_.event_buffer.event_valid_o;
        fabric.endpoint_event_tag_o = hardware_view_.event_buffer.event_tag_o;
        fabric.fault_o = barrier_.fault();
        fabric.fault_code_o = barrier_.fault_code();

        sample_distributed_endpoint_status();
        hardware_view_.all_primed_o = hardware_view_.load_complete_o;
        hardware_view_.all_done_o = true;
        bool local_fault = false;
        std::uint8_t local_code = 0;
        for (std::size_t queue = 0; queue < count; ++queue) {
            const auto& input = pending_inputs_[queue];
            const auto& output = pending_outputs_[queue];
            auto& controller = hardware_view_.controllers[queue];
            auto& endpoint = hardware_view_.endpoints[queue];
            controller.run_i = input.run;
            controller.configured_i = input.configured;
            controller.primed_i = input.primed;
            controller.done_i = input.done;
            controller.scheduler_cycle_i = input.scheduler_cycle;
            controller.scheduler_counter_ce_o = output.scheduler_counter_ce;
            controller.head_valid_i = pending_heads_[queue].has_value();
            controller.head_record_i = pending_heads_[queue]
                ? *pending_heads_[queue] : qctl_zero_record();
            controller.head_retire_o = output.retire_head;
            controller.sync_arrive_valid_o = output.sync_arrival_valid;
            controller.sync_release_valid_i = input.sync_release;
            controller.notify_valid_o = output.notify_valid;
            controller.notify_ready_i = input.notify_ready;
            controller.event_valid_i = input.event_valid;
            controller.event_tag_i = input.event_tag;
            controller.event_wait_o = output.event_wait;
            controller.event_consume_o = output.event_fire;
            controller.mem_packet_valid_o = output.mem_packet_valid;
            controller.mem_packet_ordinal_o = output.mem_packet_ordinal;
            controller.mem_packet_retired_i = input.mem_packet_retired;
            controller.fault_o = controllers_[queue].fault();
            controller.fault_code_o = controllers_[queue].fault_code();
            endpoint.run_i = input.run;
            endpoint.scheduler_cycle_i = input.scheduler_cycle;
            endpoint.scheduler_counter_ce_o = output.scheduler_counter_ce;
            endpoint.event_valid_i = input.event_valid;
            endpoint.event_tag_i = input.event_tag;
            endpoint.event_wait_o = output.event_wait;
            endpoint.event_consume_o = output.event_fire;
            endpoint.sync_arrive_valid_o = output.sync_arrival_valid;
            endpoint.sync_release_valid_i = input.sync_release;
            endpoint.notify_valid_o = output.notify_valid;
            endpoint.notify_ready_i = input.notify_ready;
            endpoint.mem_packet_valid_o = output.mem_packet_valid;
            endpoint.mem_packet_ordinal_o = output.mem_packet_ordinal;
            endpoint.mem_packet_retired_i = input.mem_packet_retired;
            hardware_view_.all_primed_o = hardware_view_.all_primed_o
                && (!configured_[queue] || endpoint.primed_o);
            hardware_view_.all_done_o = hardware_view_.all_done_o
                && (!configured_[queue] || endpoint.done_o);
            fabric.endpoint_mem_packet_valid_i[queue] = output.mem_packet_valid;
            fabric.endpoint_mem_packet_ordinal_i[queue] =
                output.mem_packet_ordinal;
            fabric.mem_packet_valid_o[queue] = output.mem_packet_valid;
            fabric.mem_packet_ordinal_o[queue] = output.mem_packet_ordinal;
            if (!local_fault && endpoint.fault_o) {
                local_fault = true;
                local_code = endpoint.fault_code_o;
            }
        }
        hardware_view_.fault_o = distributed_store_.fault()
            || event_manager_.fault() || barrier_.fault() || local_fault;
        hardware_view_.fault_code_o = distributed_store_.fault()
            ? distributed_store_.fault_code()
            : (event_manager_.fault()
                ? static_cast<std::uint8_t>(0x20 | event_manager_.fault_code())
                : (barrier_.fault()
                    ? static_cast<std::uint8_t>(0x30 | barrier_.fault_code())
                    : (local_fault
                        ? static_cast<std::uint8_t>(0x60 | local_code) : 0)));
        sample_global_aliases();
    }

    void initialize_hardware_view(std::size_t queue_count)
    {
        hardware_view_ = {};
        hardware_view_.caches.resize(queue_count);
        hardware_view_.controllers.resize(queue_count);
        hardware_view_.endpoints.resize(queue_count);
        hardware_view_.relays.resize(14);
        hardware_view_.endpoint_configure_valid_o.assign(queue_count, false);
        hardware_view_.endpoint_configure_count_o.assign(queue_count, 0);
        hardware_view_.endpoint_configure_generation_o.assign(queue_count, 0);
        hardware_view_.endpoint_store_loaded_o.assign(queue_count, false);
        hardware_view_.endpoint_head_valid_o.assign(queue_count, false);
        hardware_view_.endpoint_head_record_o.assign(queue_count,
            qctl_zero_record());
        hardware_view_.endpoint_head_error_o.assign(queue_count, false);
        hardware_view_.endpoint_configured_i.assign(queue_count, false);
        hardware_view_.endpoint_primed_i.assign(queue_count, false);
        hardware_view_.endpoint_done_i.assign(queue_count, false);
        hardware_view_.endpoint_fault_i.assign(queue_count, false);
        hardware_view_.endpoint_fault_code_i.assign(queue_count, 0);
        hardware_view_.queue_active_o.assign(queue_count, false);
        auto bools = [queue_count] { return std::vector<bool>(queue_count, false); };
        auto u32s = [queue_count] { return std::vector<std::uint32_t>(queue_count, 0); };
        auto u64s = [queue_count] { return std::vector<std::uint64_t>(queue_count, 0); };
        hardware_view_.pool.request_valid_i = bools();
        hardware_view_.pool.request_ready_o = bools();
        hardware_view_.pool.request_index_i = u32s();
        hardware_view_.pool.request_generation_i = u64s();
        hardware_view_.event_buffer.event_in_target_mask_i = bools();
        hardware_view_.event_buffer.event_wait_i = bools();
        hardware_view_.event_buffer.event_consume_i = bools();
        hardware_view_.event_buffer.event_valid_o = bools();
        hardware_view_.event_in_target_mask_i = bools();
        hardware_view_.endpoint_event_wait_i = bools();
        hardware_view_.endpoint_event_consume_i = bools();
        hardware_view_.endpoint_event_valid_o = bools();
        hardware_view_.sync_participant_mask_i = bools();
        hardware_view_.endpoint_sync_arrive_valid_i = bools();
        hardware_view_.endpoint_notify_valid_i = bools();
        hardware_view_.endpoint_notify_ready_o = bools();
        hardware_view_.endpoint_sync_release_valid_o = bools();
        hardware_view_.endpoint_mem_packet_valid_i = bools();
        hardware_view_.endpoint_mem_packet_ordinal_i = u32s();
        hardware_view_.mem_packet_valid_o = bools();
        hardware_view_.mem_packet_ordinal_o = u32s();
        hardware_view_.mem_packet_retired_i = bools();
        hardware_view_.endpoint_mem_packet_retired_o = bools();
        auto& fabric = hardware_view_.fabric;
        fabric.endpoint_pool_request_valid_i = bools();
        fabric.endpoint_pool_request_ready_o = bools();
        fabric.endpoint_pool_request_index_i = u32s();
        fabric.endpoint_pool_request_generation_i = u64s();
        fabric.pool_request_valid_o = bools();
        fabric.pool_request_ready_i = bools();
        fabric.pool_request_index_o = u32s();
        fabric.pool_request_generation_o = u64s();
        fabric.endpoint_pool_response_valid_o = bools();
        fabric.endpoint_event_wait_i = bools();
        fabric.endpoint_event_consume_i = bools();
        fabric.event_wait_o = bools();
        fabric.event_consume_o = bools();
        fabric.event_buffer_valid_i = bools();
        fabric.endpoint_event_valid_o = bools();
        fabric.sync_participant_mask_i = bools();
        fabric.endpoint_sync_arrive_valid_i = bools();
        fabric.endpoint_notify_valid_i = bools();
        fabric.endpoint_notify_ready_o = bools();
        fabric.endpoint_sync_release_valid_o = bools();
        fabric.endpoint_mem_packet_valid_i = bools();
        fabric.endpoint_mem_packet_ordinal_i = u32s();
        fabric.mem_packet_valid_o = bools();
        fabric.mem_packet_ordinal_o = u32s();
        fabric.mem_packet_retired_i = bools();
        fabric.endpoint_mem_packet_retired_o = bools();
    }

    void sample_queue_output(std::size_t queue, QueueOutputs& output)
    {
        output.state = controllers_[queue].state();
        if (!configured_[queue]) return;
        if (distributed_mode_) {
            output.fetch_offset = endpoint_retire_offset_[queue];
            output.retire_offset = endpoint_retire_offset_[queue];
            output.in_flight_reads = 0;
            output.cache_level = pending_heads_[queue].has_value() ? 1 : 0;
            output.primed = distributed_store_.loaded(queue);
            output.done = endpoint_retire_offset_[queue] == endpoint_count_[queue];
            return;
        }
        output.fetch_offset = caches_[queue].fetch_offset();
        output.retire_offset = caches_[queue].retire_offset();
        output.in_flight_reads = caches_[queue].in_flight_reads();
        output.cache_level = caches_[queue].level();
        output.primed = caches_[queue].primed();
        output.done = caches_[queue].done();
    }

    void sample_hardware_view(const std::vector<bool>& event_wait)
    {
        const auto count = caches_.size();
        hardware_view_.all_primed_o = all_primed();
        hardware_view_.all_done_o = true;
        hardware_view_.fabric.sync_participant_mask_i = barrier_.participants();
        hardware_view_.fabric.endpoint_sync_arrive_valid_i = pending_arrivals_;
        hardware_view_.fabric.endpoint_notify_valid_i =
            pending_notify_requests_;
        hardware_view_.fabric.endpoint_notify_ready_o =
            pending_barrier_outputs_.notify_ready;
        hardware_view_.fabric.endpoint_sync_release_valid_o =
            pending_barrier_outputs_.release;
        hardware_view_.event_buffer.event_wait_i = event_wait;
        hardware_view_.event_buffer.event_valid_o =
            pending_event_selection_.found ? pending_event_selection_.visible
                                           : std::vector<bool>(count, false);
        hardware_view_.event_buffer.event_tag_o =
            pending_event_selection_.found ? pending_event_selection_.tag : 0;
        std::vector<bool> predicted_event_fires(count, false);
        for (std::size_t queue = 0; queue < count; ++queue)
            predicted_event_fires[queue] = pending_outputs_[queue].event_fire;
        hardware_view_.event_buffer.event_consume_i = predicted_event_fires;
        hardware_view_.event_buffer.event_in_valid_i = pending_event_in_valid_;
        hardware_view_.event_buffer.event_in_tag_i = pending_event_in_tag_;
        hardware_view_.event_buffer.event_in_target_mask_i =
            pending_event_in_targets_.empty()
                ? std::vector<bool>(count, false) : pending_event_in_targets_;
        hardware_view_.event_buffer.event_in_ready_o =
            event_manager_.level() < event_manager_.capacity()
            || event_manager_.selection_will_retire(
                pending_event_selection_, predicted_event_fires);
        hardware_view_.event_buffer.fault_o = event_manager_.fault();
        hardware_view_.event_buffer.fault_code_o = event_manager_.fault_code();
        hardware_view_.fabric.endpoint_event_wait_i = event_wait;
        hardware_view_.fabric.endpoint_event_consume_i =
            predicted_event_fires;
        hardware_view_.fabric.event_wait_o = event_wait;
        hardware_view_.fabric.event_consume_o = predicted_event_fires;
        hardware_view_.fabric.event_buffer_valid_i =
            hardware_view_.event_buffer.event_valid_o;
        hardware_view_.fabric.event_buffer_tag_i =
            hardware_view_.event_buffer.event_tag_o;
        hardware_view_.fabric.event_in_valid_i = pending_event_in_valid_;
        hardware_view_.fabric.event_in_ready_i =
            hardware_view_.event_buffer.event_in_ready_o;
        hardware_view_.fabric.event_backpressure_o = pending_event_in_valid_
            && !hardware_view_.event_buffer.event_in_ready_o;
        hardware_view_.fabric.endpoint_event_valid_o =
            hardware_view_.event_buffer.event_valid_o;
        hardware_view_.fabric.endpoint_event_tag_o =
            hardware_view_.event_buffer.event_tag_o;
        hardware_view_.fabric.fault_o = barrier_.fault();
        hardware_view_.fabric.fault_code_o = barrier_.fault_code();
        for (std::size_t queue = 0; queue < count; ++queue) {
            const auto* head = configured_[queue] ? caches_[queue].head() : nullptr;
            const auto& input = pending_inputs_[queue];
            const auto& output = pending_outputs_[queue];
            auto& controller = hardware_view_.controllers[queue];
            controller.run_i = input.run;
            controller.configured_i = input.configured;
            controller.primed_i = input.primed;
            controller.done_i = input.done;
            controller.scheduler_cycle_i = input.scheduler_cycle;
            controller.scheduler_counter_ce_o = output.scheduler_counter_ce;
            controller.head_valid_i = head != nullptr;
            controller.head_record_i = head ? *head : qctl_zero_record();
            controller.head_retire_o = output.retire_head;
            controller.sync_arrive_valid_o = output.sync_arrival_valid;
            controller.sync_release_valid_i = input.sync_release;
            controller.notify_valid_o = output.notify_valid;
            controller.notify_ready_i = input.notify_ready;
            controller.event_valid_i = input.event_valid;
            controller.event_tag_i = input.event_tag;
            controller.event_wait_o = output.event_wait;
            controller.event_consume_o = output.event_fire;
            controller.mem_packet_valid_o = output.mem_packet_valid;
            controller.mem_packet_ordinal_o = output.mem_packet_ordinal;
            controller.mem_packet_retired_i = input.mem_packet_retired;
            controller.fault_o = output.fault;
            controller.fault_code_o = output.fault_code;

            auto& cache = hardware_view_.caches[queue];
            cache.configured_o = configured_[queue];
            cache.primed_o = configured_[queue] && caches_[queue].primed();
            cache.done_o = configured_[queue] && caches_[queue].done();
            cache.head_valid_o = head != nullptr;
            cache.head_record_o = head ? *head : qctl_zero_record();
            cache.head_retire_i = output.retire_head;
            cache.pool_request_valid_o = configured_[queue]
                && caches_[queue].wants_request();
            cache.pool_request_index_o = configured_[queue]
                ? static_cast<std::uint32_t>(caches_[queue].fetch_offset()) : 0;
            cache.pool_request_generation_o = configured_[queue]
                ? caches_[queue].generation() : 0;

            auto& endpoint = hardware_view_.endpoints[queue];
            endpoint.run_i = controller.run_i;
            endpoint.pool_request_valid_o = cache.pool_request_valid_o;
            endpoint.pool_request_index_o = cache.pool_request_index_o;
            endpoint.pool_request_generation_o =
                cache.pool_request_generation_o;
            endpoint.scheduler_cycle_i = controller.scheduler_cycle_i;
            endpoint.scheduler_counter_ce_o = controller.scheduler_counter_ce_o;
            endpoint.event_valid_i = controller.event_valid_i;
            endpoint.event_tag_i = controller.event_tag_i;
            endpoint.event_wait_o = controller.event_wait_o;
            endpoint.event_consume_o = controller.event_consume_o;
            endpoint.sync_arrive_valid_o = controller.sync_arrive_valid_o;
            endpoint.sync_release_valid_i = controller.sync_release_valid_i;
            endpoint.notify_valid_o = controller.notify_valid_o;
            endpoint.notify_ready_i = controller.notify_ready_i;
            endpoint.mem_packet_valid_o = controller.mem_packet_valid_o;
            endpoint.mem_packet_ordinal_o = controller.mem_packet_ordinal_o;
            endpoint.configured_o = cache.configured_o;
            endpoint.primed_o = cache.primed_o;
            endpoint.done_o = cache.done_o;
            endpoint.fault_o = cache.fault_o || controller.fault_o;
            endpoint.fault_code_o = cache.fault_o ? cache.fault_code_o
                                                  : controller.fault_code_o;

            hardware_view_.endpoint_configured_i[queue] = endpoint.configured_o;
            hardware_view_.endpoint_primed_i[queue] = endpoint.primed_o;
            hardware_view_.endpoint_done_i[queue] = endpoint.done_o;
            hardware_view_.endpoint_fault_i[queue] = endpoint.fault_o;
            hardware_view_.endpoint_fault_code_i[queue] = endpoint.fault_code_o;
            // RTL queue_active_o identifies configured endpoints, including an
            // already-done zero-record partition.
            hardware_view_.queue_active_o[queue] = configured_[queue];
            hardware_view_.all_done_o = hardware_view_.all_done_o
                && (!configured_[queue] || endpoint.done_o);
            hardware_view_.fabric.endpoint_mem_packet_valid_i[queue] =
                output.mem_packet_valid;
            hardware_view_.fabric.endpoint_mem_packet_ordinal_i[queue] =
                output.mem_packet_ordinal;
            hardware_view_.fabric.mem_packet_valid_o[queue] =
                output.mem_packet_valid;
            hardware_view_.fabric.mem_packet_ordinal_o[queue] =
                output.mem_packet_ordinal;
        }
        hardware_view_.fault_o = barrier_.fault() || event_manager_.fault()
            || pool_configuration_fault_ || pool_.fault()
            || std::ranges::any_of(hardware_view_.endpoint_fault_i,
                [](bool value) { return value; });
        hardware_view_.fault_code_o = encode_global_fault_code(
            pool_configuration_fault_ || pool_.fault(),
            pool_configuration_fault_ ? pool_configuration_fault_code_
                                      : pool_.fault_code(),
            event_manager_.fault(), event_manager_.fault_code(),
            barrier_.fault(), barrier_.fault_code(),
            hardware_view_.endpoint_fault_i,
            hardware_view_.endpoint_fault_code_i);
    }

    void clear_load_interface_view()
    {
        auto& pool = hardware_view_.pool;
        pool.rst_ni = true;
        pool.clear_i = false;
        pool.load_mode_i = false;
        pool.partition_valid_i = false;
        pool.partition_ready_o = false;
        pool.load_valid_i = false;
        pool.load_ready_o = false;
        std::fill(hardware_view_.endpoint_configure_valid_o.begin(),
            hardware_view_.endpoint_configure_valid_o.end(), false);
        std::fill(hardware_view_.endpoint_configure_count_o.begin(),
            hardware_view_.endpoint_configure_count_o.end(), 0);
        std::fill(hardware_view_.endpoint_configure_generation_o.begin(),
            hardware_view_.endpoint_configure_generation_o.end(), 0);
        for (std::size_t queue = 0; queue < hardware_view_.endpoints.size();
             ++queue) {
            hardware_view_.endpoints[queue].configure_valid_i = false;
            hardware_view_.caches[queue].configure_valid_i = false;
            hardware_view_.controllers[queue].configure_valid_i = false;
        }
    }

    void sample_pool_interface(bool allow_grant = true)
    {
        const auto count = caches_.size();
        auto& pool = hardware_view_.pool;
        pool.request_valid_i.assign(count, false);
        pool.request_ready_o.assign(count, false);
        pool.request_index_i.assign(count, 0);
        pool.request_generation_i.assign(count, 0);
        for (std::size_t queue = 0; queue < count; ++queue) {
            if (!configured_[queue]) continue;
            pool.request_valid_i[queue] = caches_[queue].request_valid();
            pool.request_index_i[queue] = static_cast<std::uint32_t>(
                caches_[queue].fetch_offset());
            pool.request_generation_i[queue] = caches_[queue].generation();
        }
        if (allow_grant)
            if (const auto grant = pool_.preview_grant(pool.request_valid_i))
                pool.request_ready_o[*grant] = true;
        pool.response_valid_o = pool_.last_response_valid();
        pool.response_queue_o = pool_.last_response_valid()
            ? static_cast<std::uint32_t>(pool_.last_response_queue()) : 0;
        pool.response_data_o = pool_.last_response_valid()
            ? pool_.last_response().record : qctl_zero_record();
        pool.response_error_o = pool_.last_response_valid()
            && pool_.last_response().error;
        pool.fault_o = pool_configuration_fault_ || pool_.fault();
        pool.fault_code_o = pool_configuration_fault_
            ? pool_configuration_fault_code_ : pool_.fault_code();
        auto& fabric = hardware_view_.fabric;
        fabric.endpoint_pool_request_valid_i = pool.request_valid_i;
        fabric.endpoint_pool_request_ready_o = pool.request_ready_o;
        fabric.endpoint_pool_request_index_i = pool.request_index_i;
        fabric.endpoint_pool_request_generation_i =
            pool.request_generation_i;
        fabric.pool_request_valid_o = pool.request_valid_i;
        fabric.pool_request_ready_i = pool.request_ready_o;
        fabric.pool_request_index_o = pool.request_index_i;
        fabric.pool_request_generation_o = pool.request_generation_i;
        fabric.pool_response_valid_i = pool.response_valid_o;
        fabric.pool_response_queue_i = pool.response_queue_o;
        fabric.pool_response_data_i = pool.response_data_o;
        fabric.pool_response_error_i = pool.response_error_o;
        fabric.endpoint_pool_response_valid_o.assign(count, false);
        if (pool.response_valid_o && pool.response_queue_o < count)
            fabric.endpoint_pool_response_valid_o[pool.response_queue_o] = true;
        fabric.endpoint_pool_response_data_o = pool.response_data_o;
        fabric.endpoint_pool_response_error_o = pool.response_error_o;
        for (std::size_t queue = 0; queue < count; ++queue) {
            auto& cache = hardware_view_.caches[queue];
            auto& endpoint = hardware_view_.endpoints[queue];
            cache.pool_request_valid_o = pool.request_valid_i[queue];
            cache.pool_request_ready_i = pool.request_ready_o[queue];
            cache.pool_request_index_o = pool.request_index_i[queue];
            cache.pool_request_generation_o =
                pool.request_generation_i[queue];
            cache.pool_response_valid_i =
                fabric.endpoint_pool_response_valid_o[queue];
            cache.pool_response_data_i = pool.response_data_o;
            cache.pool_response_error_i = pool.response_error_o;
            endpoint.pool_request_valid_o = cache.pool_request_valid_o;
            endpoint.pool_request_ready_i = cache.pool_request_ready_i;
            endpoint.pool_request_index_o = cache.pool_request_index_o;
            endpoint.pool_request_generation_o =
                cache.pool_request_generation_o;
            endpoint.pool_response_valid_i = cache.pool_response_valid_i;
            endpoint.pool_response_data_i = cache.pool_response_data_i;
            endpoint.pool_response_error_i = cache.pool_response_error_i;
        }
    }

    void sample_global_aliases()
    {
        auto& global = hardware_view_;
        const auto& pool = global.pool;
        const auto& fabric = global.fabric;
        global.rst_ni = pool.rst_ni;
        global.clear_i = pool.clear_i;
        global.load_mode_i = pool.load_mode_i;
        global.partition_valid_i = pool.partition_valid_i;
        global.partition_ready_o = pool.partition_ready_o;
        global.partition_queue_i = pool.partition_queue_i;
        global.partition_base_i = pool.partition_base_i;
        global.partition_count_i = pool.partition_count_i;
        global.partition_generation_i = pool.partition_generation_i;
        global.load_valid_i = pool.load_valid_i;
        global.load_ready_o = pool.load_ready_o;
        global.load_address_i = pool.load_address_i;
        global.load_data_i = pool.load_data_i;
        global.event_in_valid_i = global.event_buffer.event_in_valid_i;
        global.event_in_ready_o = global.event_buffer.event_in_ready_o;
        global.event_in_tag_i = global.event_buffer.event_in_tag_i;
        global.event_in_target_mask_i =
            global.event_buffer.event_in_target_mask_i;
        global.event_backpressure_o = fabric.event_backpressure_o;
        global.endpoint_event_wait_i = fabric.endpoint_event_wait_i;
        global.endpoint_event_consume_i =
            fabric.endpoint_event_consume_i;
        global.endpoint_event_valid_o = fabric.endpoint_event_valid_o;
        global.endpoint_event_tag_o = fabric.endpoint_event_tag_o;
        global.sync_participant_mask_i = fabric.sync_participant_mask_i;
        global.endpoint_sync_arrive_valid_i =
            fabric.endpoint_sync_arrive_valid_i;
        global.endpoint_notify_valid_i = fabric.endpoint_notify_valid_i;
        global.endpoint_notify_ready_o = fabric.endpoint_notify_ready_o;
        global.endpoint_sync_release_valid_o =
            fabric.endpoint_sync_release_valid_o;
        global.endpoint_mem_packet_valid_i =
            fabric.endpoint_mem_packet_valid_i;
        global.endpoint_mem_packet_ordinal_i =
            fabric.endpoint_mem_packet_ordinal_i;
        global.mem_packet_valid_o = fabric.mem_packet_valid_o;
        global.mem_packet_ordinal_o = fabric.mem_packet_ordinal_o;
        global.mem_packet_retired_i = fabric.mem_packet_retired_i;
        global.endpoint_mem_packet_retired_o =
            fabric.endpoint_mem_packet_retired_o;
    }

    void sample_global_status_only()
    {
        hardware_view_.all_primed_o = true;
        hardware_view_.all_done_o = true;
        hardware_view_.pool.fault_o =
            pool_configuration_fault_ || pool_.fault();
        hardware_view_.pool.fault_code_o = pool_configuration_fault_
            ? pool_configuration_fault_code_ : pool_.fault_code();
        hardware_view_.event_buffer.fault_o = event_manager_.fault();
        hardware_view_.event_buffer.fault_code_o =
            event_manager_.fault_code();
        hardware_view_.fabric.fault_o = barrier_.fault();
        hardware_view_.fabric.fault_code_o = barrier_.fault_code();
        for (std::size_t queue = 0; queue < configured_.size(); ++queue) {
            const bool cache_fault = configured_[queue]
                && caches_[queue].fault();
            const bool controller_fault = controllers_[queue].fault();
            const bool endpoint_fault = cache_fault || controller_fault;
            const auto endpoint_fault_code = cache_fault
                ? caches_[queue].fault_code() : controllers_[queue].fault_code();
            hardware_view_.caches[queue].configured_o = configured_[queue];
            hardware_view_.caches[queue].primed_o = configured_[queue]
                && caches_[queue].primed();
            hardware_view_.caches[queue].done_o = configured_[queue]
                && caches_[queue].done();
            hardware_view_.caches[queue].fault_o = cache_fault;
            hardware_view_.caches[queue].fault_code_o = cache_fault
                ? caches_[queue].fault_code() : 0;
            hardware_view_.controllers[queue].fault_o = controller_fault;
            hardware_view_.controllers[queue].fault_code_o = controller_fault
                ? controllers_[queue].fault_code() : 0;
            hardware_view_.endpoints[queue].configured_o = configured_[queue];
            hardware_view_.endpoints[queue].primed_o = configured_[queue]
                && caches_[queue].primed();
            hardware_view_.endpoints[queue].done_o = configured_[queue]
                && caches_[queue].done();
            hardware_view_.endpoints[queue].fault_o = endpoint_fault;
            hardware_view_.endpoints[queue].fault_code_o = endpoint_fault
                ? endpoint_fault_code : 0;
            hardware_view_.endpoint_configured_i[queue] = configured_[queue];
            hardware_view_.endpoint_primed_i[queue] = configured_[queue]
                && caches_[queue].primed();
            hardware_view_.endpoint_done_i[queue] = configured_[queue]
                && caches_[queue].done();
            hardware_view_.endpoint_fault_i[queue] = endpoint_fault;
            hardware_view_.endpoint_fault_code_i[queue] = endpoint_fault
                ? endpoint_fault_code : 0;
            hardware_view_.queue_active_o[queue] = configured_[queue];
            hardware_view_.all_primed_o = hardware_view_.all_primed_o
                && (!configured_[queue] || caches_[queue].primed());
            hardware_view_.all_done_o = hardware_view_.all_done_o
                && (!configured_[queue] || caches_[queue].done());
        }
        hardware_view_.fault_o = hardware_view_.pool.fault_o
            || event_manager_.fault() || barrier_.fault()
            || std::ranges::any_of(hardware_view_.endpoint_fault_i,
                [](bool value) { return value; });
        hardware_view_.fault_code_o = encode_global_fault_code(
            hardware_view_.pool.fault_o,
            hardware_view_.pool.fault_code_o,
            event_manager_.fault(), event_manager_.fault_code(),
            barrier_.fault(), barrier_.fault_code(),
            hardware_view_.endpoint_fault_i,
            hardware_view_.endpoint_fault_code_i);
    }

    void check_queue(std::size_t queue) const
    {
        if (queue >= caches_.size())
            throw StaticScheduleError("QCTL Runtime Queue index out of range");
    }

    IcuQctlPool pool_{};
    std::vector<IcuQctlCache> caches_{};
    std::vector<IcuQctlController> controllers_{};
    std::vector<bool> configured_{};
    IcuQctlDistributedProgramStore distributed_store_{};
    std::vector<std::uint32_t> endpoint_count_{};
    std::vector<std::uint64_t> endpoint_generation_{};
    std::vector<std::uint32_t> endpoint_retire_offset_{};
    std::vector<bool> endpoint_fault_{};
    std::vector<std::uint8_t> endpoint_fault_code_{};
    bool distributed_mode_{false};
    IcuQctlArrivalReleaseBarrier barrier_{};
    IcuQctlEventManager event_manager_{};
    std::size_t cache_depth_{0};
    bool barrier_configured_{false};
    IcuQctlGlobalInterfaceView hardware_view_{};
    bool cycle_open_{false};
    std::vector<IcuQctlControllerInputs> pending_inputs_{};
    std::vector<QueueOutputs> pending_outputs_{};
    std::vector<std::optional<IcuQueueControl>> pending_heads_{};
    std::vector<bool> pending_head_error_{};
    std::vector<bool> pending_arrivals_{};
    // Current held valid requests only; this is deliberately not a Token
    // store and is rebuilt from Controller outputs every cycle.
    std::vector<bool> pending_notify_requests_{};
    IcuQctlArrivalReleaseBarrier::Outputs pending_barrier_outputs_{};
    IcuQctlEventManager::Selection pending_event_selection_{};
    bool pending_event_in_valid_{false};
    std::uint32_t pending_event_in_tag_{0};
    std::vector<bool> pending_event_in_targets_{};
    bool pending_event_input_preaccepted_{false};
    bool pool_configuration_fault_{false};
    std::uint8_t pool_configuration_fault_code_{0};
    bool load_cycle_open_{false};
    bool load_clocks_read_pipeline_{false};
    LoadInputs pending_load_inputs_{};
};

} // namespace ftlpu
