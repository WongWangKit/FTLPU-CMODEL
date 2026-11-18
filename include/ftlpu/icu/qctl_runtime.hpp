#pragma once

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
        validate_queue_control_entries(std::span<const IcuQueueControl>(&record, 1));
        records_[address] = record;
    }

    const Partition& partition(std::size_t queue) const
    {
        check_queue(queue);
        if (!partitions_[queue])
            throw StaticScheduleError("QCTL Pool Queue has no partition");
        return *partitions_[queue];
    }

    void begin_cycle()
    {
        if (open_) throw StaticScheduleError("QCTL Pool cycle opened twice");
        open_ = true;
        while (!pending_.empty() && pending_.front().ready_cycle <= cycle_) {
            auto pending = std::move(pending_.front());
            pending_.pop_front();
            if (responses_[pending.queue])
                throw StaticScheduleError("QCTL Pool response was not consumed");
            responses_[pending.queue] = Response{pending.record, pending.error};
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

    void end_cycle()
    {
        if (!open_) throw StaticScheduleError("QCTL Pool cycle has no begin");
        if (cycle_ >= next_request_cycle_ && !requests_.empty()) {
            for (std::size_t step = 0; step < requests_.size(); ++step) {
                const auto queue = (round_robin_ + step) % requests_.size();
                if (!requests_[queue]) continue;
                const auto request = *requests_[queue];
                const auto& descriptor = partition(queue);
                const auto address = descriptor.base + request.offset;
                const bool error = request.generation != descriptor.generation
                    || request.offset >= descriptor.count || address >= records_.size()
                    || !records_[address].has_value();
                statistics_.maximum_arbitration_wait = std::max(
                    statistics_.maximum_arbitration_wait,
                    static_cast<std::size_t>(cycle_ - request.first_cycle));
                pending_.push_back(Pending{cycle_ + latency_, queue,
                    error ? IcuQueueControl{} : *records_[address], error});
                requests_[queue].reset();
                grants_[queue] = true;
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
        reset_runtime();
    }

    void reset_runtime()
    {
        cache_.clear();
        fetch_offset_ = 0;
        retire_offset_ = 0;
        in_flight_ = 0;
        request_pending_ = false;
        primed_ = count_ == 0;
    }

    void begin_cycle()
    {
        ensure_configured();
        if (pool_->take_grant(queue_)) {
            if (!request_pending_ || fetch_offset_ >= count_)
                throw StaticScheduleError("QCTL Cache observed an unexpected grant");
            request_pending_ = false;
            ++fetch_offset_;
            ++in_flight_;
        }
        if (auto response = pool_->take_response(queue_)) {
            if (in_flight_ == 0 || response->error)
                throw StaticScheduleError("QCTL Cache received an invalid response");
            if (cache_.size() >= cache_depth_)
                throw StaticScheduleError("QCTL Cache response overflow");
            --in_flight_;
            cache_.push_back(response->record);
        }
        if (cache_.size() == std::min(cache_depth_, count_)) primed_ = true;
    }

    void end_cycle()
    {
        ensure_configured();
        if (!request_pending_ && fetch_offset_ < count_
            && cache_.size() + in_flight_ < cache_depth_) {
            pool_->request(queue_, fetch_offset_, generation_);
            request_pending_ = true;
        }
    }

    const IcuQueueControl* head() const noexcept
    { return cache_.empty() ? nullptr : &cache_.front(); }

    void retire()
    {
        if (cache_.empty())
            throw StaticScheduleError("QCTL Cache retired an empty head");
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
    std::deque<IcuQueueControl> cache_{};
    std::size_t fetch_offset_{0};
    std::size_t retire_offset_{0};
    std::size_t in_flight_{0};
    bool request_pending_{false};
    bool primed_{false};
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
};

struct IcuQctlControllerOutputs {
    bool scheduler_counter_ce{true};
    bool retire_head{false};
    bool sync_arrival_valid{false};
    bool event_ready{false};
    bool event_fire{false};
    std::uint32_t wait_event_tag{0};
    bool notify_valid{false};
    bool notify_fire{false};
    bool mem_packet_valid{false};
    std::uint32_t mem_packet_ordinal{0};
};

// Queue-local semantic FSM.  It has no local cycle counter, Pool address,
// cache FIFO, fetch offset, retire offset or in-flight read ownership.
class IcuQctlController {
public:
    void reset() noexcept
    {
        state_ = IcuQctlControllerState::IDLE;
        held_operand_ = 0;
    }

    IcuQctlControllerOutputs evaluate(
        const IcuQctlControllerInputs& inputs) const
    {
        IcuQctlControllerOutputs outputs{};
        if (state_ == IcuQctlControllerState::IDLE) {
            if (!inputs.head || inputs.head->cycle > inputs.scheduler_cycle)
                return outputs;
            if (inputs.head->cycle < inputs.scheduler_cycle)
                throw StaticScheduleError("QCTL boundary missed by Scheduler");
            outputs.scheduler_counter_ce = false;
            return drive(inputs.head->opcode, inputs.head->operand,
                inputs, outputs);
        }
        outputs.scheduler_counter_ce = false;
        switch (state_) {
        case IcuQctlControllerState::SYNC_WAIT:
            return drive(IcuControlOpcode::Sync, 0, inputs, outputs);
        case IcuQctlControllerState::EVENT_WAIT:
            return drive(IcuControlOpcode::WaitEvent, held_operand_, inputs, outputs);
        case IcuQctlControllerState::NOTIFY_WAIT:
            return drive(IcuControlOpcode::Notify, 0, inputs, outputs);
        case IcuQctlControllerState::MEMPACKET_WAIT:
            return drive(kQueueControlMemPacket, held_operand_, inputs, outputs);
        case IcuQctlControllerState::IDLE:
            break;
        }
        return outputs;
    }

    void commit(const IcuQctlControllerInputs& inputs,
        const IcuQctlControllerOutputs& outputs)
    {
        if (outputs.retire_head) {
            state_ = IcuQctlControllerState::IDLE;
            held_operand_ = 0;
            return;
        }
        if (state_ != IcuQctlControllerState::IDLE || !inputs.head
            || inputs.head->cycle != inputs.scheduler_cycle)
            return;
        held_operand_ = inputs.head->operand;
        switch (inputs.head->opcode) {
        case IcuControlOpcode::Sync:
            if (held_operand_ != 0)
                throw StaticScheduleError("QCTL SYNC operand must be zero");
            state_ = IcuQctlControllerState::SYNC_WAIT;
            break;
        case IcuControlOpcode::WaitEvent:
            state_ = IcuQctlControllerState::EVENT_WAIT;
            break;
        case IcuControlOpcode::Notify:
            if (held_operand_ != 0)
                throw StaticScheduleError("QCTL NOTIFY operand must be zero");
            state_ = IcuQctlControllerState::NOTIFY_WAIT;
            break;
        default:
            if (inputs.head->opcode != kQueueControlMemPacket
                || held_operand_ == 0)
                throw StaticScheduleError("invalid QCTL Controller record");
            state_ = IcuQctlControllerState::MEMPACKET_WAIT;
            break;
        }
    }

    IcuQctlControllerState state() const noexcept { return state_; }
    std::uint32_t held_operand() const noexcept { return held_operand_; }

private:
    static IcuQctlControllerOutputs drive(IcuControlOpcode opcode,
        std::uint32_t operand, const IcuQctlControllerInputs& inputs,
        IcuQctlControllerOutputs outputs)
    {
        switch (opcode) {
        case IcuControlOpcode::Sync:
            if (operand != 0)
                throw StaticScheduleError("QCTL SYNC operand must be zero");
            outputs.sync_arrival_valid = true;
            outputs.retire_head = inputs.sync_release;
            return outputs;
        case IcuControlOpcode::WaitEvent:
            outputs.wait_event_tag = operand;
            outputs.event_ready = inputs.event_tag == operand;
            outputs.event_fire = inputs.event_valid && outputs.event_ready;
            outputs.retire_head = outputs.event_fire;
            return outputs;
        case IcuControlOpcode::Notify:
            if (operand != 0)
                throw StaticScheduleError("QCTL NOTIFY operand must be zero");
            outputs.notify_valid = true;
            outputs.notify_fire = inputs.notify_ready;
            outputs.retire_head = outputs.notify_fire;
            return outputs;
        default:
            if (opcode != kQueueControlMemPacket || operand == 0)
                throw StaticScheduleError("invalid QCTL Controller record");
            outputs.mem_packet_valid = true;
            outputs.mem_packet_ordinal = operand;
            outputs.retire_head = inputs.mem_packet_retired;
            return outputs;
        }
    }

    IcuQctlControllerState state_{IcuQctlControllerState::IDLE};
    std::uint32_t held_operand_{0};
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
        if (participants.empty() || std::ranges::none_of(participants,
                [](bool value) { return value; }))
            throw StaticScheduleError("QCTL Barrier has no participants");
        participants_ = std::move(participants);
        arrived_.assign(participants_.size(), false);
        round_robin_ = 0;
    }

    void reset() noexcept
    {
        std::fill(arrived_.begin(), arrived_.end(), false);
        round_robin_ = 0;
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
        Outputs outputs{std::vector<bool>(combined.size(), false),
            std::vector<bool>(combined.size(), false)};
        if (!all_arrived) return outputs;
        for (std::size_t step = 0; step < notify_valid.size(); ++step) {
            const auto queue = (round_robin_ + step) % notify_valid.size();
            if (!notify_valid[queue]) continue;
            outputs.completed = true;
            outputs.notifier = queue;
            outputs.notify_ready[queue] = true;
            for (std::size_t participant = 0;
                 participant < participants_.size(); ++participant)
                outputs.release[participant] = participants_[participant];
            break;
        }
        return outputs;
    }

    void commit(const std::vector<bool>& sync_arrival,
        const Outputs& outputs)
    {
        if (sync_arrival.size() != participants_.size())
            throw StaticScheduleError("QCTL Barrier vector width mismatch");
        if (outputs.completed) {
            std::fill(arrived_.begin(), arrived_.end(), false);
            round_robin_ = (outputs.notifier + 1) % participants_.size();
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

private:
    std::vector<bool> participants_{};
    std::vector<bool> arrived_{};
    std::size_t round_robin_{0};
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

    void configure(std::size_t queue_count, std::size_t depth)
    {
        if (queue_count == 0 || depth == 0)
            throw StaticScheduleError("invalid QCTL Event Manager geometry");
        queue_count_ = queue_count;
        depth_ = depth;
        events_.clear();
    }

    void reset() noexcept { events_.clear(); }

    void push(std::uint32_t tag, std::vector<bool> consumers)
    {
        if (consumers.size() != queue_count_
            || std::ranges::none_of(consumers, [](bool value) { return value; }))
            throw StaticScheduleError("invalid QCTL Event consumer mask");
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

    std::size_t level() const noexcept { return events_.size(); }

private:
    std::size_t queue_count_{0};
    std::size_t depth_{0};
    std::deque<Event> events_{};
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

    IcuQctlRuntime() = default;
    IcuQctlRuntime(const IcuQctlRuntime&) = delete;
    IcuQctlRuntime& operator=(const IcuQctlRuntime&) = delete;

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
        cache_depth_ = cache_depth;
        event_manager_.configure(queue_count, event_depth);
        barrier_configured_ = false;
    }

    void reset()
    {
        pool_.reset();
        for (auto& cache : caches_) cache.reset_runtime();
        for (auto& controller : controllers_) controller.reset();
        std::fill(configured_.begin(), configured_.end(), false);
        barrier_.reset();
        event_manager_.reset();
        barrier_configured_ = false;
    }

    void configure_partition(std::size_t queue, std::size_t base,
        std::vector<IcuQueueControl> records, std::uint64_t generation)
    {
        check_queue(queue);
        validate_queue_control_entries(records);
        pool_.configure_partition(queue, base, records.size(), generation);
        for (std::size_t offset = 0; offset < records.size(); ++offset)
            pool_.load(base + offset, records[offset]);
        caches_[queue].configure(pool_, queue, cache_depth_);
        controllers_[queue].reset();
        configured_[queue] = true;
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

    void prime_cycle()
    {
        pool_.begin_cycle();
        for (std::size_t queue = 0; queue < caches_.size(); ++queue)
            if (configured_[queue]) caches_[queue].begin_cycle();
        for (std::size_t queue = 0; queue < caches_.size(); ++queue)
            if (configured_[queue]) caches_[queue].end_cycle();
        pool_.end_cycle();
    }

    bool all_primed() const noexcept
    {
        for (std::size_t queue = 0; queue < caches_.size(); ++queue)
            if (configured_[queue] && !caches_[queue].primed()) return false;
        return true;
    }

    std::vector<QueueOutputs> tick(
        const std::vector<std::uint32_t>& scheduler_cycles,
        const std::vector<bool>& mem_packet_retired)
    {
        if (!barrier_configured_ || scheduler_cycles.size() != caches_.size()
            || mem_packet_retired.size() != caches_.size())
            throw StaticScheduleError("invalid QCTL Runtime cycle inputs");
        if (!all_primed())
            throw StaticScheduleError("QCTL Runtime started before priming");

        pool_.begin_cycle();
        for (std::size_t queue = 0; queue < caches_.size(); ++queue)
            if (configured_[queue]) caches_[queue].begin_cycle();

        std::vector<IcuQctlControllerInputs> inputs(caches_.size());
        std::vector<IcuQctlControllerOutputs> raw(caches_.size());
        std::vector<bool> arrivals(caches_.size(), false);
        std::vector<bool> notifies(caches_.size(), false);
        for (std::size_t queue = 0; queue < caches_.size(); ++queue) {
            inputs[queue].scheduler_cycle = scheduler_cycles[queue];
            inputs[queue].head = configured_[queue]
                ? caches_[queue].head() : nullptr;
            inputs[queue].event_valid = event_manager_.valid()
                && event_manager_.targets(queue);
            inputs[queue].event_tag = event_manager_.valid()
                ? event_manager_.tag() : 0;
            inputs[queue].mem_packet_retired = mem_packet_retired[queue];
            raw[queue] = controllers_[queue].evaluate(inputs[queue]);
            arrivals[queue] = raw[queue].sync_arrival_valid;
            notifies[queue] = raw[queue].notify_valid;
        }

        const auto barrier_outputs = barrier_.evaluate(arrivals, notifies);
        std::vector<bool> event_fires(caches_.size(), false);
        std::vector<QueueOutputs> outputs(caches_.size());
        for (std::size_t queue = 0; queue < caches_.size(); ++queue) {
            inputs[queue].sync_release = barrier_outputs.release[queue];
            inputs[queue].notify_ready = barrier_outputs.notify_ready[queue];
            raw[queue] = controllers_[queue].evaluate(inputs[queue]);
            static_cast<IcuQctlControllerOutputs&>(outputs[queue]) = raw[queue];
            if (raw[queue].retire_head) caches_[queue].retire();
            controllers_[queue].commit(inputs[queue], raw[queue]);
            event_fires[queue] = raw[queue].event_fire;
            outputs[queue].state = controllers_[queue].state();
            if (configured_[queue]) {
                outputs[queue].fetch_offset = caches_[queue].fetch_offset();
                outputs[queue].retire_offset = caches_[queue].retire_offset();
                outputs[queue].in_flight_reads = caches_[queue].in_flight_reads();
                outputs[queue].cache_level = caches_[queue].level();
                outputs[queue].primed = caches_[queue].primed();
                outputs[queue].done = caches_[queue].done();
            }
        }

        event_manager_.commit(event_fires);
        barrier_.commit(arrivals, barrier_outputs);
        for (std::size_t queue = 0; queue < caches_.size(); ++queue)
            if (configured_[queue]) caches_[queue].end_cycle();
        pool_.end_cycle();
        return outputs;
    }

    const IcuQctlPool& pool() const noexcept { return pool_; }
    const IcuQctlEventManager& event_manager() const noexcept
    { return event_manager_; }

private:
    void check_queue(std::size_t queue) const
    {
        if (queue >= caches_.size())
            throw StaticScheduleError("QCTL Runtime Queue index out of range");
    }

    IcuQctlPool pool_{};
    std::vector<IcuQctlCache> caches_{};
    std::vector<IcuQctlController> controllers_{};
    std::vector<bool> configured_{};
    IcuQctlArrivalReleaseBarrier barrier_{};
    IcuQctlEventManager event_manager_{};
    std::size_t cache_depth_{0};
    bool barrier_configured_{false};
};

} // namespace ftlpu
