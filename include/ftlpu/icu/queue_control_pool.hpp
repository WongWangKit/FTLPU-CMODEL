#pragma once

#include "ftlpu/icu/instruction.hpp"
#include <algorithm>
#include <cstdint>
#include <deque>
#include <limits>
#include <span>
#include <vector>
#include <utility>

namespace ftlpu {

inline constexpr auto kQueueControlMemPacket = static_cast<IcuControlOpcode>(8);
struct IcuQueueControl {
    std::uint32_t cycle{0};
    IcuControlOpcode opcode{IcuControlOpcode::WaitEvent};
    std::uint32_t operand{0};
};

inline void validate_queue_control_entries(std::span<const IcuQueueControl> entries)
{
    for (std::size_t i=0;i<entries.size();++i) {
        const auto& e=entries[i];
        if ((i && e.cycle<entries[i-1].cycle)
            || (e.opcode!=IcuControlOpcode::Sync && e.opcode!=IcuControlOpcode::WaitEvent
                && e.opcode!=IcuControlOpcode::Notify && e.opcode!=kQueueControlMemPacket)
            || (e.opcode!=IcuControlOpcode::WaitEvent && e.opcode!=kQueueControlMemPacket && e.operand)
            || (e.opcode==kQueueControlMemPacket && !e.operand))
            throw StaticScheduleError("invalid queue control entry or order");
    }
}

// One read port for the entire chip. Partition state contains only a directory,
// cursors, bounded cache and response credits; the only backing records are ram_.
class IcuQueueControlPool {
public:
    struct Handle { std::size_t index; std::uint64_t generation; };
    struct Statistics {
        std::uint64_t reads{0},prime_reads{0},run_reads{0},cycles{0};
        std::size_t peak_pending{0},peak_cached{0},max_arbitration_wait{0};
    };
    void reset() {
        if (generation_==std::numeric_limits<std::uint64_t>::max())
            throw StaticScheduleError("QCTL pool generation overflow");
        ++generation_;ram_.clear();partitions_.clear();pending_.clear();
        capacity_=0;cursor_=0;cycle_=0;open_=false;stats_={};
    }
    void configure(std::size_t capacity,std::size_t queues,std::size_t cache,
                   std::size_t latency,std::size_t ii=1) {
        if(!capacity || !queues || !cache || !latency || !ii)
            throw StaticScheduleError("invalid shared QCTL geometry");
        reset();capacity_=capacity;queue_limit_=queues;cache_depth_=cache;
        latency_=latency;ii_=ii;next_request_=0;
    }
    Handle add_partition(std::span<const IcuQueueControl> entries,std::size_t queue_limit) {
        if(!capacity_ || cycle_ || open_ || partitions_.size()==queue_limit_
            || entries.size()>queue_limit || entries.size()>capacity_-ram_.size())
            throw StaticScheduleError("shared QCTL pool or partition capacity exceeded");
        validate_queue_control_entries(entries);
        const auto id=partitions_.size();
        partitions_.push_back(Partition{ram_.size(),entries.size()});
        ram_.insert(ram_.end(),entries.begin(),entries.end());
        return {id,generation_};
    }
    void begin_cycle() {
        if(!capacity_) return;
        if(open_) throw StaticScheduleError("shared QCTL advanced twice in one physical cycle");
        open_=true;
        while(!pending_.empty() && pending_.front().ready<=cycle_) {
            auto response=pending_.front();pending_.pop_front();
            if(response.generation!=generation_) continue;
            auto& q=partitions_.at(response.queue);
            if(!q.inflight || q.cache.size()>=cache_depth_)
                throw StaticScheduleError("shared QCTL response lost reserved credit");
            --q.inflight;q.cache.push_back(response.entry);
            stats_.peak_cached=std::max(stats_.peak_cached,q.cache.size());
        }
    }
    void end_cycle(bool prime=false) {
        if(!capacity_) return;
        if(!open_) throw StaticScheduleError("shared QCTL cycle has no begin edge");
        // One round-robin grant, irrespective of the number of local queues.
        for(auto& q:partitions_) {
            if(wants_read(q)) { if(!q.requesting) {q.requesting=true;q.request_since=cycle_;} }
            else q.requesting=false;
        }
        if(cycle_>=next_request_ && !partitions_.empty()) {
            for(std::size_t step=0;step<partitions_.size();++step) {
                const auto id=(cursor_+step)%partitions_.size();auto& q=partitions_[id];
                if(!wants_read(q)) continue;
                if(pending_.size()>=(latency_+ii_-1)/ii_)
                    throw StaticScheduleError("shared QCTL response pipeline capacity exceeded");
                stats_.max_arbitration_wait=std::max(stats_.max_arbitration_wait,
                    static_cast<std::size_t>(cycle_-q.request_since));
                pending_.push_back({cycle_+latency_,generation_,id,ram_.at(q.base+q.fetch)});
                ++q.fetch;++q.inflight;q.requesting=false;
                ++stats_.reads;if(prime)++stats_.prime_reads;else ++stats_.run_reads;
                stats_.peak_pending=std::max(stats_.peak_pending,pending_.size());
                cursor_=(id+1)%partitions_.size();next_request_=cycle_+ii_;
                break;
            }
        }
        ++cycle_;++stats_.cycles;open_=false;
    }
    bool primed(Handle h) const {const auto& q=get(h);return q.cache.size()==std::min(cache_depth_,q.count);}
    bool done(Handle h) const {const auto& q=get(h);return q.retired==q.count;}
    std::size_t size(Handle h) const {return get(h).count;}
    std::size_t retired(Handle h) const {return get(h).retired;}
    std::size_t used() const noexcept {return ram_.size();}
    const Statistics& statistics() const noexcept {return stats_;}
    const IcuQueueControl* due(Handle h,std::size_t cycle) const {
        const auto& q=get(h);
        if(q.retired==q.count)return nullptr;
        // No backing-RAM peek: an unknown boundary must never be passed.
        if(q.cache.empty())throw StaticScheduleError("shared QCTL cache underflow: next control boundary unavailable");
        if(q.cache.front().cycle<cycle)throw StaticScheduleError("shared QCTL boundary missed");
        return q.cache.front().cycle==cycle ? &q.cache.front() : nullptr;
    }
    const IcuQueueControl* cached_due(Handle h,std::size_t cycle) const {
        const auto& q=get(h);
        return !q.cache.empty() && q.cache.front().cycle==cycle ? &q.cache.front() : nullptr;
    }
    void retire(Handle h) {
        auto& q=get(h);
        if(q.cache.empty())throw StaticScheduleError("shared QCTL retire without fetched entry");
        q.cache.pop_front();++q.retired;
    }
private:
    struct Partition {
        std::size_t base{0},count{0},fetch{0},retired{0},inflight{0};
        std::deque<IcuQueueControl> cache{};
        bool requesting{false};std::uint64_t request_since{0};
    };
    struct Response {std::uint64_t ready,generation;std::size_t queue;IcuQueueControl entry;};
    const Partition& get(Handle h) const {
        if(h.generation!=generation_ || h.index>=partitions_.size())
            throw StaticScheduleError("stale shared QCTL partition handle");
        return partitions_[h.index];
    }
    Partition& get(Handle h) {return const_cast<Partition&>(std::as_const(*this).get(h));}
    bool wants_read(const Partition& q) const {
        return q.fetch<q.count && q.cache.size()+q.inflight<cache_depth_;
    }
    std::vector<IcuQueueControl> ram_{};
    std::vector<Partition> partitions_{};
    std::deque<Response> pending_{};
    std::size_t capacity_{0},queue_limit_{0},cache_depth_{0},latency_{0},ii_{1},cursor_{0};
    std::uint64_t generation_{0},cycle_{0},next_request_{0};
    bool open_{false};Statistics stats_{};
};
} // namespace ftlpu
