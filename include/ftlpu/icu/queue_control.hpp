#pragma once

#include "ftlpu/icu/instruction.hpp"
#include <algorithm>
#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

namespace ftlpu {

// Sideband-only operation. It is never encoded as a Native ICU control word.
inline constexpr auto kQueueControlMemPacket = static_cast<IcuControlOpcode>(8);

// A control boundary in the queue's control-stripped computation timeline.
// It deliberately does not identify a packed Macro record or bit offset.
struct IcuQueueControl {
    std::uint32_t cycle{0};
    IcuControlOpcode opcode{IcuControlOpcode::WaitEvent};
    std::uint32_t operand{0};
};

// One finite, single-read-port partition of the task's public control table.
// Fetch responses reserve cache credit at request time, including while the
// queue waits. Ready controls may retire from the cache in the same dispatch.
class IcuQueueControlFrontend {
public:
    void configure(std::vector<IcuQueueControl> entries, std::size_t capacity,
                   std::size_t cache_depth, std::size_t read_latency)
    {
        if (!capacity || !cache_depth || cache_depth > capacity || !read_latency
            || entries.size() > capacity)
            throw StaticScheduleError("invalid or overflowing queue control RAM geometry");
        for (std::size_t i=0; i<entries.size(); ++i) {
            const auto& e=entries[i];
            if ((i && e.cycle < entries[i-1].cycle)
                || (e.opcode != IcuControlOpcode::Sync
                    && e.opcode != IcuControlOpcode::WaitEvent
                    && e.opcode != IcuControlOpcode::Notify
                    && e.opcode != kQueueControlMemPacket)
                || (e.opcode != IcuControlOpcode::WaitEvent
                    && e.opcode != kQueueControlMemPacket && e.operand)
                || (e.opcode == kQueueControlMemPacket && !e.operand))
                throw StaticScheduleError("invalid queue control entry or order");
        }
        *this=IcuQueueControlFrontend{};
        entries_=std::move(entries); cache_depth_=cache_depth; latency_=read_latency;
    }
    void begin_cycle()
    {
        while (!pending_.empty() && !pending_.front().remaining) {
            if (cache_.size() == cache_depth_)
                throw StaticScheduleError("queue control response has no reserved credit");
            cache_.push_back(pending_.front().entry); pending_.pop_front();
        }
    }
    void end_cycle()
    {
        if (fetch_ < entries_.size() && cache_.size()+pending_.size() < cache_depth_)
            pending_.push_back({entries_[fetch_++],latency_});
        for (auto& p:pending_) if (p.remaining) --p.remaining;
    }
    void prime_cycle() { begin_cycle(); end_cycle(); }
    bool primed() const noexcept
    { return cache_.size() == std::min(cache_depth_,entries_.size()); }
    bool done() const noexcept { return retired_ == entries_.size(); }
    std::size_t retired() const noexcept { return retired_; }
    std::size_t size() const noexcept { return entries_.size(); }
    const IcuQueueControl* due(std::size_t cycle) const
    {
        if (done()) return nullptr;
        // Only a head-cache miss is diagnosed; the backing vector is never
        // used to execute or bypass the modeled RAM port/latency.
        if (entries_[retired_].cycle < cycle)
            throw StaticScheduleError("queue control boundary was missed");
        if (entries_[retired_].cycle != cycle) return nullptr;
        if (cache_.empty())
            throw StaticScheduleError("queue control cache underflow at dispatch boundary");
        return &cache_.front();
    }
    const IcuQueueControl* cached_due(std::size_t cycle) const noexcept
    { return !cache_.empty() && cache_.front().cycle == cycle ? &cache_.front() : nullptr; }
    void retire()
    {
        if (cache_.empty()) throw StaticScheduleError("queue control retired without a fetched entry");
        cache_.pop_front(); ++retired_;
    }
private:
    struct Pending { IcuQueueControl entry; std::size_t remaining; };
    std::vector<IcuQueueControl> entries_{};
    std::deque<IcuQueueControl> cache_{};
    std::deque<Pending> pending_{};
    std::size_t cache_depth_{0},latency_{0},fetch_{0},retired_{0};
};

} // namespace ftlpu
