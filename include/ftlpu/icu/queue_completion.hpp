#pragma once
#include "ftlpu/icu/common_barrier.hpp"

namespace ftlpu {
// Page completion is a task-scoped latch, separate from Native notification
// tokens. It never consumes, combines, or deduplicates ordinary NOTIFYs.
class IcuQueueCompletionEvents {
public:
    void configure(std::uint64_t generation,const std::vector<IcuCommonEventBinding>& bindings)
    {
        if (!generation || (generation_ && generation<=generation_) || bindings.size()>65535)
            throw StaticScheduleError("invalid queue completion generation or capacity");
        IcuQueueCompletionEvents next;next.generation_=generation;
        for(const auto& binding:bindings) {
            if(!binding.event_id || binding.completion_instances.empty()
                || !next.remaining_.emplace(binding.event_id,binding.completion_instances.size()).second)
                throw StaticScheduleError("invalid queue completion event binding");
            for(const auto instance:binding.completion_instances) {
                if(next.instances_.size()>=65535 || !next.instances_.emplace(instance,Instance{binding.event_id,false}).second)
                    throw StaticScheduleError("invalid queue completion instance binding");
            }
        }
        *this=std::move(next);
    }
    void notify_mem_commit(std::uint64_t generation,std::uint32_t instance)
    {
        if(!generation_) throw StaticScheduleError("queue completion arrived before initialization");
        if(generation<generation_) return;
        if(generation!=generation_) throw StaticScheduleError("queue completion belongs to a future generation");
        const auto found=instances_.find(instance);
        if(found==instances_.end()) throw StaticScheduleError("queue completion has no transport instance");
        if(found->second.completed) return;
        found->second.completed=true;
        --remaining_.at(found->second.event);
    }
    bool ready(std::uint32_t event) const
    {
        const auto found=remaining_.find(event);
        if(found==remaining_.end()) throw StaticScheduleError("unknown queue completion event");
        return found->second==0;
    }
private:
    struct Instance {std::uint32_t event;bool completed;};
    std::uint64_t generation_{0};
    std::unordered_map<std::uint32_t,std::size_t> remaining_{};
    std::unordered_map<std::uint32_t,Instance> instances_{};
};
} // namespace ftlpu
