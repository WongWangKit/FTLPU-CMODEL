#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/icu/instruction.hpp"
#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ftlpu {

struct IcuCommonBarrierEntry {
    std::uint32_t stop_cycle{0};
    std::uint32_t event_id{0};
};

// Each instance names one concrete page load, including repeated loads into
// the same SRAM range. Only visible MEM commits may complete an instance.
struct IcuCommonEventBinding {
    std::uint32_t event_id{0};
    std::vector<std::uint32_t> completion_instances{};
};

class IcuCommonBarrierController {
public:
    enum class Phase { Init, Run, Hold };
    // Protocol bound shared with the 16-bit transport segment namespace.
    static constexpr std::size_t kMaxCompletionInstances = 65535;

    void reset() { *this = IcuCommonBarrierController{}; }

    void configure(std::uint64_t generation,
        std::vector<IcuCommonBarrierEntry> entries,
        const std::vector<IcuCommonEventBinding>& bindings)
    {
        if (started_ || generation == 0
            || (configured_ && generation <= generation_))
            throw StaticScheduleError("common barrier generation is not a fresh pre-launch generation");
        if (hw::kCommonBarrierEncodingVersion != 1
            || entries.size() > hw::kCommonBarrierCapacity)
            throw StaticScheduleError("common barrier table exceeds control RAM capacity");
        if (bindings.size() != entries.size())
            throw StaticScheduleError("common barrier event count does not match table count");
        std::size_t instance_count = 0;
        for (const auto& binding : bindings) {
            if (binding.completion_instances.size()
                > kMaxCompletionInstances - instance_count)
                throw StaticScheduleError("common barrier completion state exceeds 65535 instances");
            instance_count += binding.completion_instances.size();
        }
        std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> events;
        std::unordered_map<std::uint32_t, std::uint32_t> owners;
        for (const auto& binding : bindings) {
            if (binding.event_id == 0 || binding.completion_instances.empty()
                || !events.emplace(binding.event_id, binding.completion_instances).second)
                throw StaticScheduleError("invalid or duplicate common barrier event binding");
            for (const auto instance : binding.completion_instances)
                if (!owners.emplace(instance, binding.event_id).second)
                    throw StaticScheduleError("common barrier completion instance has multiple owners");
        }
        std::unordered_set<std::uint32_t> referenced;
        for (std::size_t i = 0; i < entries.size(); ++i) {
            if ((i != 0 && entries[i - 1].stop_cycle >= entries[i].stop_cycle)
                || !events.contains(entries[i].event_id)
                || !referenced.insert(entries[i].event_id).second)
                throw StaticScheduleError("common barriers require increasing cycles and unique bound events");
        }
        if (referenced.size() != events.size())
            throw StaticScheduleError("common barrier table contains an orphan event binding");
        generation_ = generation;
        entries_ = std::move(entries);
        events_ = std::move(events);
        instance_owners_ = std::move(owners);
        completed_instances_.clear();
        next_ = 0;
        logical_cycle_ = physical_cycle_ = wait_cycles_ = 0;
        configured_ = true;
        phase_ = Phase::Init;
        refresh_cache();
    }

    void notify_mem_commit(std::uint64_t generation, std::uint32_t instance)
    {
        if (!configured_)
            throw StaticScheduleError("common event arrived before task event state was initialized");
        if (generation != generation_) {
            if (generation < generation_) return;
            throw StaticScheduleError("common event belongs to a future task generation");
        }
        if (!instance_owners_.contains(instance))
            throw StaticScheduleError("common event has no linked page completion instance");
        completed_instances_.insert(instance); // Duplicates are idempotent.
    }

    bool barrier_due() const noexcept
    {
        return current_.has_value() && current_->stop_cycle == logical_cycle_
            && (!current_checked_ || (phase_ == Phase::Hold && current_event_ready()));
    }

    bool begin_cycle(bool issue_enabled)
    {
        started_ = true;
        if (current_ && current_->stop_cycle < logical_cycle_)
            throw StaticScheduleError("common barrier deadline was missed");
        if (current_ && current_->stop_cycle == logical_cycle_) {
            current_checked_ = true;
            if (!current_event_ready()) {
                phase_ = Phase::Hold;
                enabled_ = false;
                return false;
            }
            ++next_;
            refresh_cache();
        }
        phase_ = issue_enabled ? Phase::Run : Phase::Hold;
        enabled_ = issue_enabled;
        return enabled_;
    }

    void end_cycle()
    {
        ++physical_cycle_;
        if (enabled_) {
            // The serialized schedule cannot represent a cycle after UINT32_MAX.
            if (configured_ && logical_cycle_ == std::numeric_limits<std::uint32_t>::max())
                throw StaticScheduleError("common computation logical cycle overflow");
            ++logical_cycle_;
        } else {
            ++wait_cycles_;
        }
    }

    std::uint64_t logical_cycle() const noexcept { return logical_cycle_; }
    std::uint64_t physical_cycle() const noexcept { return physical_cycle_; }
    std::uint64_t wait_cycles() const noexcept { return wait_cycles_; }
    std::uint64_t generation() const noexcept { return generation_; }
    bool enabled() const noexcept { return enabled_; }
    bool configured() const noexcept { return configured_; }
    bool holding() const noexcept { return phase_ == Phase::Hold; }
    bool done() const noexcept { return next_ == entries_.size(); }
    Phase phase() const noexcept { return phase_; }

private:
    bool current_event_ready() const noexcept
    {
        if (!current_) return false;
        const auto found = events_.find(current_->event_id);
        if (found == events_.end()) return false;
        for (const auto instance : found->second)
            if (!completed_instances_.contains(instance)) return false;
        return true;
    }

    void refresh_cache()
    {
        current_ = next_ < entries_.size()
            ? std::optional<IcuCommonBarrierEntry>{entries_[next_]} : std::nullopt;
        following_ = next_ + 1 < entries_.size()
            ? std::optional<IcuCommonBarrierEntry>{entries_[next_ + 1]} : std::nullopt;
        current_checked_ = false;
    }

    std::vector<IcuCommonBarrierEntry> entries_{}; // Finite preloaded control RAM.
    std::optional<IcuCommonBarrierEntry> current_{}, following_{};
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> events_{};
    std::unordered_map<std::uint32_t, std::uint32_t> instance_owners_{};
    std::unordered_set<std::uint32_t> completed_instances_{};
    std::size_t next_{0};
    std::uint64_t generation_{0}, logical_cycle_{0}, physical_cycle_{0}, wait_cycles_{0};
    Phase phase_{Phase::Init};
    bool configured_{false}, started_{false}, enabled_{true}, current_checked_{false};
};

} // namespace ftlpu
