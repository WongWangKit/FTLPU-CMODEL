#pragma once

#include "ftlpu/icu/queue_control_pool.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

namespace ftlpu {

// Cycle model of the formal RTL QCTL load path:
// Load Manager -> East/West seven-stage Relay chains -> fourteen regional
// stores.  Runtime Queue reads are combinational views of the owning store;
// there is no shared read port and no per-Queue prefetch Cache in this path.
class IcuQctlDistributedProgramStore {
public:
    static constexpr std::size_t kQueueCount = 228;
    static constexpr std::size_t kMemQueueCount = 208;
    static constexpr std::size_t kMemQueuesPerHemisphere = 104;
    static constexpr std::size_t kRelayCountPerHemisphere = 7;
    static constexpr std::size_t kMemQueuesPerRelay = 16;
    static constexpr std::size_t kMxmLoadBase = 208;
    static constexpr std::size_t kMxmDequantBase = 212;
    static constexpr std::size_t kMxmComputeBase = 216;
    static constexpr std::size_t kMxmParamLoadBase = 220;
    static constexpr std::size_t kMxmBackendBase = 224;

    enum class PacketKind : std::uint8_t { Partition = 0, Record = 1 };

    struct Inputs {
        bool clear{false};
        bool load_mode{true};
        bool partition_valid{false};
        std::size_t partition_queue{0};
        std::size_t partition_base{0};
        std::size_t partition_count{0};
        std::uint64_t partition_generation{0};
        bool load_valid{false};
        std::size_t load_address{0};
        IcuQueueControl load_data{};
    };

    struct Packet {
        PacketKind kind{PacketKind::Partition};
        std::size_t queue{0};
        std::size_t offset{0};
        std::size_t count{0};
        std::uint64_t generation{0};
        IcuQueueControl record{};
    };

    struct Partition {
        std::size_t base{0};
        std::size_t count{0};
        std::uint64_t generation{0};
    };

    struct Head {
        bool configured{false};
        bool loaded{false};
        bool valid{false};
        bool error{false};
        IcuQueueControl record{};
    };

    struct RelaySignals {
        bool input_valid{false};
        bool input_ready{false};
        bool input_fire{false};
        bool local_fire{false};
        bool output_valid{false};
        bool output_ready{false};
        std::size_t fifo_level{0};
        bool fault{false};
        std::uint8_t fault_code{0};
    };

    struct Preview {
        bool partition_ready{false};
        bool load_ready{false};
        bool partition_fire{false};
        bool load_fire{false};
        bool relay_packet_valid{false};
        bool relay_packet_ready{false};
        Packet relay_packet{};
        bool manager_idle{true};
        bool manager_fault{false};
        std::uint8_t manager_fault_code{0};
        bool east_idle{true};
        bool west_idle{true};
        bool load_complete{false};
        bool fault{false};
        std::uint8_t fault_code{0};
        std::vector<bool> configure_valid{};
        std::vector<std::uint32_t> configure_count{};
        std::vector<std::uint64_t> configure_generation{};
        std::array<std::array<RelaySignals, kRelayCountPerHemisphere>, 2> relays{};
    };

    void configure(std::size_t total_capacity, std::size_t queue_count,
        std::size_t entry_limit, std::size_t region_depth = 64,
        std::size_t relay_buffer_depth = 2)
    {
        if (queue_count != kQueueCount || total_capacity == 0
            || entry_limit == 0 || region_depth == 0
            || relay_buffer_depth == 0)
            throw StaticScheduleError("invalid distributed QCTL geometry");
        total_capacity_ = total_capacity;
        entry_limit_ = entry_limit;
        region_depth_ = region_depth;
        relay_buffer_depth_ = relay_buffer_depth;
        manager_directory_.assign(queue_count, std::nullopt);
        logical_records_.assign(total_capacity, std::nullopt);
        for (std::size_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
            for (std::size_t relay = 0; relay < kRelayCountPerHemisphere; ++relay) {
                auto& state = relays_[hemisphere][relay];
                state.directory.assign(
                    relay + 1 == kRelayCountPerHemisphere ? 18 : 16,
                    std::nullopt);
                state.records.assign(region_depth, std::nullopt);
            }
        }
        reset();
    }

    void reset()
    {
        allocated_count_ = 0;
        std::fill(manager_directory_.begin(), manager_directory_.end(), std::nullopt);
        std::fill(logical_records_.begin(), logical_records_.end(), std::nullopt);
        manager_fault_ = false;
        manager_fault_code_ = 0;
        for (auto& hemisphere : relays_) {
            for (auto& relay : hemisphere) {
                std::fill(relay.directory.begin(), relay.directory.end(), std::nullopt);
                std::fill(relay.records.begin(), relay.records.end(), std::nullopt);
                relay.used_entries = 0;
                relay.fifo.clear();
                relay.fault = false;
                relay.fault_code = 0;
            }
        }
        preview_ = make_empty_preview();
        cycle_open_ = false;
    }

    const Preview& begin_cycle(const Inputs& inputs)
    {
        if (cycle_open_)
            throw StaticScheduleError("distributed QCTL load cycle already open");
        pending_inputs_ = inputs;
        preview_ = evaluate(inputs);
        cycle_open_ = true;
        return preview_;
    }

    void complete_cycle()
    {
        if (!cycle_open_)
            throw StaticScheduleError("distributed QCTL load cycle has no begin");
        const auto inputs = pending_inputs_;
        const auto actions = build_chain_actions(preview_.relay_packet_valid,
            preview_.relay_packet);
        if (inputs.clear) {
            reset();
            return;
        }

        commit_manager(inputs, preview_);
        for (std::size_t hemisphere = 0; hemisphere < 2; ++hemisphere)
            commit_chain(hemisphere, actions[hemisphere]);
        cycle_open_ = false;
    }

    const Preview& preview() const noexcept { return preview_; }

    std::optional<Partition> partition(std::size_t queue) const
    {
        check_queue(queue);
        return manager_directory_[queue];
    }

    std::optional<IcuQueueControl> committed_record(std::size_t address) const
    {
        if (address >= logical_records_.size()) return std::nullopt;
        return logical_records_[address];
    }

    bool configured(std::size_t queue) const
    {
        return local_partition(queue).has_value();
    }

    bool loaded(std::size_t queue) const
    {
        const auto* relay = owning_relay(queue);
        const auto local = local_queue_index(queue);
        if (local >= relay->directory.size() || !relay->directory[local]) return false;
        const auto& descriptor = *relay->directory[local];
        for (std::size_t offset = 0; offset < descriptor.count; ++offset) {
            const auto address = descriptor.base + offset;
            if (address >= relay->records.size() || !relay->records[address]) return false;
        }
        return true;
    }

    Head head(std::size_t queue, std::size_t index,
        std::uint64_t generation) const
    {
        check_queue(queue);
        Head result{};
        const auto* relay = owning_relay(queue);
        const auto local = local_queue_index(queue);
        if (local >= relay->directory.size() || !relay->directory[local]) return result;
        const auto& descriptor = *relay->directory[local];
        result.configured = true;
        result.loaded = loaded(queue);
        if (descriptor.count == 0) return result;
        const auto address = descriptor.base + index;
        result.error = generation != descriptor.generation
            || index >= descriptor.count || address >= relay->records.size()
            || !relay->records[address].has_value();
        if (!result.error) {
            result.valid = true;
            result.record = *relay->records[address];
        }
        return result;
    }

    bool load_complete() const
    {
        if (manager_fault_ || any_relay_fault()) return false;
        for (const auto& hemisphere : relays_)
            for (const auto& relay : hemisphere)
                if (!relay.fifo.empty()) return false;
        for (std::size_t queue = 0; queue < manager_directory_.size(); ++queue)
            if (configured(queue) && !loaded(queue)) return false;
        return true;
    }

    bool fault() const noexcept { return manager_fault_ || any_relay_fault(); }
    std::uint8_t fault_code() const noexcept
    {
        if (manager_fault_) return static_cast<std::uint8_t>(0x10 | manager_fault_code_);
        for (std::size_t relay = 0; relay < kRelayCountPerHemisphere; ++relay)
            if (relays_[0][relay].fault)
                return static_cast<std::uint8_t>(0x40 | relays_[0][relay].fault_code);
        for (std::size_t relay = 0; relay < kRelayCountPerHemisphere; ++relay)
            if (relays_[1][relay].fault)
                return static_cast<std::uint8_t>(0x50 | relays_[1][relay].fault_code);
        return 0;
    }

private:
    struct RelayState {
        std::vector<std::optional<Partition>> directory{};
        std::vector<std::optional<IcuQueueControl>> records{};
        std::size_t used_entries{0};
        std::deque<Packet> fifo{};
        bool fault{false};
        std::uint8_t fault_code{0};
    };

    struct RelayAction {
        bool input_valid{false};
        Packet input{};
        bool input_ready{false};
        bool input_fire{false};
        bool local{false};
        bool enqueue{false};
        bool dequeue{false};
    };

    Preview make_empty_preview() const
    {
        Preview result{};
        result.configure_valid.assign(manager_directory_.size(), false);
        result.configure_count.assign(manager_directory_.size(), 0);
        result.configure_generation.assign(manager_directory_.size(), 0);
        return result;
    }

    Preview evaluate(const Inputs& inputs) const
    {
        Preview result = make_empty_preview();
        const bool partition_invalid = invalid_partition(inputs);
        const bool partition_overlap = overlapping_partition(inputs);
        const bool partition_capacity = allocated_count_ + inputs.partition_count
            > total_capacity_;
        const auto owner = find_owner(inputs.load_address);

        Packet packet{};
        bool packet_valid = false;
        if (inputs.load_mode && !manager_fault_ && inputs.partition_valid
            && !partition_invalid && !partition_overlap && !partition_capacity) {
            packet_valid = true;
            packet.kind = PacketKind::Partition;
            packet.queue = inputs.partition_queue;
            packet.count = inputs.partition_count;
            packet.generation = inputs.partition_generation;
        } else if (inputs.load_mode && !manager_fault_
            && !inputs.partition_valid && inputs.load_valid
            && owner.found && !owner.multiple) {
            packet_valid = true;
            packet.kind = PacketKind::Record;
            packet.queue = owner.queue;
            packet.offset = owner.offset;
            packet.generation = manager_directory_[owner.queue]->generation;
            packet.record = inputs.load_data;
        }
        const bool relay_ready = !packet_valid
            || chain_entry_ready(queue_hemisphere(packet.queue), packet);
        result.partition_ready = inputs.load_mode && !manager_fault_
            && (partition_invalid || partition_overlap || partition_capacity
                || relay_ready);
        result.load_ready = inputs.load_mode && !manager_fault_
            && !inputs.partition_valid
            && ((!owner.found || owner.multiple) || relay_ready);
        result.partition_fire = inputs.partition_valid && result.partition_ready;
        result.load_fire = inputs.load_valid && result.load_ready;
        result.relay_packet_valid = packet_valid;
        result.relay_packet_ready = relay_ready;
        result.relay_packet = packet;
        result.manager_idle = !inputs.partition_valid && !inputs.load_valid
            && !packet_valid;
        result.manager_fault = manager_fault_;
        result.manager_fault_code = manager_fault_code_;

        const auto actions = build_chain_actions(packet_valid, packet);
        for (std::size_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
            bool idle = !packet_valid || queue_hemisphere(packet.queue) != hemisphere;
            for (std::size_t relay = 0; relay < kRelayCountPerHemisphere; ++relay) {
                const auto& state = relays_[hemisphere][relay];
                const auto& action = actions[hemisphere][relay];
                auto& signals = result.relays[hemisphere][relay];
                signals.input_valid = action.input_valid;
                signals.input_ready = action.input_ready;
                signals.input_fire = action.input_fire;
                signals.local_fire = action.input_fire && action.local;
                signals.output_valid = !state.fifo.empty();
                signals.output_ready = relay + 1 < kRelayCountPerHemisphere
                    ? actions[hemisphere][relay + 1].input_ready
                    : true;
                signals.fifo_level = state.fifo.size();
                signals.fault = state.fault;
                signals.fault_code = state.fault_code;
                idle = idle && state.fifo.empty() && !action.input_valid;
                if (action.input_fire && action.local
                    && action.input.kind == PacketKind::Partition
                    && valid_local_partition(state, action.input)) {
                    result.configure_valid[action.input.queue] = true;
                    result.configure_count[action.input.queue] =
                        static_cast<std::uint32_t>(action.input.count);
                    result.configure_generation[action.input.queue] =
                        action.input.generation;
                }
            }
            if (hemisphere == 0) result.east_idle = idle;
            else result.west_idle = idle;
        }
        result.fault = fault();
        result.fault_code = fault_code();
        result.load_complete = result.manager_idle && result.east_idle
            && result.west_idle && load_complete();
        return result;
    }

    struct Owner {
        bool found{false};
        bool multiple{false};
        std::size_t queue{0};
        std::size_t offset{0};
    };

    Owner find_owner(std::size_t address) const
    {
        Owner result{};
        for (std::size_t queue = 0; queue < manager_directory_.size(); ++queue) {
            if (!manager_directory_[queue]) continue;
            const auto& descriptor = *manager_directory_[queue];
            if (address < descriptor.base || address >= descriptor.base + descriptor.count)
                continue;
            if (result.found) result.multiple = true;
            else {
                result.found = true;
                result.queue = queue;
                result.offset = address - descriptor.base;
            }
        }
        return result;
    }

    bool invalid_partition(const Inputs& inputs) const
    {
        return inputs.partition_queue >= manager_directory_.size()
            || inputs.partition_generation == 0
            || inputs.partition_count > entry_limit_
            || inputs.partition_base > total_capacity_
            || inputs.partition_count > total_capacity_ - inputs.partition_base
            || (inputs.partition_queue < manager_directory_.size()
                && manager_directory_[inputs.partition_queue].has_value());
    }

    bool overlapping_partition(const Inputs& inputs) const
    {
        if (inputs.partition_count == 0) return false;
        for (const auto& other : manager_directory_) {
            if (!other || other->count == 0) continue;
            if (inputs.partition_base < other->base + other->count
                && other->base < inputs.partition_base + inputs.partition_count)
                return true;
        }
        return false;
    }

    void commit_manager(const Inputs& inputs, const Preview& preview)
    {
        if (preview.partition_fire) {
            if (invalid_partition(inputs)) {
                set_manager_fault(1);
            } else if (overlapping_partition(inputs)) {
                set_manager_fault(2);
            } else if (allocated_count_ + inputs.partition_count > total_capacity_) {
                set_manager_fault(3);
            } else {
                manager_directory_[inputs.partition_queue] = Partition{
                    inputs.partition_base, inputs.partition_count,
                    inputs.partition_generation};
                allocated_count_ += inputs.partition_count;
            }
        } else if (preview.load_fire) {
            const auto owner = find_owner(inputs.load_address);
            if (!owner.found) set_manager_fault(4);
            else if (owner.multiple) set_manager_fault(5);
            else if (inputs.load_address < logical_records_.size())
                logical_records_[inputs.load_address] = inputs.load_data;
        }
    }

    using ChainActions = std::array<std::array<RelayAction,
        kRelayCountPerHemisphere>, 2>;

    ChainActions build_chain_actions(bool packet_valid, const Packet& packet) const
    {
        ChainActions result{};
        for (std::size_t hemisphere = 0; hemisphere < 2; ++hemisphere) {
            for (std::size_t relay = 0; relay < kRelayCountPerHemisphere; ++relay) {
                auto& action = result[hemisphere][relay];
                if (relay == 0) {
                    action.input_valid = packet_valid
                        && queue_hemisphere(packet.queue) == hemisphere;
                    action.input = packet;
                } else if (!relays_[hemisphere][relay - 1].fifo.empty()) {
                    action.input_valid = true;
                    action.input = relays_[hemisphere][relay - 1].fifo.front();
                }
                action.local = action.input_valid
                    && queue_hemisphere(action.input.queue) == hemisphere
                    && queue_relay_index(action.input.queue) == relay;
                action.input_ready = !relays_[hemisphere][relay].fault
                    && (action.local || relay == kRelayCountPerHemisphere - 1
                        || relays_[hemisphere][relay].fifo.size()
                            < relay_buffer_depth_);
                action.input_fire = action.input_valid && action.input_ready;
                action.enqueue = action.input_fire && !action.local
                    && relay != kRelayCountPerHemisphere - 1;
            }
            for (std::size_t relay = 0; relay + 1 < kRelayCountPerHemisphere;
                 ++relay) {
                result[hemisphere][relay].dequeue =
                    !relays_[hemisphere][relay].fifo.empty()
                    && result[hemisphere][relay + 1].input_ready;
            }
        }
        return result;
    }

    bool chain_entry_ready(std::size_t hemisphere, const Packet& packet) const
    {
        const auto& relay = relays_[hemisphere][0];
        const bool local = queue_relay_index(packet.queue) == 0;
        return !relay.fault && (local || relay.fifo.size() < relay_buffer_depth_);
    }

    void commit_chain(std::size_t hemisphere,
        const std::array<RelayAction, kRelayCountPerHemisphere>& actions)
    {
        for (std::size_t relay_index = 0;
             relay_index < kRelayCountPerHemisphere; ++relay_index) {
            auto& state = relays_[hemisphere][relay_index];
            const auto& action = actions[relay_index];
            if (action.dequeue && !state.fifo.empty()) state.fifo.pop_front();
            if (action.enqueue) state.fifo.push_back(action.input);
            if (!action.input_fire) continue;
            if (!action.local) {
                if (relay_index == kRelayCountPerHemisphere - 1)
                    set_relay_fault(state, 7);
                continue;
            }
            if (action.input.kind == PacketKind::Partition)
                commit_local_partition(state, action.input);
            else commit_local_record(state, action.input);
        }
    }

    bool valid_local_partition(const RelayState& state,
        const Packet& packet) const
    {
        const auto local = local_queue_index(packet.queue);
        return local < state.directory.size() && packet.generation != 0
            && packet.count <= entry_limit_ && !state.directory[local]
            && state.used_entries + packet.count <= region_depth_;
    }

    void commit_local_partition(RelayState& state, const Packet& packet)
    {
        const auto local = local_queue_index(packet.queue);
        if (local >= state.directory.size() || packet.generation == 0
            || packet.count > entry_limit_ || state.directory[local]) {
            set_relay_fault(state, 1);
            return;
        }
        if (state.used_entries + packet.count > region_depth_) {
            set_relay_fault(state, 2);
            return;
        }
        state.directory[local] = Partition{state.used_entries, packet.count,
            packet.generation};
        state.used_entries += packet.count;
    }

    void commit_local_record(RelayState& state, const Packet& packet)
    {
        const auto local = local_queue_index(packet.queue);
        if (local >= state.directory.size() || !state.directory[local]) {
            set_relay_fault(state, 3);
            return;
        }
        const auto& descriptor = *state.directory[local];
        if (packet.generation != descriptor.generation) {
            set_relay_fault(state, 4);
            return;
        }
        const auto address = descriptor.base + packet.offset;
        if (packet.offset >= descriptor.count || address >= state.records.size()) {
            set_relay_fault(state, 5);
            return;
        }
        if (state.records[address]) {
            set_relay_fault(state, 6);
            return;
        }
        state.records[address] = packet.record;
    }

    void set_manager_fault(std::uint8_t code)
    {
        if (!manager_fault_) {
            manager_fault_ = true;
            manager_fault_code_ = code;
        }
    }

    static void set_relay_fault(RelayState& state, std::uint8_t code)
    {
        if (!state.fault) {
            state.fault = true;
            state.fault_code = code;
        }
    }

    bool any_relay_fault() const noexcept
    {
        for (const auto& hemisphere : relays_)
            for (const auto& relay : hemisphere)
                if (relay.fault) return true;
        return false;
    }

    const RelayState* owning_relay(std::size_t queue) const
    {
        check_queue(queue);
        return &relays_[queue_hemisphere(queue)][queue_relay_index(queue)];
    }

    std::optional<Partition> local_partition(std::size_t queue) const
    {
        const auto* relay = owning_relay(queue);
        const auto local = local_queue_index(queue);
        if (local >= relay->directory.size()) return std::nullopt;
        return relay->directory[local];
    }

    static std::size_t queue_hemisphere(std::size_t queue)
    {
        if (queue < kMemQueueCount)
            return queue >= kMemQueuesPerHemisphere;
        std::size_t mxm = 0;
        if (queue < kMxmDequantBase) mxm = queue - kMxmLoadBase;
        else if (queue < kMxmComputeBase) mxm = queue - kMxmDequantBase;
        else if (queue < kMxmParamLoadBase)
            mxm = queue - kMxmComputeBase;
        else if (queue < kMxmBackendBase)
            mxm = queue - kMxmParamLoadBase;
        else mxm = queue - kMxmBackendBase;
        return mxm >= 2;
    }

    static std::size_t queue_relay_index(std::size_t queue)
    {
        if (queue < kMemQueueCount)
            return (queue % kMemQueuesPerHemisphere) / kMemQueuesPerRelay;
        return kRelayCountPerHemisphere - 1;
    }

    static std::size_t local_queue_index(std::size_t queue)
    {
        if (queue < kMemQueueCount)
            return (queue % kMemQueuesPerHemisphere) % kMemQueuesPerRelay;
        std::size_t mxm = 0;
        std::size_t stream = 0;
        if (queue < kMxmDequantBase) {
            mxm = queue - kMxmLoadBase;
            stream = 0;
        } else if (queue < kMxmComputeBase) {
            mxm = queue - kMxmDequantBase;
            stream = 1;
        } else if (queue < kMxmParamLoadBase) {
            mxm = queue - kMxmComputeBase;
            stream = 2;
        } else if (queue < kMxmBackendBase) {
            mxm = queue - kMxmParamLoadBase;
            stream = 3;
        } else {
            mxm = queue - kMxmBackendBase;
            stream = 4;
        }
        return 8 + (mxm % 2) * 5 + stream;
    }

    void check_queue(std::size_t queue) const
    {
        if (queue >= manager_directory_.size())
            throw StaticScheduleError("distributed QCTL Queue index out of range");
    }

    std::size_t total_capacity_{0};
    std::size_t entry_limit_{0};
    std::size_t region_depth_{0};
    std::size_t relay_buffer_depth_{0};
    std::size_t allocated_count_{0};
    std::vector<std::optional<Partition>> manager_directory_{};
    std::vector<std::optional<IcuQueueControl>> logical_records_{};
    std::array<std::array<RelayState, kRelayCountPerHemisphere>, 2> relays_{};
    bool manager_fault_{false};
    std::uint8_t manager_fault_code_{0};
    bool cycle_open_{false};
    Inputs pending_inputs_{};
    Preview preview_{};
};

} // namespace ftlpu
