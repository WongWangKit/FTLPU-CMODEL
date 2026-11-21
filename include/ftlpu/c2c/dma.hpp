#pragma once

#include "ftlpu/c2c/ddr4.hpp"
#include "ftlpu/c2c/dma_instruction.hpp"
#include "ftlpu/core/timing.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ftlpu {

class C2cDmaEngine : public timing::ClockedModule {
public:
    struct Error {std::uint64_t address{};bool write{};std::size_t transferred_bytes{};};
    static constexpr std::size_t kDescriptorSlots=64;
    struct BeatTrace {
        C2cDmaDirection direction{C2cDmaDirection::Ddr4ToC2c};
        std::uint64_t ddr4_address{0};
        std::size_t vector_index{0};
        std::uint64_t vector_tag{0};
    };

    struct Completion {
        C2cDmaDirection direction{C2cDmaDirection::Ddr4ToC2c};
        std::uint64_t ddr4_address{0};
        std::size_t vector_count{0};
        std::uint32_t sync_tag{0};
    };

    explicit C2cDmaEngine(
        Ddr4Model& ddr4,
        std::size_t fifo_depth_vectors = hw::config::kC2cDmaSizing.rt,
        std::size_t stream_count = 1,
        std::size_t transport_lane_base = 0)
        : ddr4_(ddr4)
        , fifo_depth_vectors_(fifo_depth_vectors)
        , stream_count_(stream_count)
        , transport_lane_base_(transport_lane_base)
    {
        if (fifo_depth_vectors_ == 0) {
            throw std::invalid_argument(
                "C2C DMA FIFO depth must be non-zero");
        }
        if (stream_count_ == 0
            || stream_count_ > hw::kC2cStreamsPerDirection
            || transport_lane_base_ + stream_count_
                > hw::kHemispheres * hw::kC2cStreamsPerDirection) {
            throw std::invalid_argument(
                "C2C DMA stream count exceeds the physical fabric");
        }
    }

    const std::optional<Error>& error() const noexcept {return state_.error;}
    bool error_drained() const noexcept {
        if(!state_.error && !state_.aborting)return false;
        for(const auto& p:state_.pending_requests)if(!p.empty())return false;
        return true;
    }
    void recover_error() {
        if(next_ || !error_drained())throw std::logic_error("DMA error not drained at cycle boundary");
        const auto saved=state_.cycle;
        for(std::size_t n=0;n<stream_count_;++n) {
            ddr4_.axi().resume_client(2*(transport_lane_base_+n));
            ddr4_.axi().resume_client(2*(transport_lane_base_+n)+1);
        }
        reset();state_.cycle=saved;
    }
    void begin_task_abort() {
        if(next_)throw std::logic_error("abort requires cycle boundary");
        state_.aborting=true;
        for(std::size_t n=0;n<stream_count_;++n) {
            ddr4_.axi().abort_client(2*(transport_lane_base_+n));
            ddr4_.axi().abort_client(2*(transport_lane_base_+n)+1);
        }
    }
    bool can_issue() const {
        return !ddr4_.axi().isolated() && !state_.error && !state_.aborting
            && queued_instruction_count()+state_.pending_completion_notifications<kDescriptorSlots
            && queued_instruction_count()+state_.pending_completion_sync_tags.size()<kDescriptorSlots;
    }

    void enable_clocked_mode()
    {
        if (clock_managed_ || state_.cycle != 0 || next_.has_value()
            || has_pending_inputs()) {
            throw std::logic_error(
                "C2C DMA clocked mode must be enabled before execution");
        }
        clock_managed_ = true;
    }

    bool clocked_mode() const noexcept { return clock_managed_; }

    void low() override
    {
        if (!clock_managed_) {
            tick_state(state_);
            return;
        }
        if (next_.has_value())
            throw std::logic_error("C2C DMA LOW executed more than once");
        next_ = state_;
        apply_pending_inputs(*next_);
        tick_state(*next_);
    }

    void validate_posedge() const override
    {
        if (clock_managed_ && !next_.has_value())
            throw std::logic_error(
                "C2C DMA posedge requires one complete LOW");
    }

    void posedge() override
    {
        if (!clock_managed_) return;
        validate_posedge();
        state_ = std::move(*next_);
        next_.reset();
        clear_pending_inputs();
    }

    void high() override {}

    void reset()
    {
        state_ = {};
        next_.reset();
        clear_pending_inputs();
    }

    void issue(C2cDmaInstruction instruction)
    {
        require_input_window("instruction");
        instruction.validate();
        // Reject a permanently inadmissible vector before the first descriptor
        // effect. A split vector owns both 4 KiB fragments until retirement.
        const auto& cfg=ddr4_.config();
        if(instruction.ddr4_address%cfg.beat_bytes ||
           (instruction.vector_count>1 && instruction.address_stride_bytes%cfg.beat_bytes) ||
           instruction.vector_address(instruction.vector_count-1)>UINT64_MAX-hw::kPhysicalVectorBytes)
            throw std::invalid_argument("DMA descriptor address alignment/range");
        if(cfg.request_queue_depth==1 && cfg.beat_bytes<hw::kPhysicalVectorBytes){
            const auto period=4096/std::gcd(std::size_t(4096),instruction.address_stride_bytes);
            for(std::size_t i=0;i<std::min(period,instruction.vector_count);++i)
                if(instruction.vector_address(i)%4096>4096-hw::kPhysicalVectorBytes)
                    throw std::invalid_argument("split DMA vector needs two AXI transaction slots");
        }
        if(!can_issue())throw std::logic_error("DMA descriptor/completion capacity");
        require_stream(instruction.stream_index);
        if (clock_managed_)
            pending_instructions_[instruction.stream_index].push_back(
                std::move(instruction));
        else
            state_.queues[instruction.stream_index].push_back(
                std::move(instruction));
    }

    bool idle() const noexcept
    {
        for (std::size_t stream = 0; stream < stream_count_; ++stream) {
            if (!state_.queues[stream].empty()
                || state_.active[stream].has_value()
                || !state_.pending_requests[stream].empty()
                || !pending_instructions_[stream].empty())
                return false;
        }
        return true;
    }

    bool drained() const noexcept
    {
        if (!idle()) return false;
        for (std::size_t stream = 0; stream < stream_count_; ++stream) {
            if (receive_queue_size(stream) != 0
                || state_.outbound[stream].size()
                    + pending_outbound_[stream].size() != 0)
                return false;
        }
        return true;
    }

    std::size_t queued_instruction_count() const noexcept
    {
        std::size_t count = 0;
        for (std::size_t stream = 0; stream < stream_count_; ++stream) {
            count += state_.queues[stream].size()
                + pending_instructions_[stream].size()
                + (state_.active[stream].has_value() ? 1U : 0U);
        }
        return count;
    }

    std::size_t cycle() const noexcept { return state_.cycle; }
    const std::optional<std::uint64_t>& last_read_request() const noexcept
    {
        return state_.last_read_request;
    }
    // SystemPhase consumers which run after DMA LOW must observe the prepared
    // transaction, while ordinary software queries continue to see Current.
    // These accessors keep that distinction explicit and prevent a second
    // private DMA clock from being hidden inside a caller.
    const std::optional<std::uint64_t>& low_last_read_request() const noexcept
    {
        return cycle_state().last_read_request;
    }
    bool load_credit_wait() const noexcept { return state_.load_credit_wait; }
    bool low_load_credit_wait() const noexcept
    {
        return cycle_state().load_credit_wait;
    }
    std::size_t fifo_depth_vectors() const noexcept
    {
        return fifo_depth_vectors_;
    }
    std::size_t outstanding_reads(std::size_t stream = 0) const noexcept
    {
        return stream < stream_count_
            ? state_.pending_requests[stream].size() : 0;
    }
    const std::optional<BeatTrace>& last_beat() const noexcept
    {
        return state_.last_beat;
    }
    const std::optional<BeatTrace>& low_last_beat() const noexcept
    {
        return cycle_state().last_beat;
    }
    std::size_t low_outstanding_reads(std::size_t stream = 0) const noexcept
    {
        const auto& visible = cycle_state();
        return stream < stream_count_
            ? visible.pending_requests[stream].size() : 0;
    }
    std::size_t low_receive_queue_size(std::size_t stream = 0) const noexcept
    {
        const auto& visible = cycle_state();
        return stream < stream_count_ ? visible.inbound[stream].size() : 0;
    }
    bool low_idle() const noexcept
    {
        const auto& visible = cycle_state();
        for (std::size_t stream = 0; stream < stream_count_; ++stream) {
            if (!visible.queues[stream].empty()
                || visible.active[stream].has_value()
                || !visible.pending_requests[stream].empty())
                return false;
        }
        return true;
    }
    const std::vector<Completion>& completions() const noexcept
    {
        return state_.completions;
    }

    bool take_completion_notification() noexcept
    {
        auto& state = consumption_state();
        if (state.pending_completion_notifications == 0) return false;
        --state.pending_completion_notifications;
        return true;
    }

    std::optional<std::uint32_t> take_completion_sync_tag() noexcept
    {
        auto& state = consumption_state();
        if (state.pending_completion_sync_tags.empty()) return std::nullopt;
        const auto tag = state.pending_completion_sync_tags.front();
        state.pending_completion_sync_tags.pop_front();
        return tag;
    }

    void clear_completion_history() noexcept
    {
        consumption_state().completions.clear();
    }

    bool can_send(std::size_t stream = 0) const noexcept
    {
        return !ddr4_.axi().isolated() && !state_.error && !state_.aborting && stream < stream_count_
            && state_.outbound[stream].size()
                    + pending_outbound_[stream].size()
                < fifo_depth_vectors_;
    }

    void send(C2cVector vector, std::size_t stream = 0)
    {
        require_input_window("TX vector");
        require_stream(stream);
        if (!can_send(stream))
            throw std::logic_error("C2C DMA TX FIFO is full");
        if (clock_managed_)
            pending_outbound_[stream].push_back(std::move(vector));
        else
            state_.outbound[stream].push_back(std::move(vector));
    }

    bool receive_ready(std::size_t stream = 0) const noexcept
    {
        return receive_queue_size(stream) != 0;
    }

    std::size_t receive_queue_size(std::size_t stream = 0) const noexcept
    {
        if (stream >= stream_count_) return 0;
        const auto popped = pending_inbound_pops_[stream];
        return popped <= state_.inbound[stream].size()
            ? state_.inbound[stream].size() - popped : 0;
    }

    C2cVector pop_received(std::size_t stream = 0)
    {
        require_input_window("RX vector pop");
        require_stream(stream);
        if (!receive_ready(stream))
            throw std::logic_error("C2C DMA RX FIFO is empty");
        auto it = state_.inbound[stream].begin();
        std::advance(it,
            static_cast<std::ptrdiff_t>(pending_inbound_pops_[stream]));
        auto vector = *it;
        if (clock_managed_)
            ++pending_inbound_pops_[stream];
        else
            state_.inbound[stream].erase(it);
        return vector;
    }

    const C2cVector& front_received(std::size_t stream = 0) const
    {
        require_stream(stream);
        if (!receive_ready(stream))
            throw std::logic_error("C2C DMA RX FIFO is empty");
        auto it = state_.inbound[stream].begin();
        std::advance(it,
            static_cast<std::ptrdiff_t>(pending_inbound_pops_[stream]));
        return *it;
    }

    std::size_t outbound_queue_size(std::size_t stream = 0) const noexcept
    {
        return stream < stream_count_
            ? state_.outbound[stream].size()
                + pending_outbound_[stream].size()
            : 0;
    }

    void tick()
    {
        if (clock_managed_) {
            low();
            validate_posedge();
            posedge();
            high();
            return;
        }
        tick_state(state_);
    }

private:
    struct ActiveTransfer {
        C2cDmaInstruction instruction{};
        std::size_t next_vector_index{0};
        std::size_t completed_vectors{0};
    };

    struct PendingRequest {
        Ddr4Model::RequestId id{0};
        std::size_t vector_index{0};
        std::uint64_t vector_tag{0};
    };

    struct State {
        std::optional<Error> error;
        bool aborting{};
        std::array<std::deque<C2cVector>,
            hw::kC2cStreamsPerDirection> outbound{};
        std::array<std::deque<C2cVector>,
            hw::kC2cStreamsPerDirection> inbound{};
        std::array<std::deque<C2cDmaInstruction>,
            hw::kC2cStreamsPerDirection> queues{};
        std::array<std::optional<ActiveTransfer>,
            hw::kC2cStreamsPerDirection> active{};
        std::array<std::deque<PendingRequest>,
            hw::kC2cStreamsPerDirection> pending_requests{};
        std::optional<BeatTrace> last_beat{};
        std::vector<Completion> completions{};
        std::size_t pending_completion_notifications{0};
        std::deque<std::uint32_t> pending_completion_sync_tags{};
        std::size_t cycle{0};
        std::optional<std::uint64_t> last_read_request{};
        bool load_credit_wait{false};
    };

    void require_input_window(const char* operation) const
    {
        if (clock_managed_ && next_.has_value())
            throw std::logic_error(std::string("C2C DMA ") + operation
                + " arrived after DMA LOW");
    }

    bool has_pending_inputs() const noexcept
    {
        for (std::size_t stream = 0; stream < stream_count_; ++stream) {
            if (!pending_instructions_[stream].empty()
                || !pending_outbound_[stream].empty()
                || pending_inbound_pops_[stream] != 0)
                return true;
        }
        return false;
    }

    void clear_pending_inputs() noexcept
    {
        for (auto& queue : pending_instructions_) queue.clear();
        for (auto& queue : pending_outbound_) queue.clear();
        pending_inbound_pops_.fill(0);
    }

    void apply_pending_inputs(State& state)
    {
        for (std::size_t stream = 0; stream < stream_count_; ++stream) {
            if (pending_inbound_pops_[stream]
                > state.inbound[stream].size()) {
                throw std::logic_error(
                    "C2C DMA staged RX pops exceed committed occupancy");
            }
            for (std::size_t pop = 0;
                 pop < pending_inbound_pops_[stream]; ++pop)
                state.inbound[stream].pop_front();
            for (const auto& instruction : pending_instructions_[stream])
                state.queues[stream].push_back(instruction);
            for (const auto& vector : pending_outbound_[stream])
                state.outbound[stream].push_back(vector);
        }
    }

    State& consumption_state() noexcept
    {
        return clock_managed_ && next_.has_value() ? *next_ : state_;
    }

    const State& cycle_state() const noexcept
    {
        return clock_managed_ && next_.has_value() ? *next_ : state_;
    }

    void tick_state(State& state)
    {
        state.last_beat.reset();
        state.last_read_request.reset();
        state.load_credit_wait = false;
        for (std::size_t stream = 0; stream < stream_count_; ++stream) {
            if(state.error || state.aborting) {
                // All requests already offered remain live. Consume their
                // completion records without publishing corrupt/stale payload.
                auto& pending=state.pending_requests[stream];
                while(!pending.empty()) {
                    const auto id=pending.front().id;
                    if(ddr4_.read_completion_ready(id))ddr4_.pop_read_completion(id);
                    else if(ddr4_.write_completion_ready(id))ddr4_.pop_write_completion(id);
                    else break;
                    pending.pop_front();
                }
                continue;
            }
            if (!state.active[stream].has_value() && !state.queues[stream].empty()) {
                state.active[stream] = ActiveTransfer {
                    std::move(state.queues[stream].front()), 0, 0};
                state.queues[stream].pop_front();
            }
            if (!state.active[stream].has_value()) continue;
            if (state.active[stream]->instruction.direction
                == C2cDmaDirection::Ddr4ToC2c) {
                tick_load(state, stream);
            } else {
                tick_store(state, stream);
            }
        }
        ++state.cycle;
    }

    void tick_load(State& state, std::size_t stream)
    {
        auto& active = *state.active[stream];
        auto& pending = state.pending_requests[stream];
        while (!pending.empty()
            && ddr4_.read_completion_ready(pending.front().id)
            && state.inbound[stream].size() < fifo_depth_vectors_) {
            const auto request = pending.front();
            pending.pop_front();
            auto completion = ddr4_.pop_read_completion(request.id);
            if(completion.error || completion.aborted) {
                state.error=Error{completion.address,false,completion.transferred_bytes};return;
            }
            completion.vector.vector_tag =
                active.instruction.vector_tag_base + request.vector_index;
            state.last_beat = BeatTrace {
                active.instruction.direction,
                completion.address,
                request.vector_index,
                completion.vector.vector_tag,
            };
            state.inbound[stream].push_back(std::move(completion.vector));
            ++active.completed_vectors;
        }
        if (active.next_vector_index < active.instruction.vector_count
            && !ddr4_.axi().client_stopped(2*(transport_lane_base_+stream))
            && pending.size() + state.inbound[stream].size() < fifo_depth_vectors_
            && ddr4_.can_accept_request(active.instruction.vector_address(active.next_vector_index))) {
            const auto vector_index = active.next_vector_index++;
            state.last_read_request = active.instruction.vector_address(vector_index);
            pending.push_back(PendingRequest {
                ddr4_.request_read(
                    active.instruction.vector_address(vector_index),
                    2 * (transport_lane_base_ + stream)),
                vector_index,
                0});
        }
        else if (active.next_vector_index < active.instruction.vector_count
            && pending.size() + state.inbound[stream].size() >= fifo_depth_vectors_)
            state.load_credit_wait = true;
        finish_transfer_if_complete(state, stream);
    }

    void tick_store(State& state, std::size_t stream)
    {
        auto& active = *state.active[stream];
        auto& pending = state.pending_requests[stream];
        while (!pending.empty()
            && ddr4_.write_completion_ready(pending.front().id)) {
            const auto completion =
                ddr4_.pop_write_completion(pending.front().id);
            const auto request = pending.front();
            pending.pop_front();
            if(completion.error || completion.aborted) {
                state.error=Error{completion.address,true,completion.transferred_bytes};return;
            }
            state.last_beat = BeatTrace {
                active.instruction.direction,
                completion.address,
                request.vector_index,
                request.vector_tag,
            };
            ++active.completed_vectors;
        }
        finish_transfer_if_complete(state, stream);
        if (!state.active[stream].has_value()) return;
        while (!state.outbound[stream].empty()
            && !ddr4_.axi().client_stopped(2*(transport_lane_base_+stream)+1)
            && active.next_vector_index < active.instruction.vector_count
            && pending.size() < fifo_depth_vectors_
            && ddr4_.can_accept_request(active.instruction.vector_address(active.next_vector_index))) {
            auto vector = std::move(state.outbound[stream].front());
            state.outbound[stream].pop_front();
            const auto vector_index = active.next_vector_index++;
            const auto tag = vector.vector_tag;
            pending.push_back(PendingRequest {
                ddr4_.request_write(
                    active.instruction.vector_address(vector_index),
                    std::move(vector),2 * (transport_lane_base_ + stream) + 1),
                vector_index,
                tag});
        }
    }

    void finish_transfer_if_complete(State& state, std::size_t stream)
    {
        auto& active = state.active[stream];
        if (!active.has_value()
            || active->completed_vectors != active->instruction.vector_count) {
            return;
        }
        if(state.completions.size()==kDescriptorSlots)state.completions.erase(state.completions.begin());
        state.completions.push_back(Completion {
            active->instruction.direction,
            active->instruction.ddr4_address,
            active->instruction.vector_count,
            active->instruction.sync_tag,
        });
        active.reset();
        ++state.pending_completion_notifications;
        state.pending_completion_sync_tags.push_back(
            state.completions.back().sync_tag);
    }

    void require_stream(std::size_t stream) const
    {
        if (stream >= stream_count_)
            throw std::out_of_range(
                "C2C DMA stream is disabled by hardware configuration");
    }

    Ddr4Model& ddr4_;
    std::size_t fifo_depth_vectors_{hw::config::kC2cDmaSizing.rt};
    std::size_t stream_count_{1};
    std::size_t transport_lane_base_{0};
    State state_{};
    std::optional<State> next_{};
    std::array<std::deque<C2cDmaInstruction>,
        hw::kC2cStreamsPerDirection> pending_instructions_{};
    std::array<std::deque<C2cVector>,
        hw::kC2cStreamsPerDirection> pending_outbound_{};
    std::array<std::size_t,
        hw::kC2cStreamsPerDirection> pending_inbound_pops_{};
    bool clock_managed_{false};
};

} // namespace ftlpu
