#pragma once

#include "ftlpu/c2c/ddr4.hpp"
#include "ftlpu/c2c/dma_instruction.hpp"

#include <cstddef>
#include <array>
#include <deque>
#include <optional>
#include <utility>
#include <vector>

namespace ftlpu {

class C2cDmaEngine {
public:
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
    };

    explicit C2cDmaEngine(
        Ddr4Model& ddr4,
        std::size_t fifo_depth_vectors = 4,
        std::size_t stream_count = 1)
        : ddr4_(ddr4)
        , fifo_depth_vectors_(fifo_depth_vectors)
        , stream_count_(stream_count)
    {
        if (fifo_depth_vectors_ == 0) {
            throw std::invalid_argument(
                "C2C DMA FIFO depth must be non-zero");
        }
        if (stream_count_ == 0
            || stream_count_ > hw::kC2cStreamsPerDirection) {
            throw std::invalid_argument(
                "C2C DMA stream count exceeds the physical fabric");
        }
    }

    void reset()
    {
        for (auto& queue : queues_) queue.clear();
        for (auto& active : active_) active.reset();
        for (auto& fifo : outbound_) fifo.clear();
        for (auto& fifo : inbound_) fifo.clear();
        for (auto& pending : pending_requests_) pending.clear();
        last_beat_.reset();
        completions_.clear();
        pending_completion_notifications_ = 0;
        cycle_ = 0;
    }

    void issue(C2cDmaInstruction instruction)
    {
        instruction.validate();
        require_stream(instruction.stream_index);
        queues_[instruction.stream_index].push_back(std::move(instruction));
    }

    bool idle() const noexcept
    {
        for (std::size_t stream = 0; stream < stream_count_; ++stream) {
            if (!queues_[stream].empty() || active_[stream].has_value()
                || !pending_requests_[stream].empty())
                return false;
        }
        return true;
    }

    std::size_t queued_instruction_count() const noexcept
    {
        std::size_t count = 0;
        for (std::size_t stream = 0; stream < stream_count_; ++stream)
            count += queues_[stream].size()
                + (active_[stream].has_value() ? 1U : 0U);
        return count;
    }

    std::size_t cycle() const noexcept { return cycle_; }
    const std::optional<BeatTrace>& last_beat() const noexcept
    {
        return last_beat_;
    }
    const std::vector<Completion>& completions() const noexcept
    {
        return completions_;
    }

    bool take_completion_notification() noexcept
    {
        if (pending_completion_notifications_ == 0) {
            return false;
        }
        --pending_completion_notifications_;
        return true;
    }

    // C2C TX-facing FIFO.
    bool can_send(std::size_t stream = 0) const noexcept
    {
        return stream < stream_count_
            && outbound_[stream].size() < fifo_depth_vectors_;
    }

    void send(C2cVector vector, std::size_t stream = 0)
    {
        require_stream(stream);
        if (!can_send(stream)) {
            throw std::logic_error("C2C DMA TX FIFO is full");
        }
        outbound_[stream].push_back(std::move(vector));
    }

    // C2C RX-facing FIFO.
    bool receive_ready(std::size_t stream = 0) const noexcept
    {
        return stream < stream_count_ && !inbound_[stream].empty();
    }

    std::size_t receive_queue_size(std::size_t stream = 0) const noexcept
    {
        return stream < stream_count_ ? inbound_[stream].size() : 0;
    }

    C2cVector pop_received(std::size_t stream = 0)
    {
        require_stream(stream);
        if (inbound_[stream].empty()) {
            throw std::logic_error("C2C DMA RX FIFO is empty");
        }
        auto vector = std::move(inbound_[stream].front());
        inbound_[stream].pop_front();
        return vector;
    }

    std::size_t outbound_queue_size(std::size_t stream = 0) const noexcept
    {
        return stream < stream_count_ ? outbound_[stream].size() : 0;
    }

    void tick()
    {
        last_beat_.reset();
        for (std::size_t stream = 0; stream < stream_count_; ++stream) {
            if (!active_[stream].has_value() && !queues_[stream].empty()) {
                active_[stream] = ActiveTransfer {
                    std::move(queues_[stream].front()), 0, 0};
                queues_[stream].pop_front();
            }
            if (!active_[stream].has_value()) continue;
            if (active_[stream]->instruction.direction
                == C2cDmaDirection::Ddr4ToC2c) {
                tick_load(stream);
            } else {
                tick_store(stream);
            }
        }
        ++cycle_;
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

    void tick_load(std::size_t stream)
    {
        auto& active = *active_[stream];
        auto& pending = pending_requests_[stream];
        while (!pending.empty()
            && ddr4_.read_completion_ready(pending.front().id)
            && inbound_[stream].size() < fifo_depth_vectors_) {
            const auto request = pending.front();
            pending.pop_front();
            auto completion = ddr4_.pop_read_completion(request.id);
            completion.vector.vector_tag =
                active.instruction.vector_tag_base + request.vector_index;
            last_beat_ = BeatTrace {
                active.instruction.direction,
                completion.address,
                request.vector_index,
                completion.vector.vector_tag,
            };
            inbound_[stream].push_back(std::move(completion.vector));
            ++active.completed_vectors;
        }
        if (active.next_vector_index < active.instruction.vector_count
            && pending.size() + inbound_[stream].size() < fifo_depth_vectors_
            && ddr4_.can_accept_request()) {
            const auto vector_index = active.next_vector_index++;
            pending.push_back(PendingRequest {
                ddr4_.request_read(
                    active.instruction.vector_address(vector_index)),
                vector_index,
                0});
        }
        finish_transfer_if_complete(stream);
    }

    void tick_store(std::size_t stream)
    {
        auto& active = *active_[stream];
        auto& pending = pending_requests_[stream];
        while (!pending.empty()
            && ddr4_.write_completion_ready(pending.front().id)) {
            const auto completion =
                ddr4_.pop_write_completion(pending.front().id);
            const auto request = pending.front();
            pending.pop_front();
            last_beat_ = BeatTrace {
                active.instruction.direction,
                completion.address,
                request.vector_index,
                request.vector_tag,
            };
            ++active.completed_vectors;
        }
        finish_transfer_if_complete(stream);
        if (!active_[stream].has_value()) return;
        while (!outbound_[stream].empty()
            && pending.size() < fifo_depth_vectors_
            && ddr4_.can_accept_request()) {
            auto vector = std::move(outbound_[stream].front());
            outbound_[stream].pop_front();
            const auto vector_index = active.next_vector_index++;
            const auto tag = vector.vector_tag;
            pending.push_back(PendingRequest {
                ddr4_.request_write(
                    active.instruction.vector_address(vector_index),
                    std::move(vector)),
                vector_index,
                tag});
        }
    }

    void finish_transfer_if_complete(std::size_t stream)
    {
        auto& active = active_[stream];
        if (!active.has_value()
            || active->completed_vectors != active->instruction.vector_count) {
            return;
        }
        completions_.push_back(Completion {
            active->instruction.direction,
            active->instruction.ddr4_address,
            active->instruction.vector_count,
        });
        active.reset();
        ++pending_completion_notifications_;
    }

    void require_stream(std::size_t stream) const
    {
        if (stream >= stream_count_)
            throw std::out_of_range(
                "C2C DMA stream is disabled by hardware configuration");
    }

    Ddr4Model& ddr4_;
    std::size_t fifo_depth_vectors_{4};
    std::size_t stream_count_{1};
    std::array<std::deque<C2cVector>, hw::kC2cStreamsPerDirection> outbound_{};
    std::array<std::deque<C2cVector>, hw::kC2cStreamsPerDirection> inbound_{};
    std::array<std::deque<C2cDmaInstruction>, hw::kC2cStreamsPerDirection> queues_{};
    std::array<std::optional<ActiveTransfer>, hw::kC2cStreamsPerDirection> active_{};
    std::array<std::deque<PendingRequest>, hw::kC2cStreamsPerDirection>
        pending_requests_{};
    std::optional<BeatTrace> last_beat_{};
    std::vector<Completion> completions_{};
    std::size_t pending_completion_notifications_{0};
    std::size_t cycle_{0};
};

} // namespace ftlpu
