#pragma once

#include "ftlpu/c2c/ddr4.hpp"
#include "ftlpu/c2c/dma_instruction.hpp"

#include <cstddef>
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
        std::size_t fifo_depth_vectors = 4)
        : ddr4_(ddr4)
        , fifo_depth_vectors_(fifo_depth_vectors)
    {
        if (fifo_depth_vectors_ == 0) {
            throw std::invalid_argument(
                "C2C DMA FIFO depth must be non-zero");
        }
    }

    void reset()
    {
        queue_.clear();
        active_.reset();
        outbound_.clear();
        inbound_.clear();
        pending_request_.reset();
        last_beat_.reset();
        completions_.clear();
        pending_completion_notifications_ = 0;
        cycle_ = 0;
    }

    void issue(C2cDmaInstruction instruction)
    {
        instruction.validate();
        queue_.push_back(std::move(instruction));
    }

    bool idle() const noexcept
    {
        return queue_.empty() && !active_.has_value()
            && !pending_request_.has_value();
    }

    std::size_t queued_instruction_count() const noexcept
    {
        return queue_.size() + (active_.has_value() ? 1U : 0U);
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
    bool can_send() const noexcept
    {
        return outbound_.size() < fifo_depth_vectors_;
    }

    void send(C2cVector vector)
    {
        if (!can_send()) {
            throw std::logic_error("C2C DMA TX FIFO is full");
        }
        outbound_.push_back(std::move(vector));
    }

    // C2C RX-facing FIFO.
    bool receive_ready() const noexcept { return !inbound_.empty(); }

    std::size_t receive_queue_size() const noexcept
    {
        return inbound_.size();
    }

    C2cVector pop_received()
    {
        if (inbound_.empty()) {
            throw std::logic_error("C2C DMA RX FIFO is empty");
        }
        auto vector = std::move(inbound_.front());
        inbound_.pop_front();
        return vector;
    }

    std::size_t outbound_queue_size() const noexcept
    {
        return outbound_.size();
    }

    void tick()
    {
        last_beat_.reset();
        if (!active_.has_value() && !queue_.empty()) {
            active_ = ActiveTransfer {std::move(queue_.front()), 0};
            queue_.pop_front();
        }

        if (active_.has_value()) {
            if (active_->instruction.direction
                == C2cDmaDirection::Ddr4ToC2c) {
                tick_load();
            } else {
                tick_store();
            }
        }
        ++cycle_;
    }

private:
    struct ActiveTransfer {
        C2cDmaInstruction instruction{};
        std::size_t vector_index{0};
    };

    void tick_load()
    {
        auto& active = *active_;
        if (pending_request_.has_value()) {
            if (!ddr4_.read_completion_ready(*pending_request_)
                || inbound_.size() >= fifo_depth_vectors_) {
                return;
            }
            auto completion =
                ddr4_.pop_read_completion(*pending_request_);
            pending_request_.reset();
            completion.vector.vector_tag =
                active.instruction.vector_tag_base + active.vector_index;
            last_beat_ = BeatTrace {
                active.instruction.direction,
                completion.address,
                active.vector_index,
                completion.vector.vector_tag,
            };
            inbound_.push_back(std::move(completion.vector));
            finish_vector();
            return;
        }

        if (inbound_.size() < fifo_depth_vectors_ && ddr4_.can_accept_request()) {
            pending_request_ = ddr4_.request_read(
                active.instruction.vector_address(active.vector_index));
        }
    }

    void tick_store()
    {
        auto& active = *active_;
        if (pending_request_.has_value()) {
            if (!ddr4_.write_completion_ready(*pending_request_)) {
                return;
            }
            const auto completion =
                ddr4_.pop_write_completion(*pending_request_);
            pending_request_.reset();
            last_beat_ = BeatTrace {
                active.instruction.direction,
                completion.address,
                active.vector_index,
                pending_vector_tag_,
            };
            finish_vector();
            return;
        }

        if (!outbound_.empty() && ddr4_.can_accept_request()) {
            auto vector = std::move(outbound_.front());
            outbound_.pop_front();
            pending_vector_tag_ = vector.vector_tag;
            pending_request_ = ddr4_.request_write(
                active.instruction.vector_address(active.vector_index),
                std::move(vector));
        }
    }

    void finish_vector()
    {
        ++active_->vector_index;
        if (active_->vector_index != active_->instruction.vector_count) {
            return;
        }
        completions_.push_back(Completion {
            active_->instruction.direction,
            active_->instruction.ddr4_address,
            active_->instruction.vector_count,
        });
        active_.reset();
        ++pending_completion_notifications_;
    }

    Ddr4Model& ddr4_;
    std::size_t fifo_depth_vectors_{4};
    std::deque<C2cVector> outbound_{};
    std::deque<C2cVector> inbound_{};
    std::deque<C2cDmaInstruction> queue_{};
    std::optional<ActiveTransfer> active_{};
    std::optional<Ddr4Model::RequestId> pending_request_{};
    std::uint64_t pending_vector_tag_{0};
    std::optional<BeatTrace> last_beat_{};
    std::vector<Completion> completions_{};
    std::size_t pending_completion_notifications_{0};
    std::size_t cycle_{0};
};

} // namespace ftlpu
