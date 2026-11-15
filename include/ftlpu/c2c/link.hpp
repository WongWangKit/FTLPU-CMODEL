#pragma once

#include "ftlpu/c2c/types.hpp"
#include "ftlpu/core/timing.hpp"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <optional>
#include <stdexcept>
#include <utility>

namespace ftlpu {

class C2cLink : public timing::ClockedModule {
private:
    struct SerializingTransfer {
        C2cVector vector{};
        std::size_t bytes_serialized{0};
    };

    struct InFlightTransfer {
        C2cVector vector{};
        std::size_t remaining_cycles{0};
    };

    struct State {
        std::optional<SerializingTransfer> serializing{};
        std::deque<InFlightTransfer> in_flight{};
        std::deque<C2cVector> rx_ready{};
        std::size_t outstanding_vectors{};
        std::size_t cycle{};
    };

public:
    explicit C2cLink(C2cLinkConfig config = {})
        : config_(config)
    {
        config_.validate();
    }

    void reset()
    {
        current_ = {};
        next_.reset();
        pending_send_.reset();
        pending_receive_pops_ = 0;
        low_prepared_ = false;
    }

    const C2cLinkConfig& config() const noexcept { return config_; }
    std::size_t cycle() const noexcept { return current_.cycle; }

    bool can_send() const noexcept
    {
        const auto outstanding = current_.outstanding_vectors
            - std::min(pending_receive_pops_, current_.outstanding_vectors);
        return !pending_send_.has_value()
            && !current_.serializing.has_value()
            && outstanding < config_.rx_fifo_depth_vectors;
    }

    bool receive_ready() const noexcept
    {
        return pending_receive_pops_ < current_.rx_ready.size();
    }
    std::size_t receive_queue_size() const noexcept
    {
        return current_.rx_ready.size() - std::min(
            pending_receive_pops_, current_.rx_ready.size());
    }
    std::size_t outstanding_vector_count() const noexcept
    {
        return current_.outstanding_vectors
            - std::min(pending_receive_pops_, current_.outstanding_vectors)
            + (pending_send_.has_value() ? 1U : 0U);
    }

    void send(C2cVector vector)
    {
        if (!can_send()) {
            throw std::logic_error(
                "C2C Send issued without link credit");
        }
        pending_send_ = std::move(vector);
    }

    const C2cVector& front_received() const
    {
        if (!receive_ready()) {
            throw std::logic_error("C2C receive queue is empty");
        }
        return current_.rx_ready[pending_receive_pops_];
    }

    C2cVector pop_received()
    {
        if (!receive_ready()) {
            throw std::logic_error("C2C receive queue is empty");
        }
        auto vector = current_.rx_ready[pending_receive_pops_];
        ++pending_receive_pops_;
        return vector;
    }

    void tick()
    {
        low();
        validate_posedge();
        posedge();
    }

    void low() override
    {
        if (low_prepared_)
            throw std::logic_error("C2C link LOW executed more than once");
        next_ = current_;
        for (std::size_t pop = 0; pop < pending_receive_pops_; ++pop) {
            if (next_->rx_ready.empty() || next_->outstanding_vectors == 0)
                throw std::logic_error("C2C link pending receive pop lost its head");
            next_->rx_ready.pop_front();
            --next_->outstanding_vectors;
        }
        pending_receive_pops_ = 0;
        if (pending_send_) {
            if (next_->serializing
                || next_->outstanding_vectors >= config_.rx_fifo_depth_vectors)
                throw std::logic_error("C2C pending Send lost link credit");
            next_->serializing = SerializingTransfer {
                std::move(*pending_send_), 0};
            pending_send_.reset();
            ++next_->outstanding_vectors;
        }

        advance_existing_flight(*next_);

        if (next_->serializing.has_value()) {
            auto& transfer = *next_->serializing;
            const auto remaining =
                hw::kPhysicalVectorBytes - transfer.bytes_serialized;
            transfer.bytes_serialized +=
                std::min(config_.beat_bytes, remaining);

            if (transfer.bytes_serialized == hw::kPhysicalVectorBytes) {
                auto completed = InFlightTransfer {
                    std::move(transfer.vector),
                    config_.flight_latency_cycles,
                };
                next_->serializing.reset();
                if (completed.remaining_cycles == 0) {
                    next_->rx_ready.push_back(std::move(completed.vector));
                } else {
                    next_->in_flight.push_back(std::move(completed));
                }
            }
        }

        ++next_->cycle;
        low_prepared_ = true;
    }

    void validate_posedge() const override
    {
        if (!low_prepared_ || !next_)
            throw std::logic_error("C2C link posedge requires one complete LOW");
    }

    void posedge() override
    {
        validate_posedge();
        current_ = std::move(*next_);
        next_.reset();
        low_prepared_ = false;
    }

    void high() override {}

private:

    static void advance_existing_flight(State& state)
    {
        for (auto& transfer : state.in_flight) {
            if (transfer.remaining_cycles != 0) {
                --transfer.remaining_cycles;
            }
        }
        while (!state.in_flight.empty()
               && state.in_flight.front().remaining_cycles == 0) {
            state.rx_ready.push_back(std::move(state.in_flight.front().vector));
            state.in_flight.pop_front();
        }
    }

    C2cLinkConfig config_{};
    State current_{};
    std::optional<State> next_{};
    std::optional<C2cVector> pending_send_{};
    std::size_t pending_receive_pops_{};
    bool low_prepared_{};
};

} // namespace ftlpu
