#pragma once

#include "ftlpu/c2c/types.hpp"
#include "ftlpu/core/timing.hpp"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace ftlpu {

class C2cLink : public timing::ClockedModule {
private:
    friend struct C2cLinkBoundaryFixture; // seed an otherwise unreachable 64-bit terminal state in tests
    struct SerializingTransfer {
        C2cVector vector{};
        std::size_t bytes_serialized{0};
    };

    struct InFlightTransfer {
        C2cVector vector{};
        std::size_t remaining_cycles{0};
    };

public:
    // Fault decisions are sampled on arrival of a complete frame, not per beat.
    enum class FrameResult { Good, Lost, Corrupt, LostAck };
    using FaultSchedule = std::function<FrameResult(std::uint64_t, std::uint64_t, std::size_t)>;
private:
    struct Replay {
        C2cVector payload;
        std::uint64_t seq;
        std::size_t attempt{};
        std::optional<std::size_t> deadline, ack_at;
    };
    struct State {
        std::optional<SerializingTransfer> serializing{};
        std::deque<InFlightTransfer> in_flight{};
        std::deque<C2cVector> rx_ready{};
        std::size_t outstanding_vectors{};
        std::size_t cycle{};
        std::uint64_t issued{}, rx_next{};
        std::size_t retries{}, duplicates{};
        bool error{};
        std::optional<Replay> replay;
        // One entry per consumed but unreturned reservation; bounded by RX depth.
        std::deque<std::pair<std::size_t,std::size_t>> credits;

    };

public:
    explicit C2cLink(C2cLinkConfig config = {})
        : config_(config)
    {
        config_.validate();
    }

    void reset()
    {
        if(epoch_==std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("peer generation exhausted");
        ++epoch_;
        current_ = {};
        next_.reset();
        pending_send_.reset();
        pending_receive_pops_ = 0;
        low_prepared_ = false;
    }

    const C2cLinkConfig& config() const noexcept { return config_; }
    std::size_t cycle() const noexcept { return current_.cycle; }

    void set_peer_fault_schedule(FaultSchedule f) { fault_ = std::move(f); }
    bool peer_error() const noexcept { return current_.error; }
    std::size_t peer_retries() const noexcept { return current_.retries; }
    std::size_t peer_duplicates() const noexcept { return current_.duplicates; }
    bool awaiting_ack() const noexcept { return current_.replay.has_value(); }

    bool can_send() const noexcept
    {
        const auto outstanding = current_.outstanding_vectors
            - std::min(pending_receive_pops_, current_.outstanding_vectors);
        if (config_.retry_timeout)
            return !current_.error && !current_.replay && !pending_send_
                && current_.issued != std::numeric_limits<std::uint64_t>::max()
                && current_.outstanding_vectors < config_.rx_fifo_depth_vectors;
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
        if (config_.retry_timeout) return current_.outstanding_vectors + std::size_t(pending_send_.has_value());
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
            if (!config_.retry_timeout) --next_->outstanding_vectors;
        }
        if(config_.retry_timeout) {
            if (pending_receive_pops_) next_->credits.emplace_back(
                next_->cycle + config_.effective_rx_credit_latency(), pending_receive_pops_);
            while (!next_->credits.empty() && next_->credits.front().first <= next_->cycle) {
                next_->outstanding_vectors -= next_->credits.front().second;
                next_->credits.pop_front();
            }
            auto& r=next_->replay;
            // ACK has priority on a coincident timeout edge.
            if(r && r->ack_at && *r->ack_at<=next_->cycle) r.reset();
            if(r && r->deadline && *r->deadline<=next_->cycle && !next_->error
                && !next_->serializing && next_->in_flight.empty()) {
                if(r->attempt==config_.max_retries) next_->error=true;
                else {
                    ++r->attempt; ++next_->retries; r->deadline.reset();
                    next_->serializing=SerializingTransfer{r->payload,0};
                }
            }
        }
        pending_receive_pops_ = 0;
        if (pending_send_) {
            if (next_->serializing
                || next_->outstanding_vectors >= config_.rx_fifo_depth_vectors)
                throw std::logic_error("C2C pending Send lost link credit");
            if(config_.retry_timeout) next_->replay=Replay{*pending_send_,next_->issued++,0,{},{}};
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
                    arrive(*next_,std::move(completed.vector));
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

    // Stop-and-wait: a returned ACK must precede the retry deadline.
    // Delayed/foreign external ACK injection is not part of this channel model.
    void arrive(State& s,C2cVector vector) {
        if(!config_.retry_timeout) { s.rx_ready.push_back(std::move(vector)); return; }
        auto& r=s.replay.value();
        auto result=fault_?fault_(epoch_,r.seq,r.attempt):FrameResult::Good;
        if(result==FrameResult::Good || result==FrameResult::LostAck) {
            if(r.seq==s.rx_next) { s.rx_ready.push_back(std::move(vector)); ++s.rx_next; }
            else if(r.seq<s.rx_next) ++s.duplicates;
            else throw std::logic_error("stop-and-wait receive gap");
            if(result==FrameResult::Good) r.ack_at=s.cycle+config_.control_latency;
        }
        r.deadline=s.cycle+config_.retry_timeout;
    }
    void advance_existing_flight(State& state)
    {
        for (auto& transfer : state.in_flight) {
            if (transfer.remaining_cycles != 0) {
                --transfer.remaining_cycles;
            }
        }
        while (!state.in_flight.empty()
               && state.in_flight.front().remaining_cycles == 0) {
            arrive(state,std::move(state.in_flight.front().vector));
            state.in_flight.pop_front();
        }
    }

    std::uint64_t epoch_{1};
    FaultSchedule fault_;
    C2cLinkConfig config_{};
    State current_{};
    std::optional<State> next_{};
    std::optional<C2cVector> pending_send_{};
    std::size_t pending_receive_pops_{};
    bool low_prepared_{};
};

} // namespace ftlpu
