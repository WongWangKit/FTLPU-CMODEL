#pragma once

#include "ftlpu/c2c/types.hpp"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <optional>
#include <stdexcept>
#include <utility>

namespace ftlpu {

class C2cLink {
public:
    explicit C2cLink(C2cLinkConfig config = {})
        : config_(config)
    {
        config_.validate();
    }

    void reset()
    {
        serializing_.reset();
        in_flight_.clear();
        rx_ready_.clear();
        outstanding_vectors_ = 0;
        cycle_ = 0;
    }

    const C2cLinkConfig& config() const noexcept { return config_; }
    std::size_t cycle() const noexcept { return cycle_; }

    bool can_send() const noexcept
    {
        return !serializing_.has_value()
            && outstanding_vectors_ < config_.rx_fifo_depth_vectors;
    }

    bool receive_ready() const noexcept { return !rx_ready_.empty(); }
    std::size_t receive_queue_size() const noexcept { return rx_ready_.size(); }
    std::size_t outstanding_vector_count() const noexcept
    {
        return outstanding_vectors_;
    }

    void send(C2cVector vector)
    {
        if (!can_send()) {
            throw std::logic_error(
                "C2C Send issued without link credit");
        }
        serializing_ = SerializingTransfer {std::move(vector), 0};
        ++outstanding_vectors_;
    }

    const C2cVector& front_received() const
    {
        if (rx_ready_.empty()) {
            throw std::logic_error("C2C receive queue is empty");
        }
        return rx_ready_.front();
    }

    C2cVector pop_received()
    {
        if (rx_ready_.empty()) {
            throw std::logic_error("C2C receive queue is empty");
        }
        auto vector = std::move(rx_ready_.front());
        rx_ready_.pop_front();
        --outstanding_vectors_;
        return vector;
    }

    void tick()
    {
        advance_existing_flight();

        if (serializing_.has_value()) {
            auto& transfer = *serializing_;
            const auto remaining =
                hw::kPhysicalVectorBytes - transfer.bytes_serialized;
            transfer.bytes_serialized +=
                std::min(config_.beat_bytes, remaining);

            if (transfer.bytes_serialized == hw::kPhysicalVectorBytes) {
                auto completed = InFlightTransfer {
                    std::move(transfer.vector),
                    config_.flight_latency_cycles,
                };
                serializing_.reset();
                if (completed.remaining_cycles == 0) {
                    rx_ready_.push_back(std::move(completed.vector));
                } else {
                    in_flight_.push_back(std::move(completed));
                }
            }
        }

        ++cycle_;
    }

private:
    struct SerializingTransfer {
        C2cVector vector{};
        std::size_t bytes_serialized{0};
    };

    struct InFlightTransfer {
        C2cVector vector{};
        std::size_t remaining_cycles{0};
    };

    void advance_existing_flight()
    {
        for (auto& transfer : in_flight_) {
            if (transfer.remaining_cycles != 0) {
                --transfer.remaining_cycles;
            }
        }
        while (!in_flight_.empty()
               && in_flight_.front().remaining_cycles == 0) {
            rx_ready_.push_back(std::move(in_flight_.front().vector));
            in_flight_.pop_front();
        }
    }

    C2cLinkConfig config_{};
    std::optional<SerializingTransfer> serializing_{};
    std::deque<InFlightTransfer> in_flight_{};
    std::deque<C2cVector> rx_ready_{};
    std::size_t outstanding_vectors_{0};
    std::size_t cycle_{0};
};

} // namespace ftlpu
