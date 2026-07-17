#pragma once

#include "ftlpu/c2c/types.hpp"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <optional>
#include <stdexcept>
#include <utility>

namespace ftlpu {

// One unidirectional physical C2C link. Instantiate a second link for the
// opposite direction when modeling a bidirectional chip-to-chip connection.
class C2cLink {
public:
    explicit C2cLink(C2cLinkConfig config = {})
        : config_(config)
    {
        config_.validate();
    }

    const C2cLinkConfig& config() const noexcept
    {
        return config_;
    }

    std::size_t cycle() const noexcept
    {
        return cycle_;
    }

    bool can_send() const noexcept
    {
        return !serializing_.has_value();
    }

    bool receive_ready() const noexcept
    {
        return !rx_ready_.empty();
    }

    std::size_t receive_queue_size() const noexcept
    {
        return rx_ready_.size();
    }

    void send(C2cVector vector)
    {
        if (!can_send()) {
            throw std::logic_error(
                "C2C Send issued while the physical link is still serializing");
        }
        serializing_ = SerializingTransfer{std::move(vector), 0};
    }

    const C2cVector& front_received() const
    {
        if (rx_ready_.empty()) {
            throw std::logic_error(
                "C2C Receive issued before a complete vector arrived");
        }
        return rx_ready_.front();
    }

    void pop_received()
    {
        if (rx_ready_.empty()) {
            throw std::logic_error("C2C receive queue is empty");
        }
        rx_ready_.pop_front();
    }

    // Advance serialization and fixed flight delay by one core cycle.
    // Recommended system order:
    //   TX evaluate -> RX evaluate -> topology routes -> fabric commit -> link.tick
    // This makes a one-cycle 320-byte transfer visible to RX on the next cycle.
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
                InFlightTransfer completed{
                    std::move(transfer.vector),
                    config_.flight_latency_cycles,
                };
                serializing_.reset();

                if (completed.remaining_cycles == 0) {
                    push_received(std::move(completed.vector));
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
            if (transfer.remaining_cycles > 0) {
                --transfer.remaining_cycles;
            }
        }

        while (!in_flight_.empty() &&
               in_flight_.front().remaining_cycles == 0) {
            auto vector = std::move(in_flight_.front().vector);
            in_flight_.pop_front();
            push_received(std::move(vector));
        }
    }

    void push_received(C2cVector vector)
    {
        if (rx_ready_.size() >= config_.rx_fifo_depth_vectors) {
            throw std::logic_error(
                "C2C RX FIFO overflow; the static Receive schedule is too late");
        }
        rx_ready_.push_back(std::move(vector));
    }

    C2cLinkConfig config_{};
    std::optional<SerializingTransfer> serializing_{};
    std::deque<InFlightTransfer> in_flight_{};
    std::deque<C2cVector> rx_ready_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
