#pragma once

#include "ftlpu/c2c/instruction.hpp"
#include "ftlpu/c2c/link.hpp"
#include "ftlpu/core/stream_port.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace ftlpu {

struct C2cStreamPortMap {
    struct InputEndpoint {
        std::size_t column{0};
        StreamDirection direction{StreamDirection::East};
    };

    struct OutputEndpoint {
        std::size_t column{0};
        StreamDirection direction{StreamDirection::West};
    };

    InputEndpoint tx_input{};
    OutputEndpoint rx_output{};

    static C2cStreamPortMap EastEdge(std::size_t column)
    {
        return C2cStreamPortMap {
            InputEndpoint {column, StreamDirection::East},
            OutputEndpoint {column, StreamDirection::West},
        };
    }

    static C2cStreamPortMap WestEdge(std::size_t column)
    {
        return C2cStreamPortMap {
            InputEndpoint {column, StreamDirection::West},
            OutputEndpoint {column, StreamDirection::East},
        };
    }
};

struct C2cReceiveNotification {
    C2cConsumer consumer{};
    std::size_t stream_index{0};
    std::uint64_t vector_tag{0};
};

class C2cTxSlice {
public:
    explicit C2cTxSlice(
        C2cStreamPortMap::InputEndpoint endpoint,
        std::string name = "C2C TX")
        : endpoint_(endpoint)
        , name_(std::move(name))
    {
    }

    void reset()
    {
        queue_.clear();
        buffer_ = {};
        next_tile_ = 0;
    }

    void issue(C2cInstruction instruction)
    {
        if (instruction.opcode != C2cOpcode::Send) {
            throw std::invalid_argument("C2C TX accepts only Send instructions");
        }
        queue_.push_back(std::move(instruction));
    }

    bool idle() const noexcept { return queue_.empty(); }
    std::size_t queued_instruction_count() const noexcept { return queue_.size(); }
    std::size_t gathered_tile_count() const noexcept { return next_tile_; }

    void evaluate(StreamRegisterFabric& fabric, C2cLink& link)
    {
        if (queue_.empty()) {
            return;
        }

        if (next_tile_ == hw::kTileRows) {
            try_send(link);
            return;
        }

        const auto& instruction = queue_.front();
        auto input = StreamInputPort(
            fabric, endpoint_.column, endpoint_.direction, name_);
        if (!input.segment_valid(next_tile_, instruction.stream_index)) {
            return;
        }

        const auto segment =
            input.consume_segment(next_tile_, instruction.stream_index);
        if (next_tile_ == 0) {
            buffer_.vector_tag = segment.front().vector_tag;
        }
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            buffer_.payload[next_tile_][lane] = segment[lane].data;
        }
        ++next_tile_;
        if (next_tile_ == hw::kTileRows) {
            try_send(link);
        }
    }

private:
    void try_send(C2cLink& link)
    {
        if (!link.can_send()) {
            return;
        }
        link.send(std::move(buffer_));
        queue_.pop_front();
        buffer_ = {};
        next_tile_ = 0;
    }

    C2cStreamPortMap::InputEndpoint endpoint_{};
    std::string name_{};
    std::deque<C2cInstruction> queue_{};
    C2cVector buffer_{};
    std::size_t next_tile_{0};
};

class C2cRxSlice {
public:
    explicit C2cRxSlice(
        C2cStreamPortMap::OutputEndpoint endpoint,
        std::string name = "C2C RX")
        : endpoint_(endpoint)
        , name_(std::move(name))
    {
    }

    void reset()
    {
        queue_.clear();
        vector_.reset();
        next_tile_ = 0;
    }

    void issue(C2cInstruction instruction)
    {
        if (instruction.opcode != C2cOpcode::Receive) {
            throw std::invalid_argument("C2C RX accepts only Receive instructions");
        }
        queue_.push_back(std::move(instruction));
    }

    bool idle() const noexcept { return queue_.empty() && !vector_.has_value(); }
    std::size_t queued_instruction_count() const noexcept { return queue_.size(); }
    std::size_t replayed_tile_count() const noexcept { return next_tile_; }

    std::optional<C2cReceiveNotification> evaluate(
        StreamRegisterFabric& fabric,
        C2cLink& link)
    {
        if (queue_.empty()) {
            return std::nullopt;
        }

        std::optional<C2cReceiveNotification> notification;
        if (!vector_.has_value()) {
            if (!link.receive_ready()) {
                return std::nullopt;
            }
            vector_ = link.pop_received();
            next_tile_ = 0;
            notification = C2cReceiveNotification {
                queue_.front().consumer,
                queue_.front().stream_index,
                vector_->vector_tag,
            };
        }

        const auto& instruction = queue_.front();
        auto output = StreamOutputPort(
            fabric, endpoint_.column, endpoint_.direction, name_);
        output.write_payload_segment(
            next_tile_,
            instruction.stream_index,
            vector_->payload[next_tile_],
            vector_->vector_tag);

        ++next_tile_;
        if (next_tile_ == hw::kTileRows) {
            vector_.reset();
            queue_.pop_front();
            next_tile_ = 0;
        }
        return notification;
    }

private:
    C2cStreamPortMap::OutputEndpoint endpoint_{};
    std::string name_{};
    std::deque<C2cInstruction> queue_{};
    std::optional<C2cVector> vector_{};
    std::size_t next_tile_{0};
};

class C2cEndpoint {
public:
    explicit C2cEndpoint(
        C2cStreamPortMap ports,
        std::string name = "C2C")
        : tx_(ports.tx_input, name + " TX")
        , rx_(ports.rx_output, name + " RX")
    {
    }

    void reset()
    {
        tx_.reset();
        rx_.reset();
    }

    void issue(C2cInstruction instruction)
    {
        if (instruction.opcode == C2cOpcode::Send) {
            tx_.issue(std::move(instruction));
        } else {
            rx_.issue(std::move(instruction));
        }
    }

    C2cTxSlice& tx() noexcept { return tx_; }
    const C2cTxSlice& tx() const noexcept { return tx_; }
    C2cRxSlice& rx() noexcept { return rx_; }
    const C2cRxSlice& rx() const noexcept { return rx_; }

    std::optional<C2cReceiveNotification> evaluate(
        StreamRegisterFabric& fabric,
        C2cLink& tx_link,
        C2cLink& rx_link)
    {
        tx_.evaluate(fabric, tx_link);
        return rx_.evaluate(fabric, rx_link);
    }

private:
    C2cTxSlice tx_;
    C2cRxSlice rx_;
};

} // namespace ftlpu
