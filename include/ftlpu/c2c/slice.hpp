#pragma once

#include "ftlpu/c2c/instruction.hpp"
#include "ftlpu/c2c/types.hpp"
#include "ftlpu/core/stream_port.hpp"

#include <algorithm>
#include <array>
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
    C2cVector vector{};
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
        pipeline_ = {};
        completed_.clear();
    }

    void issue(C2cInstruction instruction)
    {
        if (instruction.opcode != C2cOpcode::Send) {
            throw std::invalid_argument("C2C TX accepts only Send instructions");
        }
        queue_.push_back(std::move(instruction));
    }

    bool idle() const noexcept
    {
        return queue_.empty() && completed_.empty()
            && std::none_of(
                pipeline_.begin(), pipeline_.end(),
                [](const auto& stage) { return stage.has_value(); });
    }
    std::size_t queued_instruction_count() const noexcept { return queue_.size(); }
    std::size_t gathered_tile_count() const noexcept
    {
        return static_cast<std::size_t>(std::count_if(
            pipeline_.begin(), pipeline_.end(),
            [](const auto& stage) { return stage.has_value(); }));
    }

    template <typename Transport>
    void evaluate(
        StreamRegisterFabric& fabric,
        Transport& external)
    {
        if (!completed_.empty() && external.can_send()) {
            external.send(std::move(completed_.front()));
            completed_.pop_front();
        }

        if (!queue_.empty() && !pipeline_[0].has_value()) {
            pipeline_[0] = ActiveSend {std::move(queue_.front()), {}};
            queue_.pop_front();
        }

        auto input = StreamInputPort(
            fabric, endpoint_.column, endpoint_.direction, name_);
        auto advance = std::array<bool, hw::kTileRows> {};
        for (std::size_t tile = hw::kTileRows; tile-- > 0;) {
            auto& stage = pipeline_[tile];
            if (!stage.has_value()) {
                continue;
            }
            const auto downstream_ready = tile + 1 == hw::kTileRows
                ? completed_.empty()
                : !pipeline_[tile + 1].has_value()
                    || advance[tile + 1];
            advance[tile] = downstream_ready
                && input.segment_valid(
                    tile, stage->instruction.stream_index);
        }

        auto next_pipeline =
            std::array<std::optional<ActiveSend>, hw::kTileRows> {};
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            auto& stage = pipeline_[tile];
            if (!stage.has_value()) {
                continue;
            }
            if (!advance[tile]) {
                next_pipeline[tile] = std::move(stage);
                continue;
            }
            const auto segment = input.consume_segment(
                tile, stage->instruction.stream_index);
            if (tile == 0) {
                stage->vector.vector_tag = segment.front().vector_tag;
            }
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                stage->vector.payload[tile][lane] = segment[lane].data;
            }
            if (tile + 1 == hw::kTileRows) {
                completed_.push_back(std::move(stage->vector));
            } else {
                next_pipeline[tile + 1] = std::move(stage);
            }
        }
        pipeline_ = std::move(next_pipeline);

        if (!completed_.empty() && external.can_send()) {
            external.send(std::move(completed_.front()));
            completed_.pop_front();
        }
    }

private:
    struct ActiveSend {
        C2cInstruction instruction{};
        C2cVector vector{};
    };

    C2cStreamPortMap::InputEndpoint endpoint_{};
    std::string name_{};
    std::deque<C2cInstruction> queue_{};
    std::array<std::optional<ActiveSend>, hw::kTileRows> pipeline_{};
    std::deque<C2cVector> completed_{};
};

class C2cRxSlice {
public:
    explicit C2cRxSlice(
        C2cStreamPortMap::OutputEndpoint endpoint,
        std::string name = "C2C RX",
        bool dedicated = false)
        : endpoint_(endpoint)
        , name_(std::move(name))
        , dedicated_(dedicated)
    {
    }

    void reset()
    {
        queue_.clear();
        pipeline_ = {};
        for (auto& queue : dedicated_queues_) queue.clear();
        for (auto& active : dedicated_active_) active.reset();
    }

    void issue(C2cInstruction instruction)
    {
        if (instruction.opcode != C2cOpcode::Receive) {
            throw std::invalid_argument("C2C RX accepts only Receive instructions");
        }
        if (dedicated_) {
            dedicated_queues_[instruction.stream_index].push_back(
                std::move(instruction));
        } else {
            queue_.push_back(std::move(instruction));
        }
    }

    bool idle() const noexcept
    {
        const auto dedicated_idle = std::all_of(
            dedicated_queues_.begin(), dedicated_queues_.end(),
            [](const auto& queue) { return queue.empty(); })
            && std::none_of(
                dedicated_active_.begin(), dedicated_active_.end(),
                [](const auto& active) { return active.has_value(); });
        return queue_.empty() && dedicated_idle
            && std::none_of(
                pipeline_.begin(), pipeline_.end(),
                [](const auto& stage) { return stage.has_value(); });
    }

    template <typename Transport, typename Consumer>
    void evaluate_dedicated(
        Transport& external,
        std::size_t stream_count,
        Consumer&& consume)
    {
        if (!dedicated_)
            throw std::logic_error(
                "legacy C2C RX cannot use the dedicated stream fabric");
        if (stream_count == 0
            || stream_count > hw::kC2cStreamsPerDirection)
            throw std::out_of_range("invalid dedicated C2C stream count");

        for (std::size_t stream = 0; stream < stream_count; ++stream) {
            auto& active = dedicated_active_[stream];
            if (!active.has_value() && !dedicated_queues_[stream].empty()) {
                active = ActiveDedicatedReceive {
                    std::move(dedicated_queues_[stream].front()), 0};
                dedicated_queues_[stream].pop_front();
            }
            if (!active.has_value() || !external.receive_ready(stream))
                continue;

            auto vector = external.pop_received(stream);
            auto consumer = active->instruction.consumer;
            consumer.base_row +=
                active->vector_index * consumer.row_stride;
            consumer.vector_count = 1;
            consume(C2cReceiveNotification {
                consumer,
                stream,
                vector.vector_tag,
                std::move(vector)});
            ++active->vector_index;
            if (active->vector_index
                == active->instruction.consumer.vector_count)
                active.reset();
        }
    }
    std::size_t queued_instruction_count() const noexcept { return queue_.size(); }
    std::size_t replayed_tile_count() const noexcept
    {
        return static_cast<std::size_t>(std::count_if(
            pipeline_.begin(), pipeline_.end(),
            [](const auto& stage) { return stage.has_value(); }));
    }

    template <typename Transport>
    std::optional<C2cReceiveNotification> evaluate(
        StreamRegisterFabric& fabric,
        Transport& external)
    {
        for (std::size_t tile = hw::kTileRows - 1; tile > 0; --tile) {
            pipeline_[tile] = std::move(pipeline_[tile - 1]);
        }
        pipeline_[0].reset();

        std::optional<C2cReceiveNotification> notification;
        if (!queue_.empty() && external.receive_ready()) {
            auto vector = external.pop_received();
            notification = C2cReceiveNotification {
                queue_.front().consumer,
                queue_.front().stream_index,
                vector.vector_tag,
            };
            pipeline_[0] = ActiveReceive {
                std::move(queue_.front()), std::move(vector)};
            queue_.pop_front();
        }

        auto output = StreamOutputPort(
            fabric, endpoint_.column, endpoint_.direction, name_);
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            const auto& stage = pipeline_[tile];
            if (!stage.has_value()) {
                continue;
            }
            output.write_payload_segment(
                tile,
                stage->instruction.stream_index,
                stage->vector.payload[tile],
                stage->vector.vector_tag);
        }
        pipeline_.back().reset();
        return notification;
    }

private:
    struct ActiveReceive {
        C2cInstruction instruction{};
        C2cVector vector{};
    };

    struct ActiveDedicatedReceive {
        C2cInstruction instruction{};
        std::size_t vector_index{0};
    };

    C2cStreamPortMap::OutputEndpoint endpoint_{};
    std::string name_{};
    std::deque<C2cInstruction> queue_{};
    std::array<std::optional<ActiveReceive>, hw::kTileRows> pipeline_{};
    bool dedicated_{false};
    std::array<std::deque<C2cInstruction>, hw::kC2cStreamsPerDirection>
        dedicated_queues_{};
    std::array<std::optional<ActiveDedicatedReceive>,
        hw::kC2cStreamsPerDirection> dedicated_active_{};
};

class C2cEndpoint {
public:
    explicit C2cEndpoint(
        C2cStreamPortMap ports,
        std::string name = "C2C",
        bool dedicated_rx = false)
        : tx_(ports.tx_input, name + " TX")
        , rx_(ports.rx_output, name + " RX", dedicated_rx)
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

    template <typename TxTransport, typename RxTransport>
    std::optional<C2cReceiveNotification> evaluate(
        StreamRegisterFabric& fabric,
        TxTransport& tx_external,
        RxTransport& rx_external)
    {
        tx_.evaluate(fabric, tx_external);
        return rx_.evaluate(fabric, rx_external);
    }

private:
    C2cTxSlice tx_;
    C2cRxSlice rx_;
};

} // namespace ftlpu
