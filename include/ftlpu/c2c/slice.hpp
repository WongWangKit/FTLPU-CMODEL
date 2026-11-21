#pragma once

#include "ftlpu/c2c/instruction.hpp"
#include "ftlpu/c2c/types.hpp"
#include "ftlpu/core/stream_port.hpp"
#include "ftlpu/core/timing.hpp"

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
    std::uint32_t sync_tag{0};
    std::uint64_t vector_tag{0};
    C2cVector vector{};
};

struct C2cTransmitReadRequest {
    C2cConsumer consumer{};
    std::size_t stream_index{0};
    std::uint32_t sync_tag{0};
};

class C2cTxSlice {
public:
    static constexpr std::size_t kDescriptorSlots=64;
    static constexpr std::size_t kGatherSlots=hw::kTileRows+1;
private:
    struct ActiveSend {
        C2cInstruction instruction{};
        C2cVector vector{};
    };

    struct State {
        std::deque<C2cInstruction> queue{};
        std::array<std::optional<ActiveSend>, hw::kTileRows> pipeline{};
        std::deque<C2cVector> completed{};
        std::array<std::deque<C2cInstruction>, hw::kC2cStreamsPerDirection>
            stream_queues{};
        std::array<std::array<std::optional<ActiveSend>, hw::kTileRows>,
            hw::kC2cStreamsPerDirection> stream_pipelines{};
        std::array<std::deque<C2cVector>, hw::kC2cStreamsPerDirection>
            stream_completed{};
        std::deque<C2cTransmitReadRequest> read_requests{};
    };

public:
    explicit C2cTxSlice(
        C2cStreamPortMap::InputEndpoint endpoint,
        std::string name = "C2C TX",
        bool indexed_transport = false)
        : endpoint_(endpoint)
        , name_(std::move(name))
        , indexed_transport_(indexed_transport)
    {
    }

    void reset()
    {
        current_ = {};
        next_.reset();
        pending_issues_.clear();
        low_prepared_ = false;
        low_complete_ = false;
    }

    void issue(C2cInstruction instruction)
    {
        if (instruction.opcode != C2cOpcode::Send) {
            throw std::invalid_argument("C2C TX accepts only Send instructions");
        }
        if (low_prepared_)
            throw std::logic_error("C2C TX issue must precede LOW evaluation");
        if(!instruction.vector_count || instruction.stream_index>=hw::kC2cStreamsPerDirection || !can_issue())
            throw std::invalid_argument("C2C descriptor range/capacity");
        pending_issues_.push_back(std::move(instruction));
    }

    bool idle() const noexcept
    {
        if (!pending_issues_.empty()) return false;
        const auto& state = current_;
        const bool indexed_idle = std::all_of(
            state.stream_queues.begin(), state.stream_queues.end(),
            [](const auto& queue) { return queue.empty(); })
            && std::all_of(state.stream_completed.begin(), state.stream_completed.end(),
                [](const auto& completed) { return completed.empty(); })
            && std::all_of(state.stream_pipelines.begin(), state.stream_pipelines.end(),
                [](const auto& pipeline) {
                    return std::none_of(
                        pipeline.begin(), pipeline.end(),
                        [](const auto& stage) { return stage.has_value(); });
                });
        return state.queue.empty() && state.completed.empty() && state.read_requests.empty()
            && indexed_idle
            && std::none_of(
                state.pipeline.begin(), state.pipeline.end(),
                [](const auto& stage) { return stage.has_value(); });
    }
    bool can_issue() const noexcept {return queued_instruction_count()<kDescriptorSlots;}
    std::size_t queued_instruction_count() const noexcept
    {
        std::size_t count = current_.queue.size();
        for (const auto& queue : current_.stream_queues) count += queue.size();
        count += pending_issues_.size();
        return count;
    }
    std::size_t gathered_tile_count() const noexcept
    {
        return static_cast<std::size_t>(std::count_if(
            current_.pipeline.begin(), current_.pipeline.end(),
            [](const auto& stage) { return stage.has_value(); }));
    }

    std::optional<C2cTransmitReadRequest> take_mem_read_request()
    {
        if (!next_)
            throw std::logic_error("C2C TX read request consumed outside LOW");
        if (next_->read_requests.empty()) return std::nullopt;
        auto request = std::move(next_->read_requests.front());
        next_->read_requests.pop_front();
        return request;
    }

    template <typename Transport>
    void evaluate(
        StreamRegisterFabric& fabric,
        Transport& external)
    {
        prepare_low();
        if (indexed_transport_) {
            evaluate_indexed(fabric, external);
            low_complete_ = true;
            return;
        }
        auto& state = *next_;
        if (!state.completed.empty() && external.can_send()) {
            external.send(std::move(state.completed.front()));
            state.completed.pop_front();
        }

        if (!state.queue.empty() && !state.pipeline[0].has_value()
            && state.read_requests.size()<kDescriptorSlots
            && occupied(state.pipeline)+state.completed.size()<kGatherSlots) {
            state.pipeline[0] = ActiveSend {take_vector(state.queue), {}};
            publish_mem_read_request(state, state.pipeline[0]->instruction);
        }

        auto input = StreamInputPort(
            fabric, endpoint_.column, endpoint_.direction, name_);
        auto advance = std::array<bool, hw::kTileRows> {};
        for (std::size_t tile = hw::kTileRows; tile-- > 0;) {
            auto& stage = state.pipeline[tile];
            if (!stage.has_value()) {
                continue;
            }
            const auto downstream_ready = tile + 1 == hw::kTileRows
                ? state.completed.size()<kGatherSlots
                : !state.pipeline[tile + 1].has_value()
                    || advance[tile + 1];
            advance[tile] = downstream_ready
                && input.segment_valid(
                    tile, stage->instruction.fabric_stream_index);
        }

        auto next_pipeline =
            std::array<std::optional<ActiveSend>, hw::kTileRows> {};
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            auto& stage = state.pipeline[tile];
            if (!stage.has_value()) {
                continue;
            }
            if (!advance[tile]) {
                next_pipeline[tile] = std::move(stage);
                continue;
            }
            const auto segment = input.consume_segment(
                tile, stage->instruction.fabric_stream_index);
            if (tile == 0) {
                stage->vector.vector_tag = segment.front().vector_tag;
            }
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                stage->vector.payload[tile][lane] = segment[lane].data;
            }
            if (tile + 1 == hw::kTileRows) {
                state.completed.push_back(std::move(stage->vector));
            } else {
                next_pipeline[tile + 1] = std::move(stage);
            }
        }
        state.pipeline = std::move(next_pipeline);

        if (!state.completed.empty() && external.can_send()) {
            external.send(std::move(state.completed.front()));
            state.completed.pop_front();
        }
        low_complete_ = true;
    }

    void validate_posedge() const
    {
        if (!low_prepared_ || !low_complete_ || !next_)
            throw std::logic_error("C2C TX posedge requires one complete LOW");
    }

    void posedge()
    {
        validate_posedge();
        current_ = std::move(*next_);
        next_.reset();
        low_prepared_ = false;
        low_complete_ = false;
    }

private:
    template <typename Transport>
    void evaluate_indexed(
        StreamRegisterFabric& fabric,
        Transport& external)
    {
        auto& state = *next_;
        auto input = StreamInputPort(
            fabric, endpoint_.column, endpoint_.direction, name_);
        for (std::size_t stream = 0;
             stream < hw::kC2cStreamsPerDirection; ++stream) {
            auto& completed = state.stream_completed[stream];
            if (!completed.empty()
                && transport_can_send(external, stream)) {
                transport_send(
                    external, std::move(completed.front()), stream);
                completed.pop_front();
            }

            auto& queue = state.stream_queues[stream];
            auto& pipeline = state.stream_pipelines[stream];
            if (!queue.empty() && !pipeline[0].has_value()
                && state.read_requests.size()<kDescriptorSlots
                && occupied(pipeline)+completed.size()<kGatherSlots) {
                pipeline[0] = ActiveSend {take_vector(queue), {}};
                publish_mem_read_request(state, pipeline[0]->instruction);
            }

            auto advance = std::array<bool, hw::kTileRows> {};
            for (std::size_t tile = hw::kTileRows; tile-- > 0;) {
                auto& stage = pipeline[tile];
                if (!stage.has_value()) continue;
                const bool downstream_ready = tile + 1 == hw::kTileRows
                    ? completed.size()<kGatherSlots
                    : !pipeline[tile + 1].has_value()
                        || advance[tile + 1];
                advance[tile] = downstream_ready
                    && input.segment_valid(
                        tile, stage->instruction.fabric_stream_index);
            }

            auto next_pipeline =
                std::array<std::optional<ActiveSend>, hw::kTileRows> {};
            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
                auto& stage = pipeline[tile];
                if (!stage.has_value()) continue;
                if (!advance[tile]) {
                    next_pipeline[tile] = std::move(stage);
                    continue;
                }
                const auto segment = input.consume_segment(
                    tile, stage->instruction.fabric_stream_index);
                if (tile == 0)
                    stage->vector.vector_tag = segment.front().vector_tag;
                for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
                    stage->vector.payload[tile][lane] = segment[lane].data;
                if (tile + 1 == hw::kTileRows)
                    completed.push_back(std::move(stage->vector));
                else
                    next_pipeline[tile + 1] = std::move(stage);
            }
            pipeline = std::move(next_pipeline);

            if (!completed.empty()
                && transport_can_send(external, stream)) {
                transport_send(
                    external, std::move(completed.front()), stream);
                completed.pop_front();
            }
        }
    }

    template <typename Transport>
    static bool transport_can_send(
        const Transport& external, std::size_t stream)
    {
        if constexpr (requires { external.can_send(stream); })
            return external.can_send(stream);
        else
            return stream == 0 && external.can_send();
    }

    template <typename Transport>
    static void transport_send(
        Transport& external, C2cVector vector, std::size_t stream)
    {
        if constexpr (requires {
                          external.send(std::move(vector), stream);
                      })
            external.send(std::move(vector), stream);
        else
            external.send(std::move(vector));
    }

    static C2cInstruction take_vector(std::deque<C2cInstruction>& queue) {
        auto instruction=queue.front();instruction.vector_count=1;
        if(--queue.front().vector_count==0)queue.pop_front();
        return instruction;
    }
    template<class P> static std::size_t occupied(const P& p) {
        return std::count_if(p.begin(),p.end(),[](const auto& e){return e.has_value();});
    }
    void prepare_low()
    {
        if (low_prepared_)
            throw std::logic_error("C2C TX LOW executed more than once");
        next_ = current_;
        while (!pending_issues_.empty()) {
            auto instruction = std::move(pending_issues_.front());
            pending_issues_.pop_front();
            if(indexed_transport_)next_->stream_queues[instruction.stream_index].push_back(instruction);
            else next_->queue.push_back(instruction);
        }
        low_prepared_ = true;
    }

    static void publish_mem_read_request(
        State& state, const C2cInstruction& instruction)
    {
        if (!instruction.consumer.notify_mem) return;
        state.read_requests.push_back(C2cTransmitReadRequest {
            instruction.consumer, instruction.stream_index,
            instruction.sync_tag});
    }

    C2cStreamPortMap::InputEndpoint endpoint_{};
    std::string name_{};
    bool indexed_transport_{false};
    State current_{};
    std::optional<State> next_{};
    std::deque<C2cInstruction> pending_issues_{};
    bool low_prepared_{};
    bool low_complete_{};
};

class C2cRxSlice {
public:
    static constexpr std::size_t kDescriptorSlots=64;
private:
    struct ActiveReceive {
        C2cInstruction instruction{};
        C2cVector vector{};
    };

    struct ActiveIndexedReceive {
        C2cInstruction instruction{};
        std::size_t vector_index{0};
    };

    struct State {
        std::deque<C2cInstruction> queue{};
        std::array<std::optional<ActiveReceive>, hw::kTileRows> pipeline{};
        std::array<std::deque<C2cInstruction>, hw::kC2cStreamsPerDirection>
            stream_queues{};
        std::array<std::optional<ActiveIndexedReceive>,
            hw::kC2cStreamsPerDirection> stream_active{};
        std::array<std::optional<ActiveReceive>, hw::kC2cStreamsPerDirection>
            shared_ingress{};
        std::array<std::array<std::optional<ActiveReceive>, hw::kTileRows>,
            hw::kC2cStreamsPerDirection> shared_pipelines{};
        std::array<std::size_t, hw::kC2cStreamsPerDirection>
            completed_instructions{};
    };

public:
    explicit C2cRxSlice(
        C2cStreamPortMap::OutputEndpoint endpoint,
        std::string name = "C2C RX",
        bool indexed_transport = false)
        : endpoint_(endpoint)
        , name_(std::move(name))
        , indexed_transport_(indexed_transport)
    {
    }

    void reset()
    {
        current_ = {};
        next_.reset();
        pending_issues_.clear();
        low_prepared_ = false;
        low_complete_ = false;
    }

    void issue(C2cInstruction instruction)
    {
        if (instruction.opcode != C2cOpcode::Receive)
            throw std::invalid_argument("C2C RX accepts only Receive instructions");
        if (low_prepared_)
            throw std::logic_error("C2C RX issue must precede LOW evaluation");
        if(!instruction.vector_count || instruction.stream_index>=hw::kC2cStreamsPerDirection || !can_issue())
            throw std::invalid_argument("C2C descriptor range/capacity");
        pending_issues_.push_back(std::move(instruction));
    }

    bool idle() const noexcept
    {
        if (!pending_issues_.empty()) return false;
        const auto& state = current_;
        const auto stream_queues_idle = std::all_of(
            state.stream_queues.begin(), state.stream_queues.end(),
            [](const auto& queue) { return queue.empty(); })
            && std::none_of(
                state.stream_active.begin(), state.stream_active.end(),
                [](const auto& active) { return active.has_value(); });
        const auto shared_pipeline_idle = std::none_of(
            state.shared_ingress.begin(), state.shared_ingress.end(),
            [](const auto& ingress) { return ingress.has_value(); })
            && std::all_of(state.shared_pipelines.begin(),
                state.shared_pipelines.end(), [](const auto& pipeline) {
                    return std::none_of(pipeline.begin(), pipeline.end(),
                        [](const auto& stage) { return stage.has_value(); });
                });
        return state.queue.empty() && stream_queues_idle && shared_pipeline_idle
            && std::none_of(state.pipeline.begin(), state.pipeline.end(),
                [](const auto& stage) { return stage.has_value(); });
    }

    template <typename Transport, typename Ready, typename Consumer>
    void evaluate_shared(
        StreamRegisterFabric& fabric,
        Transport& external,
        std::size_t stream_count,
        Ready&& ready,
        Consumer&& notify)
    {
        if (!indexed_transport_)
            throw std::logic_error(
                "C2C RX is not configured for the shared SR fabric");
        if (stream_count == 0
            || stream_count > hw::kC2cStreamsPerDirection)
            throw std::out_of_range("invalid shared C2C stream count");

        prepare_low();
        auto& state = *next_;
        auto output = StreamOutputPort(
            fabric, endpoint_.column, endpoint_.direction, name_, true);
        for (std::size_t stream = 0; stream < stream_count; ++stream) {
            auto& pipeline = state.shared_pipelines[stream];
            for (std::size_t tile = hw::kTileRows - 1; tile > 0; --tile)
                pipeline[tile] = std::move(pipeline[tile - 1]);
            pipeline[0] = std::move(state.shared_ingress[stream]);
            state.shared_ingress[stream].reset();

            auto& active = state.stream_active[stream];
            if (!active.has_value() && !state.stream_queues[stream].empty()) {
                active = ActiveIndexedReceive {
                    std::move(state.stream_queues[stream].front()), 0};
                state.stream_queues[stream].pop_front();
            }
            if (active.has_value() && external.receive_ready(stream)
                && ready(active->instruction)) {
                auto vector = external.pop_received(stream);
                auto consumer = active->instruction.consumer;
                consumer.base_row +=
                    active->vector_index * consumer.row_stride;
                consumer.vector_count = 1;
                notify(C2cReceiveNotification {
                    consumer,
                    active->instruction.fabric_stream_index,
                    active->instruction.sync_tag,
                    vector.vector_tag,
                    {}});
                state.shared_ingress[stream] = ActiveReceive {
                    active->instruction, std::move(vector)};
                ++active->vector_index;
                if (active->vector_index
                    == active->instruction.consumer.vector_count) {
                    ++state.completed_instructions[stream];
                    active.reset();
                }
            }

            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
                const auto& stage = pipeline[tile];
                if (!stage.has_value()) continue;
                output.write_payload_segment(tile,
                    stage->instruction.fabric_stream_index,
                    stage->vector.payload[tile],
                    stage->vector.vector_tag);
            }
        }
        low_complete_ = true;
    }

    bool can_issue() const noexcept {return queued_instruction_count()<kDescriptorSlots;}
    std::size_t queued_instruction_count() const noexcept
    {
        std::size_t count = current_.queue.size() + pending_issues_.size();
        for (const auto& queue : current_.stream_queues) count += queue.size();
        return count;
    }
    std::size_t completed_instruction_count(std::size_t stream) const
    {
        if (stream >= current_.completed_instructions.size())
            throw std::out_of_range("C2C RX completion stream is invalid");
        return current_.completed_instructions[stream];
    }
    std::size_t replayed_tile_count() const noexcept
    {
        return static_cast<std::size_t>(std::count_if(
            current_.pipeline.begin(), current_.pipeline.end(),
            [](const auto& stage) { return stage.has_value(); }))
            + static_cast<std::size_t>(std::count_if(
                current_.shared_ingress.begin(), current_.shared_ingress.end(),
                [](const auto& stage) { return stage.has_value(); }));
    }

    template <typename Transport>
    std::optional<C2cReceiveNotification> evaluate(
        StreamRegisterFabric& fabric,
        Transport& external)
    {
        prepare_low();
        auto& state = *next_;
        for (std::size_t tile = hw::kTileRows - 1; tile > 0; --tile)
            state.pipeline[tile] = std::move(state.pipeline[tile - 1]);
        state.pipeline[0].reset();

        std::optional<C2cReceiveNotification> notification;
        if (!state.queue.empty() && external.receive_ready()) {
            auto vector = external.pop_received();
            notification = C2cReceiveNotification {
                state.queue.front().consumer,
                state.queue.front().stream_index,
                state.queue.front().sync_tag,
                vector.vector_tag,
            };
            state.pipeline[0] = ActiveReceive {
                std::move(state.queue.front()), std::move(vector)};
            state.queue.pop_front();
        }

        auto output = StreamOutputPort(
            fabric, endpoint_.column, endpoint_.direction, name_, true);
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            const auto& stage = state.pipeline[tile];
            if (!stage.has_value()) continue;
            output.write_payload_segment(tile,
                stage->instruction.stream_index,
                stage->vector.payload[tile],
                stage->vector.vector_tag);
        }
        state.pipeline.back().reset();
        low_complete_ = true;
        return notification;
    }

    void validate_posedge() const
    {
        if (!low_prepared_ || !low_complete_ || !next_)
            throw std::logic_error("C2C RX posedge requires one complete LOW");
    }

    void posedge()
    {
        validate_posedge();
        current_ = std::move(*next_);
        next_.reset();
        low_prepared_ = false;
        low_complete_ = false;
    }

private:
    void prepare_low()
    {
        if (low_prepared_)
            throw std::logic_error("C2C RX LOW executed more than once");
        next_ = current_;
        while (!pending_issues_.empty()) {
            auto instruction = std::move(pending_issues_.front());
            pending_issues_.pop_front();
            if (indexed_transport_)
                next_->stream_queues[instruction.stream_index].push_back(
                    std::move(instruction));
            else
                next_->queue.push_back(std::move(instruction));
        }
        low_prepared_ = true;
    }

    C2cStreamPortMap::OutputEndpoint endpoint_{};
    std::string name_{};
    bool indexed_transport_{false};
    State current_{};
    std::optional<State> next_{};
    std::deque<C2cInstruction> pending_issues_{};
    bool low_prepared_{};
    bool low_complete_{};
};

class C2cEndpoint : public timing::ClockedModule {
public:
    explicit C2cEndpoint(
        C2cStreamPortMap ports,
        std::string name = "C2C",
        bool indexed_rx = false)
        : tx_(ports.tx_input, name + " TX", indexed_rx)
        , rx_(ports.rx_output, name + " RX", indexed_rx)
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

    // SystemPhase owns LOW evaluation because TX/RX depend on the stream
    // fabric and external transports.  Clock owns only the shared barrier.
    void low() override {}

    void validate_posedge() const override
    {
        tx_.validate_posedge();
        rx_.validate_posedge();
    }

    void posedge() override
    {
        validate_posedge();
        tx_.posedge();
        rx_.posedge();
    }

    void high() override {}

private:
    C2cTxSlice tx_;
    C2cRxSlice rx_;
};

} // namespace ftlpu

