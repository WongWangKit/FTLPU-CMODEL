#pragma once

#include "ftlpu/c2c/instruction.hpp"
#include "ftlpu/c2c/link.hpp"
#include "ftlpu/core/stream_port.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
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

    static C2cStreamPortMap EastHemisphere(
        std::size_t tx_column,
        std::size_t rx_column)
    {
        return C2cStreamPortMap{
            InputEndpoint{tx_column, StreamDirection::East},
            OutputEndpoint{rx_column, StreamDirection::West},
        };
    }

    static C2cStreamPortMap WestHemisphere(
        std::size_t tx_column,
        std::size_t rx_column)
    {
        return C2cStreamPortMap{
            InputEndpoint{tx_column, StreamDirection::West},
            OutputEndpoint{rx_column, StreamDirection::East},
        };
    }
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

    void enqueue(C2cInstruction instruction)
    {
        if (instruction.opcode != C2cOpcode::Send) {
            throw std::invalid_argument("C2C TX accepts only Send instructions");
        }
        queue_.push_back(instruction);
    }

    void enqueue_send(std::size_t stream_index)
    {
        enqueue(C2cInstruction::Send(stream_index));
    }

    bool idle() const noexcept
    {
        return queue_.empty();
    }

    std::size_t queued_instruction_count() const noexcept
    {
        return queue_.size();
    }

    void evaluate(StreamRegisterFabric& fabric, C2cLink& link)
    {
        if (queue_.empty()) {
            return;
        }
        if (!link.can_send()) {
            throw std::logic_error(
                "C2C TX instruction reached issue while the link is busy");
        }

        const auto instruction = queue_.front();
        StreamInputPort input(
            fabric,
            endpoint_.column,
            endpoint_.direction,
            name_);

        // Validate the complete 320-byte transaction before consuming any tile.
        // This keeps a failed static schedule from partially consuming a vector.
        C2cVector vector{};
        bool tag_initialized = false;

        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            if (!input.segment_valid(tile, instruction.stream_index)) {
                throw std::logic_error(
                    "C2C Send requires all twenty 16-byte segments in the same cycle");
            }

            const auto segment = input.peek_segment(
                tile, instruction.stream_index);
            for (const auto& cell : segment) {
                if (!tag_initialized) {
                    vector.vector_tag = cell.vector_tag;
                    tag_initialized = true;
                } else if (cell.vector_tag != vector.vector_tag) {
                    throw std::logic_error(
                        "C2C Send observed mismatched vector_tag values across the 320-byte vector");
                }
            }
        }

        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            const auto segment =
                input.consume_segment(tile, instruction.stream_index);

            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                vector.payload[tile][lane] = segment[lane].data;
            }
        }

        link.send(std::move(vector));
        queue_.pop_front();
    }

private:
    C2cStreamPortMap::InputEndpoint endpoint_{};
    std::string name_{};
    std::deque<C2cInstruction> queue_{};
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

    void enqueue(C2cInstruction instruction)
    {
        if (instruction.opcode != C2cOpcode::Receive) {
            throw std::invalid_argument(
                "C2C RX accepts only Receive instructions");
        }
        queue_.push_back(instruction);
    }

    void enqueue_receive(std::size_t stream_index)
    {
        enqueue(C2cInstruction::Receive(stream_index));
    }

    bool idle() const noexcept
    {
        return queue_.empty();
    }

    std::size_t queued_instruction_count() const noexcept
    {
        return queue_.size();
    }

    void evaluate(StreamRegisterFabric& fabric, C2cLink& link)
    {
        if (queue_.empty()) {
            return;
        }
        if (!link.receive_ready()) {
            throw std::logic_error(
                "C2C Receive instruction issued before the vector arrived");
        }

        const auto instruction = queue_.front();
        const auto& vector = link.front_received();

        StreamOutputPort output(
            fabric,
            endpoint_.column,
            endpoint_.direction,
            name_);

        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            output.write_payload_segment(
                tile,
                instruction.stream_index,
                vector.payload[tile],
                vector.vector_tag);
        }

        link.pop_received();
        queue_.pop_front();
    }

private:
    C2cStreamPortMap::OutputEndpoint endpoint_{};
    std::string name_{};
    std::deque<C2cInstruction> queue_{};
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

    C2cTxSlice& tx() noexcept
    {
        return tx_;
    }

    const C2cTxSlice& tx() const noexcept
    {
        return tx_;
    }

    C2cRxSlice& rx() noexcept
    {
        return rx_;
    }

    const C2cRxSlice& rx() const noexcept
    {
        return rx_;
    }

private:
    C2cTxSlice tx_;
    C2cRxSlice rx_;
};

} // namespace ftlpu
