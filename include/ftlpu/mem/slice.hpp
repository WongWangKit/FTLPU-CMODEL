#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/stream.hpp"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ftlpu {

enum class MemOpcode {
    Read,
    Write,
    ReadWrite,
    Gather,
    Scatter,
    Accumulate,
};

enum class MemAccumulatorDestination {
    Sram,
    Stream,
};

struct MemInstruction {
    MemOpcode opcode{MemOpcode::Read};
    // SRAM row address (0..8191), not a byte address.
    std::size_t address{0};

    // Packed ISA selector retained for codec compatibility:
    //   0..31  = E0..E31
    //   32..63 = W0..W31
    // Architectural code should call stream_id()/map_stream_id().
    std::size_t stream{0};
    std::size_t map_stream{0};
    MemAccumulatorDestination accumulator_destination{MemAccumulatorDestination::Sram};
    // Second SRAM port fields, used only by ReadWrite.
    std::size_t write_address{0};
    std::size_t write_stream{0};

    StreamId stream_id() const
    {
        return StreamId::from_packed(stream);
    }

    StreamId map_stream_id() const
    {
        return StreamId::from_packed(map_stream);
    }

    static MemInstruction Read(std::size_t address, StreamId stream)
    {
        return MemInstruction {MemOpcode::Read, address, stream.packed(), 0};
    }

    static MemInstruction Read(std::size_t address, std::size_t packed_stream)
    {
        return Read(address, StreamId::from_packed(packed_stream));
    }

    static MemInstruction Write(std::size_t address, StreamId stream)
    {
        return MemInstruction {MemOpcode::Write, address, stream.packed(), 0};
    }

    static MemInstruction Write(std::size_t address, std::size_t packed_stream)
    {
        return Write(address, StreamId::from_packed(packed_stream));
    }

    StreamId write_stream_id() const
    {
        return StreamId::from_packed(write_stream);
    }

    static MemInstruction ReadWrite(
        std::size_t read_address,
        StreamId read_stream,
        std::size_t write_address,
        StreamId write_stream)
    {
        if (read_address == write_address) {
            throw std::invalid_argument("MEM ReadWrite requires distinct read and write addresses");
        }
        auto instruction = MemInstruction {MemOpcode::ReadWrite, read_address, read_stream.packed(), 0};
        instruction.write_address = write_address;
        instruction.write_stream = write_stream.packed();
        return instruction;
    }

    static MemInstruction ReadWrite(
        std::size_t read_address,
        std::size_t read_packed_stream,
        std::size_t write_address,
        std::size_t write_packed_stream)
    {
        return ReadWrite(
            read_address,
            StreamId::from_packed(read_packed_stream),
            write_address,
            StreamId::from_packed(write_packed_stream));
    }

    static MemInstruction Accumulate(
        std::size_t address,
        StreamId stream,
        MemAccumulatorDestination destination = MemAccumulatorDestination::Sram)
    {
        if (stream.direction() != StreamDirection::West) {
            throw std::invalid_argument("MEM Accumulate requires a west stream base");
        }
        if (stream.index() + sizeof(float) > hw::kWestStreams) {
            throw std::out_of_range("MEM Accumulate FP32 input exceeds west streams");
        }
        return MemInstruction {MemOpcode::Accumulate, address, stream.packed(), 0, destination};
    }

    static MemInstruction Accumulate(
        std::size_t address,
        std::size_t packed_stream,
        MemAccumulatorDestination destination = MemAccumulatorDestination::Sram)
    {
        return Accumulate(address, StreamId::from_packed(packed_stream), destination);
    }

    static MemInstruction Gather(StreamId stream, StreamId map_stream)
    {
        return MemInstruction {MemOpcode::Gather, 0, stream.packed(), map_stream.packed()};
    }

    static MemInstruction Gather(std::size_t packed_stream, std::size_t packed_map_stream)
    {
        return Gather(
            StreamId::from_packed(packed_stream),
            StreamId::from_packed(packed_map_stream));
    }

    static MemInstruction Scatter(StreamId stream, StreamId map_stream)
    {
        return MemInstruction {MemOpcode::Scatter, 0, stream.packed(), map_stream.packed()};
    }

    static MemInstruction Scatter(std::size_t packed_stream, std::size_t packed_map_stream)
    {
        return Scatter(
            StreamId::from_packed(packed_stream),
            StreamId::from_packed(packed_map_stream));
    }
};

template <typename T>
struct MemStreamWord {
    std::size_t stream{0};
    StreamWord<T> word{};

    StreamId stream_id() const
    {
        return StreamId::from_packed(stream);
    }
};

// Small generic scalar MEM helper retained for unit-level experimentation.
// The full 4x44 MEM functional-slice model is MemArrayModel in mem_array.hpp.
template <typename T>
class MemSlice {
public:
    MemSlice(std::vector<T> memory, std::size_t vector_length)
        : memory_(std::move(memory))
        , vector_length_(vector_length)
    {
        if (vector_length_ == 0) {
            throw std::invalid_argument("MemSlice vector length must be non-zero");
        }
    }

    void reset()
    {
        busy_ = false;
        address_ = 0;
        remaining_ = 0;
        stream_ = 0;
        output_.reset();
    }

    bool busy() const
    {
        return busy_;
    }

    const std::optional<MemStreamWord<T>>& output() const
    {
        return output_;
    }

    void issue(const MemInstruction& instruction)
    {
        if (busy_) {
            throw std::logic_error("MemSlice instruction issued while busy");
        }
        if (instruction.opcode != MemOpcode::Read) {
            throw std::logic_error("MemSlice currently implements Read only");
        }
        if (instruction.address > memory_.size()
            || vector_length_ > memory_.size() - instruction.address) {
            throw std::out_of_range("MemSlice Read range is outside memory");
        }

        address_ = instruction.address;
        remaining_ = vector_length_;
        stream_ = instruction.stream;
        busy_ = true;
        output_.reset();
    }

    void tick()
    {
        output_.reset();

        if (!busy_) {
            return;
        }

        output_ = MemStreamWord<T> {
            stream_,
            StreamWord<T> {
                memory_[address_],
                remaining_ == 1,
            },
        };

        ++address_;
        --remaining_;
        busy_ = remaining_ != 0;
    }

private:
    std::vector<T> memory_{};
    std::size_t vector_length_{0};
    bool busy_{false};
    std::size_t address_{0};
    std::size_t remaining_{0};
    std::size_t stream_{0};
    std::optional<MemStreamWord<T>> output_{};
};

} // namespace ftlpu
