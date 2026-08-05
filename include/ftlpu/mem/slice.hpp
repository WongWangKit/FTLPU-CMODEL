#pragma once

#include "ftlpu/core/stream.hpp"
#include "ftlpu/mem/address.hpp"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ftlpu {

enum class MemOpcode {
    Read,
    Write,
    Gather,
    Scatter,
    ReadWrite,
};

struct MemInstruction {
    MemOpcode opcode{MemOpcode::Read};
    // Slice-local word address; its configured width is exposed by
    // MemLocalWordAddress13::kBits (the type name is legacy compatibility).
    MemLocalWordAddress13 address{};

    // Packed ISA selector retained for codec compatibility:
    //   0..31  = E0..E31
    //   32..63 = W0..W31
    // Architectural code should call stream_id()/map_stream_id().
    std::size_t stream{0};
    std::size_t map_stream{0};
    // Independent second-port fields, used only by ReadWrite.
    MemLocalWordAddress13 write_address{};
    std::size_t write_stream{0};

    StreamId stream_id() const
    {
        return StreamId::from_packed(stream);
    }

    StreamId map_stream_id() const
    {
        return StreamId::from_packed(map_stream);
    }

    static MemInstruction Read(MemLocalWordAddress13 address, StreamId stream)
    {
        return MemInstruction {MemOpcode::Read, address, stream.packed(), 0};
    }

    StreamId write_stream_id() const
    {
        return StreamId::from_packed(write_stream);
    }

    static MemInstruction Read(std::size_t address, StreamId stream)
    {
        return Read(MemLocalWordAddress13(address), stream);
    }

    static MemInstruction Read(MemLocalWordAddress13 address, std::size_t packed_stream)
    {
        return Read(address, StreamId::from_packed(packed_stream));
    }

    static MemInstruction Read(std::size_t address, std::size_t packed_stream)
    {
        return Read(address, StreamId::from_packed(packed_stream));
    }

    static MemInstruction Write(MemLocalWordAddress13 address, StreamId stream)
    {
        return MemInstruction {MemOpcode::Write, address, stream.packed(), 0};
    }

    static MemInstruction Write(std::size_t address, StreamId stream)
    {
        return Write(MemLocalWordAddress13(address), stream);
    }

    static MemInstruction Write(MemLocalWordAddress13 address, std::size_t packed_stream)
    {
        return Write(address, StreamId::from_packed(packed_stream));
    }

    static MemInstruction Write(std::size_t address, std::size_t packed_stream)
    {
        return Write(address, StreamId::from_packed(packed_stream));
    }

    static MemInstruction ReadWrite(
        MemLocalWordAddress13 read_address,
        StreamId read_stream,
        MemLocalWordAddress13 destination,
        StreamId destination_stream)
    {
        if (read_address == destination) {
            throw std::invalid_argument(
                "MEM ReadWrite requires distinct read and write addresses");
        }
        auto instruction = MemInstruction {
            MemOpcode::ReadWrite,
            read_address,
            read_stream.packed(),
            0};
        instruction.write_address = destination;
        instruction.write_stream = destination_stream.packed();
        return instruction;
    }

    static MemInstruction ReadWrite(
        std::size_t read_address,
        StreamId read_stream,
        std::size_t destination,
        StreamId destination_stream)
    {
        return ReadWrite(
            MemLocalWordAddress13(read_address),
            read_stream,
            MemLocalWordAddress13(destination),
            destination_stream);
    }

    static MemInstruction ReadWrite(
        std::size_t read_address,
        std::size_t read_packed_stream,
        std::size_t destination,
        std::size_t destination_packed_stream)
    {
        return ReadWrite(
            read_address,
            StreamId::from_packed(read_packed_stream),
            destination,
            StreamId::from_packed(destination_packed_stream));
    }

    static MemInstruction Gather(StreamId stream, StreamId map_stream)
    {
        return MemInstruction {
            MemOpcode::Gather,
            MemLocalWordAddress13(0),
            stream.packed(),
            map_stream.packed()};
    }

    static MemInstruction Gather(std::size_t packed_stream, std::size_t packed_map_stream)
    {
        return Gather(
            StreamId::from_packed(packed_stream),
            StreamId::from_packed(packed_map_stream));
    }

    static MemInstruction Scatter(StreamId stream, StreamId map_stream)
    {
        return MemInstruction {
            MemOpcode::Scatter,
            MemLocalWordAddress13(0),
            stream.packed(),
            map_stream.packed()};
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
// The full 20x44 MEM functional-slice model is MemArrayModel in mem_array.hpp.
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
        const auto address = instruction.address.encoded();
        if (address > memory_.size()
            || vector_length_ > memory_.size() - address) {
            throw std::out_of_range("MemSlice Read range is outside memory");
        }

        address_ = address;
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
