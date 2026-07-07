#pragma once

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
    Gather,
    Scatter,
};

struct MemInstruction {
    MemOpcode opcode{MemOpcode::Read};
    std::size_t address{0};
    std::size_t stream{0};
    std::size_t map_stream{0};

    static MemInstruction Read(std::size_t address, std::size_t stream)
    {
        return MemInstruction {MemOpcode::Read, address, stream, 0};
    }

    static MemInstruction Write(std::size_t address, std::size_t stream)
    {
        return MemInstruction {MemOpcode::Write, address, stream, 0};
    }

    static MemInstruction Gather(std::size_t stream, std::size_t map_stream)
    {
        return MemInstruction {MemOpcode::Gather, 0, stream, map_stream};
    }

    static MemInstruction Scatter(std::size_t stream, std::size_t map_stream)
    {
        return MemInstruction {MemOpcode::Scatter, 0, stream, map_stream};
    }
};

template <typename T>
struct MemStreamWord {
    std::size_t stream{0};
    StreamWord<T> word{};
};

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
