#pragma once

#include <optional>
#include <utility>

namespace ftlpu {

template <typename T>
struct StreamWord {
    T data{};
    bool last{false};
};

template <typename T>
using StreamValue = std::optional<StreamWord<T>>;

template <typename T>
class StreamRegister {
public:
    void reset()
    {
        input_.reset();
        output_.reset();
    }

    void set_input(StreamValue<T> input)
    {
        input_ = std::move(input);
    }

    void clear_input()
    {
        input_.reset();
    }

    const StreamValue<T>& output() const
    {
        return output_;
    }

    void tick()
    {
        output_ = input_;
        input_.reset();
    }

private:
    StreamValue<T> input_{};
    StreamValue<T> output_{};
};

} // namespace ftlpu
