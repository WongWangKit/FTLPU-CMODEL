#pragma once

#include <memory>
#include <utility>

namespace ftlpu {

// A unique owner whose copy operation explicitly deep-clones the pointee.
// This is used only for finite hardware-state storage that participates in a
// Current/Next snapshot; it is not a shared ownership model.
template <typename T>
class ClonePtr {
public:
    ClonePtr() = default;
    ClonePtr(std::nullptr_t) noexcept {}
    ClonePtr(std::unique_ptr<T> pointer) noexcept
        : pointer_(std::move(pointer))
    {}

    ClonePtr(const ClonePtr& other)
        : pointer_(other.pointer_
              ? std::make_unique<T>(*other.pointer_)
              : nullptr)
    {}

    ClonePtr& operator=(const ClonePtr& other)
    {
        if (this == &other) return *this;
        auto copy = ClonePtr(other);
        pointer_.swap(copy.pointer_);
        return *this;
    }

    ClonePtr(ClonePtr&&) noexcept = default;
    ClonePtr& operator=(ClonePtr&&) noexcept = default;

    ClonePtr& operator=(std::unique_ptr<T> pointer) noexcept
    {
        pointer_ = std::move(pointer);
        return *this;
    }

    void reset() noexcept { pointer_.reset(); }
    explicit operator bool() const noexcept
    {
        return static_cast<bool>(pointer_);
    }
    T& operator*() noexcept { return *pointer_; }
    const T& operator*() const noexcept { return *pointer_; }
    T* operator->() noexcept { return pointer_.get(); }
    const T* operator->() const noexcept { return pointer_.get(); }

private:
    std::unique_ptr<T> pointer_{};
};

} // namespace ftlpu
