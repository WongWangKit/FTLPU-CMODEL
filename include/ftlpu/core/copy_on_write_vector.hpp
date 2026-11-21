#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace ftlpu {

// Program images are Memory, not a bank of DFFs. Runtime Current/Next
// snapshots therefore share the immutable backing vector. Configuration-time
// writes detach before modifying it.
template <typename T>
class CopyOnWriteVector {
public:
    using Container = std::vector<T>;

    CopyOnWriteVector()
        : values_(std::make_shared<Container>())
    {}

    std::size_t size() const noexcept { return values_->size(); }
    bool empty() const noexcept { return values_->empty(); }

    const T& operator[](std::size_t index) const noexcept
    {
        return (*values_)[index];
    }
    T& operator[](std::size_t index)
    {
        detach();
        return (*values_)[index];
    }

    void clear()
    {
        detach();
        values_->clear();
    }
    void resize(std::size_t size)
    {
        detach();
        values_->resize(size);
    }
    void swap(Container& other)
    {
        detach();
        values_->swap(other);
    }

    const Container& values() const noexcept { return *values_; }

private:
    void detach()
    {
        if (values_.use_count() != 1)
            values_ = std::make_shared<Container>(*values_);
    }

    std::shared_ptr<Container> values_;
};

} // namespace ftlpu
