#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ftlpu {

class HostBufferId {
public:
    constexpr HostBufferId() noexcept = default;

    explicit constexpr HostBufferId(std::uint64_t value) noexcept
        : value_(value)
    {
    }

    constexpr std::uint64_t value() const noexcept { return value_; }
    constexpr bool valid() const noexcept { return value_ != 0; }

    friend constexpr bool operator==(HostBufferId, HostBufferId) = default;

private:
    std::uint64_t value_{0};
};

struct HostBufferIdHash {
    std::size_t operator()(HostBufferId id) const noexcept
    {
        return static_cast<std::size_t>(id.value());
    }
};

class HostMemorySpace {
public:
    HostBufferId register_buffer(std::vector<std::uint8_t> bytes)
    {
        const auto id = HostBufferId(next_buffer_id_++);
        buffers_.emplace(id, std::move(bytes));
        return id;
    }

    HostBufferId allocate_buffer(
        std::size_t size_bytes,
        std::uint8_t initial_value = 0)
    {
        return register_buffer(
            std::vector<std::uint8_t>(size_bytes, initial_value));
    }

    bool contains(HostBufferId id) const noexcept
    {
        return buffers_.find(id) != buffers_.end();
    }

    std::size_t buffer_size(HostBufferId id) const
    {
        return buffer(id).size();
    }

    const std::vector<std::uint8_t>& buffer(HostBufferId id) const
    {
        const auto it = buffers_.find(id);
        if (it == buffers_.end()) {
            throw std::out_of_range("Host buffer handle is not registered");
        }
        return it->second;
    }

    std::vector<std::uint8_t>& buffer(HostBufferId id)
    {
        const auto it = buffers_.find(id);
        if (it == buffers_.end()) {
            throw std::out_of_range("Host buffer handle is not registered");
        }
        return it->second;
    }

    void unregister_buffer(HostBufferId id)
    {
        if (buffers_.erase(id) == 0) {
            throw std::out_of_range("Host buffer handle is not registered");
        }
    }

private:
    std::unordered_map<
        HostBufferId,
        std::vector<std::uint8_t>,
        HostBufferIdHash>
        buffers_{};
    std::uint64_t next_buffer_id_{1};
};

} // namespace ftlpu
