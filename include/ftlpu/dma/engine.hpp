#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/stream.hpp"
#include "ftlpu/dma/descriptor.hpp"
#include "ftlpu/dma/global_memory.hpp"

#include <cstddef>
#include <deque>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ftlpu {

class DmaEngine {
public:
    struct BeatTrace {
        DmaTransferId id{};
        DmaDirection direction{DmaDirection::HostToMemory};
        DmaPurpose purpose{DmaPurpose::General};
        HostBufferId host_buffer{};
        std::size_t host_offset_bytes{0};
        MemGlobalAddress24 memory_address{};
        std::size_t vector_index{0};
    };

    struct Completion {
        DmaTransferId id{};
        DmaDirection direction{DmaDirection::HostToMemory};
        DmaPurpose purpose{DmaPurpose::General};
        std::size_t vector_count{0};
    };

    DmaEngine(HostMemorySpace& host, GlobalMemoryAddressSpace& memory)
        : host_(host)
        , memory_(memory)
    {
    }

    void reset()
    {
        cycle_ = 0;
        queue_.clear();
        active_.reset();
        last_beat_.reset();
        completions_.clear();
    }

    std::size_t cycle() const noexcept { return cycle_; }

    bool idle() const noexcept
    {
        return queue_.empty() && !active_.has_value();
    }

    std::size_t queued_descriptor_count() const noexcept
    {
        return queue_.size() + (active_.has_value() ? 1 : 0);
    }

    const std::optional<BeatTrace>& last_beat() const noexcept
    {
        return last_beat_;
    }

    const std::vector<Completion>& completions() const noexcept
    {
        return completions_;
    }

    void enqueue(DmaDescriptor descriptor)
    {
        validate(descriptor);
        queue_.push_back(std::move(descriptor));
    }

    // Move at most one 320-byte vector and advance one DMA cycle. Returns true
    // when this cycle transferred a vector and false when the engine was idle.
    bool tick()
    {
        last_beat_.reset();
        if (!active_.has_value() && !queue_.empty()) {
            active_ = ActiveTransfer {std::move(queue_.front()), 0};
            queue_.pop_front();
        }

        if (!active_.has_value()) {
            ++cycle_;
            return false;
        }

        auto& active = *active_;
        const auto address = vector_address(
            active.descriptor.memory_address,
            active.vector_index);
        const auto host_offset = active.descriptor.host_offset_bytes
            + active.vector_index * hw::kPhysicalVectorBytes;

        if (active.descriptor.direction == DmaDirection::HostToMemory) {
            memory_.write_vector(
                address,
                load_host_vector(active.descriptor.host_buffer, host_offset));
        } else {
            store_host_vector(
                active.descriptor.host_buffer,
                host_offset,
                memory_.read_vector(address));
        }

        last_beat_ = BeatTrace {
            active.descriptor.id,
            active.descriptor.direction,
            active.descriptor.purpose,
            active.descriptor.host_buffer,
            host_offset,
            address,
            active.vector_index,
        };

        ++active.vector_index;
        if (active.vector_index == active.descriptor.vector_count) {
            completions_.push_back(Completion {
                active.descriptor.id,
                active.descriptor.direction,
                active.descriptor.purpose,
                active.descriptor.vector_count,
            });
            active_.reset();
        }

        ++cycle_;
        return true;
    }

private:
    struct ActiveTransfer {
        DmaDescriptor descriptor{};
        std::size_t vector_index{0};
    };

    void validate(const DmaDescriptor& descriptor) const
    {
        if (!descriptor.host_buffer.valid()
            || !host_.contains(descriptor.host_buffer)) {
            throw std::out_of_range("DMA descriptor references an unknown Host buffer");
        }
        if (descriptor.vector_count == 0) {
            throw std::invalid_argument("DMA descriptor vector_count must be non-zero");
        }

        const auto buffer_size = host_.buffer_size(descriptor.host_buffer);
        if (descriptor.host_offset_bytes > buffer_size) {
            throw std::out_of_range("DMA descriptor exceeds the Host buffer range");
        }
        const auto available_bytes = buffer_size - descriptor.host_offset_bytes;
        if (descriptor.vector_count
            > available_bytes / hw::kPhysicalVectorBytes) {
            throw std::out_of_range("DMA descriptor exceeds the Host buffer range");
        }

        const auto decoded = memory_.decode(descriptor.memory_address);
        (void)decoded.local_word.advance_words(descriptor.vector_count - 1);
    }

    static MemGlobalAddress24 vector_address(
        MemGlobalAddress24 base,
        std::size_t vector_index)
    {
        const auto local = base.slice_byte_address().local_word_address()
            .advance_words(vector_index);
        return MemGlobalAddress24::FromFields(
            base.hemisphere(),
            base.mem_slice(),
            local.slice_byte_address());
    }

    StreamPayloadVector320 load_host_vector(
        HostBufferId buffer_id,
        std::size_t offset) const
    {
        const auto& buffer = host_.buffer(buffer_id);
        StreamPayloadVector320 result{};
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                result[tile][lane] = buffer[
                    offset + tile * hw::kLanesPerTile + lane];
            }
        }
        return result;
    }

    void store_host_vector(
        HostBufferId buffer_id,
        std::size_t offset,
        const StreamPayloadVector320& values)
    {
        auto& buffer = host_.buffer(buffer_id);
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                buffer[offset + tile * hw::kLanesPerTile + lane]
                    = values[tile][lane];
            }
        }
    }

    HostMemorySpace& host_;
    GlobalMemoryAddressSpace& memory_;
    std::deque<DmaDescriptor> queue_{};
    std::optional<ActiveTransfer> active_{};
    std::optional<BeatTrace> last_beat_{};
    std::vector<Completion> completions_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
