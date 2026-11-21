#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ftlpu::axi {

struct DramConfig {
    std::size_t read_latency_cycles{35};
    std::size_t latency_jitter_cycles{15};
    std::size_t bandwidth_bytes_per_cycle{32};
    std::size_t request_queue_depth{32};
    std::uint64_t random_seed{1};

    void validate() const
    {
        if (bandwidth_bytes_per_cycle == 0)
            throw std::invalid_argument(
                "memory bandwidth must be non-zero");
        if (request_queue_depth == 0)
            throw std::invalid_argument(
                "memory request queue depth must be non-zero");
    }
};

// Compatibility name retained for existing users of the generic backend.
using MemoryConfig = DramConfig;

// Interface between a protocol model and a byte-addressable read service.
// It intentionally contains no AXI signal names or AXI response semantics.
class ReadMemoryBackend {
public:
    using RequestId = std::uint64_t;

    struct ReadBeat {
        RequestId request_id{0};
        std::size_t beat_index{0};
        std::uint64_t address{0};
        std::vector<std::uint8_t> data{};
        bool last{false};
    };

    struct TickResult {
        std::vector<ReadBeat> completed_beats{};
        std::size_t bytes_serviced{0};
    };

    virtual ~ReadMemoryBackend() = default;

    virtual bool can_accept_read() const noexcept = 0;
    virtual RequestId request_read(
        std::uint64_t address,
        std::size_t beat_bytes,
        std::size_t beat_count) = 0;
    virtual TickResult tick(std::size_t max_completed_beats) = 0;
    virtual void reset_execution_state() = 0;
};

// Simple external-DRAM behavior visible at the controller boundary. This is
// intentionally not a bank/row/refresh/PHY timing model. Request latency
// overlaps, while returned beats remain in request acceptance order.
class SimpleDramBackend final : public ReadMemoryBackend {
public:
    struct ReadRequestTrace {
        RequestId id{0};
        std::uint64_t address{0};
        std::size_t beat_bytes{0};
        std::size_t beat_count{0};
        std::size_t submitted_cycle{0};
        std::size_t initial_latency_cycles{0};
        std::optional<std::size_t> first_service_cycle{};
        std::optional<std::size_t> completion_cycle{};
    };

    explicit SimpleDramBackend(DramConfig config = {})
        : config_(config)
    {
        config_.validate();
    }

    const DramConfig& config() const noexcept { return config_; }
    std::size_t cycle() const noexcept { return cycle_; }
    bool idle() const noexcept { return reads_.empty(); }
    std::size_t queued_read_count() const noexcept { return reads_.size(); }
    std::uint64_t bytes_serviced() const noexcept
    {
        return bytes_serviced_;
    }
    const std::vector<ReadRequestTrace>& traces() const noexcept
    {
        return traces_;
    }

    bool can_accept_read() const noexcept override
    {
        return reads_.size() < config_.request_queue_depth;
    }

    RequestId request_read(std::uint64_t address,
        std::size_t beat_bytes, std::size_t beat_count) override
    {
        if (!can_accept_read())
            throw std::logic_error("memory read request queue is full");
        validate_read_range(address, beat_bytes, beat_count);
        if (next_request_id_ == 0)
            throw std::overflow_error(
                "memory read request ID space is exhausted");

        const auto id = next_request_id_++;
        const auto latency = config_.read_latency_cycles
            + latency_jitter(id, address);
        const auto traceIndex = traces_.size();
        traces_.push_back(ReadRequestTrace {
            id,
            address,
            beat_bytes,
            beat_count,
            cycle_,
            latency,
        });
        reads_.push_back(ActiveRead {
            id,
            address,
            beat_bytes,
            beat_count,
            latency,
            0,
            0,
            std::vector<std::uint8_t>(beat_bytes, 0),
            traceIndex,
        });
        return id;
    }

    TickResult tick(std::size_t max_completed_beats) override
    {
        auto result = TickResult {};

        // All accepted requests age together, even though this first version
        // returns data globally in request order.
        for (auto& read : reads_)
            if (read.remaining_latency != 0)
                --read.remaining_latency;

        auto byteBudget = config_.bandwidth_bytes_per_cycle;
        while (byteBudget != 0 && max_completed_beats != 0
            && !reads_.empty()) {
            auto& read = reads_.front();
            if (read.remaining_latency != 0)
                break;

            auto& trace = traces_[read.trace_index];
            if (!trace.first_service_cycle.has_value())
                trace.first_service_cycle = cycle_;

            const auto bytesRemaining =
                read.beat_bytes - read.bytes_into_beat;
            const auto byteCount = std::min(byteBudget, bytesRemaining);
            const auto beatAddress = read.address
                + read.completed_beats * read.beat_bytes;
            for (std::size_t byte = 0; byte < byteCount; ++byte) {
                read.current_beat[read.bytes_into_beat + byte] =
                    read_byte(beatAddress + read.bytes_into_beat + byte);
            }
            read.bytes_into_beat += byteCount;
            byteBudget -= byteCount;
            result.bytes_serviced += byteCount;
            bytes_serviced_ += byteCount;

            if (read.bytes_into_beat != read.beat_bytes)
                continue;

            const auto beatIndex = read.completed_beats;
            const auto last = beatIndex + 1 == read.beat_count;
            result.completed_beats.push_back(ReadBeat {
                read.id,
                beatIndex,
                beatAddress,
                std::move(read.current_beat),
                last,
            });
            --max_completed_beats;
            ++read.completed_beats;
            read.bytes_into_beat = 0;

            if (last) {
                trace.completion_cycle = cycle_;
                reads_.pop_front();
            } else {
                read.current_beat.assign(read.beat_bytes, 0);
            }
        }

        ++cycle_;
        return result;
    }

    void reset_execution_state() override
    {
        reads_.clear();
        traces_.clear();
        cycle_ = 0;
        next_request_id_ = 1;
        bytes_serviced_ = 0;
    }

    void initialize_bytes(
        std::uint64_t address, std::span<const std::uint8_t> data)
    {
        validate_byte_range(address, data.size());
        for (std::size_t offset = 0; offset < data.size(); ++offset)
            write_byte(address + offset, data[offset]);
    }

    std::vector<std::uint8_t> read_bytes(
        std::uint64_t address, std::size_t byte_count) const
    {
        validate_byte_range(address, byte_count);
        auto result = std::vector<std::uint8_t>(byte_count, 0);
        for (std::size_t offset = 0; offset < byte_count; ++offset)
            result[offset] = read_byte(address + offset);
        return result;
    }

    std::uint8_t read_byte(std::uint64_t address) const noexcept
    {
        const auto page = pages_.find(address / kStoragePageBytes);
        return page == pages_.end()
            ? 0
            : page->second[static_cast<std::size_t>(
                  address % kStoragePageBytes)];
    }

    void write_byte(std::uint64_t address, std::uint8_t value)
    {
        pages_[address / kStoragePageBytes]
            [static_cast<std::size_t>(address % kStoragePageBytes)] =
            value;
    }

private:
    struct ActiveRead {
        RequestId id{0};
        std::uint64_t address{0};
        std::size_t beat_bytes{0};
        std::size_t beat_count{0};
        std::size_t remaining_latency{0};
        std::size_t completed_beats{0};
        std::size_t bytes_into_beat{0};
        std::vector<std::uint8_t> current_beat{};
        std::size_t trace_index{0};
    };

    static void validate_byte_range(
        std::uint64_t address, std::size_t byte_count)
    {
        if (byte_count == 0)
            return;
        if (byte_count - 1
            > std::numeric_limits<std::uint64_t>::max() - address)
            throw std::out_of_range(
                "memory byte range crosses the address space");
    }

    static void validate_read_range(std::uint64_t address,
        std::size_t beat_bytes, std::size_t beat_count)
    {
        if (beat_bytes == 0 || beat_count == 0)
            throw std::invalid_argument(
                "memory read beat size and count must be non-zero");
        if (beat_count
            > std::numeric_limits<std::size_t>::max() / beat_bytes)
            throw std::out_of_range("memory read byte count overflows");
        validate_byte_range(address, beat_bytes * beat_count);
    }

    std::size_t latency_jitter(
        RequestId id, std::uint64_t address) const noexcept
    {
        if (config_.latency_jitter_cycles == 0)
            return 0;
        auto value = config_.random_seed
            ^ (id * 0x9e3779b97f4a7c15ULL) ^ address;
        value ^= value >> 30;
        value *= 0xbf58476d1ce4e5b9ULL;
        value ^= value >> 27;
        value *= 0x94d049bb133111ebULL;
        value ^= value >> 31;
        return static_cast<std::size_t>(
            value
            % (static_cast<std::uint64_t>(
                   config_.latency_jitter_cycles)
                + 1));
    }

    static constexpr std::uint64_t kStoragePageBytes = 4096;
    using StoragePage = std::array<std::uint8_t, kStoragePageBytes>;

    DramConfig config_{};
    std::unordered_map<std::uint64_t, StoragePage> pages_{};
    std::deque<ActiveRead> reads_{};
    std::vector<ReadRequestTrace> traces_{};
    RequestId next_request_id_{1};
    std::size_t cycle_{0};
    std::uint64_t bytes_serviced_{0};
};

// Compatibility alias; SimpleDramBackend is the canonical DRAM model name.
using ByteMemoryBackend = SimpleDramBackend;

} // namespace ftlpu::axi
