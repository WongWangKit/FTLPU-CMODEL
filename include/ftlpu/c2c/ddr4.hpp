#pragma once

#include "ftlpu/c2c/types.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ftlpu {

struct Ddr4Config {
    std::size_t beat_bytes{hw::kPhysicalVectorBytes};
    // Default platform: a 500 MHz LPU attached to dual-channel DDR4-3200.
    // 51.2 GB/s corresponds to an average 102.4 bytes per LPU cycle.
    std::size_t read_latency_cycles{35};
    std::size_t write_latency_cycles{25};
    std::size_t request_queue_depth{256};
    // Maximum requests serviced in one cycle. Aggregate byte bandwidth is
    // independently limited by peak_bandwidth_bytes_per_second.
    std::size_t transfer_channels{
        hw::kHemispheres * hw::kC2cStreamsPerDirection};
    std::uint64_t lpu_clock_hz{500'000'000};
    std::uint64_t peak_bandwidth_bytes_per_second{51'200'000'000};
    std::size_t read_latency_jitter_cycles{15};
    std::size_t write_latency_jitter_cycles{10};
    std::uint64_t latency_random_seed{0x46544c5055444452ULL};

    void validate() const
    {
        if (beat_bytes == 0 || beat_bytes > hw::kPhysicalVectorBytes) {
            throw std::invalid_argument(
                "DDR4 beat_bytes must fit within one C2C vector");
        }
        if (request_queue_depth == 0) {
            throw std::invalid_argument(
                "DDR4 request queue depth must be non-zero");
        }
        if (transfer_channels == 0
            || transfer_channels
                > hw::kHemispheres * hw::kC2cStreamsPerDirection) {
            throw std::invalid_argument(
                "DDR4 transfer channel count exceeds the aggregate C2C fabric");
        }
        if (lpu_clock_hz == 0
            || peak_bandwidth_bytes_per_second == 0) {
            throw std::invalid_argument(
                "DDR4 clock and peak bandwidth must be non-zero");
        }
    }
};

class Ddr4Model {
public:
    using RequestId = std::uint64_t;

    struct ReadCompletion {
        RequestId id{0};
        std::uint64_t address{0};
        C2cVector vector{};
    };

    struct WriteCompletion {
        RequestId id{0};
        std::uint64_t address{0};
    };

    struct BeatTrace {
        RequestId id{0};
        bool write{false};
        std::uint64_t address{0};
        std::size_t byte_count{0};
    };

    explicit Ddr4Model(Ddr4Config config = {})
        : config_(config)
    {
        config_.validate();
    }

    void configure(Ddr4Config config)
    {
        if (!idle())
            throw std::logic_error(
                "cannot reconfigure DDR4 while requests are active");
        config.validate();
        config_ = config;
        bandwidth_remainder_ = 0;
        round_robin_cursor_ = 0;
        read_bytes_transferred_ = 0;
        write_bytes_transferred_ = 0;
    }

    void reset_execution_state()
    {
        requests_.clear();
        active_.clear();
        read_completions_.clear();
        write_completions_.clear();
        cycle_ = 0;
        next_request_id_ = 1;
        last_beat_.reset();
        bandwidth_remainder_ = 0;
        round_robin_cursor_ = 0;
        read_bytes_transferred_ = 0;
        write_bytes_transferred_ = 0;
    }

    const Ddr4Config& config() const noexcept { return config_; }
    std::size_t read_vector_service_cycles() const noexcept
    {
        const auto transferCycles = static_cast<std::size_t>(
            (static_cast<std::uint64_t>(hw::kPhysicalVectorBytes)
                    * config_.lpu_clock_hz
                + config_.peak_bandwidth_bytes_per_second - 1)
            / config_.peak_bandwidth_bytes_per_second);
        return config_.read_latency_cycles
            + config_.read_latency_jitter_cycles + transferCycles + 1;
    }
    std::size_t cycle() const noexcept { return cycle_; }
    bool idle() const noexcept
    {
        return requests_.empty() && active_.empty();
    }
    const std::optional<BeatTrace>& last_beat() const noexcept
    {
        return last_beat_;
    }
    std::uint64_t read_bytes_transferred() const noexcept
    {
        return read_bytes_transferred_;
    }
    std::uint64_t write_bytes_transferred() const noexcept
    {
        return write_bytes_transferred_;
    }
    std::uint64_t bytes_transferred() const noexcept
    {
        return read_bytes_transferred_ + write_bytes_transferred_;
    }
    bool can_accept_request() const noexcept
    {
        return requests_.size() + active_.size()
            < config_.request_queue_depth;
    }

    RequestId request_read(std::uint64_t address)
    {
        require_request_credit();
        validate_vector_range(address);
        const auto id = allocate_request_id();
        requests_.push_back(Request {
            id, Operation::Read, address, {},
            config_.read_latency_cycles
                + latency_jitter(id, address, Operation::Read)});
        return id;
    }

    RequestId request_write(std::uint64_t address, C2cVector vector)
    {
        require_request_credit();
        validate_vector_range(address);
        const auto id = allocate_request_id();
        requests_.push_back(Request {
            id,
            Operation::Write,
            address,
            std::move(vector),
            config_.write_latency_cycles
                + latency_jitter(id, address, Operation::Write),
        });
        return id;
    }

    bool read_completion_ready(RequestId id) const noexcept
    {
        return std::any_of(read_completions_.begin(), read_completions_.end(),
            [id](const ReadCompletion& completion) {
                return completion.id == id;
            });
    }

    ReadCompletion pop_read_completion(RequestId id)
    {
        if (!read_completion_ready(id)) {
            throw std::logic_error(
                "DDR4 read completion is not ready or is out of order");
        }
        const auto it = std::find_if(
            read_completions_.begin(), read_completions_.end(),
            [id](const ReadCompletion& completion) {
                return completion.id == id;
            });
        auto completion = std::move(*it);
        read_completions_.erase(it);
        return completion;
    }

    bool write_completion_ready(RequestId id) const noexcept
    {
        return std::any_of(write_completions_.begin(), write_completions_.end(),
            [id](const WriteCompletion& completion) {
                return completion.id == id;
            });
    }

    WriteCompletion pop_write_completion(RequestId id)
    {
        if (!write_completion_ready(id)) {
            throw std::logic_error(
                "DDR4 write completion is not ready or is out of order");
        }
        const auto it = std::find_if(
            write_completions_.begin(), write_completions_.end(),
            [id](const WriteCompletion& completion) {
                return completion.id == id;
            });
        const auto completion = *it;
        write_completions_.erase(it);
        return completion;
    }

    void initialize_vector(std::uint64_t address, const C2cVector& vector)
    {
        validate_vector_range(address);
        write_payload(address, vector);
    }

    C2cVector read_vector(std::uint64_t address) const
    {
        validate_vector_range(address);
        auto vector = C2cVector {};
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                const auto offset = tile * hw::kLanesPerTile + lane;
                const std::uint64_t byte_address = address + offset;
                const auto page = pages_.find(
                    byte_address / kStoragePageBytes);
                vector.payload[tile][lane] = page == pages_.end()
                    ? 0
                    : page->second[static_cast<std::size_t>(
                          byte_address % kStoragePageBytes)];
            }
        }
        return vector;
    }

    void tick()
    {
        last_beat_.reset();
        while (!requests_.empty()) {
            active_.push_back(
                ActiveRequest {std::move(requests_.front()), 0});
            requests_.pop_front();
        }

        bandwidth_remainder_ += config_.peak_bandwidth_bytes_per_second;
        std::uint64_t byteBudget =
            bandwidth_remainder_ / config_.lpu_clock_hz;
        bandwidth_remainder_ %= config_.lpu_clock_hz;

        std::size_t channelsUsed = 0;
        const std::size_t activeCount = active_.size();
        const std::size_t start = activeCount == 0
            ? 0 : round_robin_cursor_ % activeCount;
        for (std::size_t visited = 0;
             visited < activeCount && byteBudget != 0
                && channelsUsed < config_.transfer_channels;
             ++visited) {
            auto& active = active_[(start + visited) % activeCount];
            if (active.request.remaining_latency != 0) continue;
            const auto remaining =
                hw::kPhysicalVectorBytes - active.bytes_transferred;
            const auto byteCount = static_cast<std::size_t>(
                std::min<std::uint64_t>(
                    {config_.beat_bytes, remaining, byteBudget}));
            transfer_active_beat(active, byteCount);
            byteBudget -= byteCount;
            ++channelsUsed;
        }
        if (activeCount != 0)
            round_robin_cursor_ = (start + 1) % activeCount;
        for (auto& active : active_) {
            if (active.request.remaining_latency != 0)
                --active.request.remaining_latency;
        }
        for (std::size_t index = active_.size(); index-- > 0;) {
            if (active_[index].bytes_transferred
                == hw::kPhysicalVectorBytes)
                complete_active_request(index);
        }
        ++cycle_;
    }

private:
    enum class Operation { Read, Write };

    struct Request {
        RequestId id{0};
        Operation operation{Operation::Read};
        std::uint64_t address{0};
        C2cVector vector{};
        std::size_t remaining_latency{0};
    };

    struct ActiveRequest {
        Request request{};
        std::size_t bytes_transferred{0};
    };

    void complete_active_request(std::size_t index)
    {
        auto request = std::move(active_[index].request);
        active_.erase(active_.begin() + index);
        if (request.operation == Operation::Read) {
            read_completions_.push_back(ReadCompletion {
                request.id, request.address, std::move(request.vector)});
            return;
        }
        write_completions_.push_back(
            WriteCompletion {request.id, request.address});
    }

    void write_payload(std::uint64_t address, const C2cVector& vector)
    {
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                const auto offset = tile * hw::kLanesPerTile + lane;
                const std::uint64_t byte_address = address + offset;
                pages_[byte_address / kStoragePageBytes]
                    [static_cast<std::size_t>(
                        byte_address % kStoragePageBytes)] =
                    vector.payload[tile][lane];
            }
        }
    }

    void require_request_credit() const
    {
        if (!can_accept_request()) {
            throw std::logic_error("DDR4 request queue is full");
        }
    }

    RequestId allocate_request_id()
    {
        if (next_request_id_ == 0) {
            throw std::overflow_error("DDR4 request ID space is exhausted");
        }
        return next_request_id_++;
    }
    std::size_t latency_jitter(
        RequestId id, std::uint64_t address, Operation operation) const
    {
        const std::size_t maximum = operation == Operation::Read
            ? config_.read_latency_jitter_cycles
            : config_.write_latency_jitter_cycles;
        if (maximum == 0) return 0;
        std::uint64_t value = config_.latency_random_seed
            ^ (id * 0x9e3779b97f4a7c15ULL)
            ^ (address + (operation == Operation::Write
                    ? 0xd1b54a32d192ed03ULL : 0));
        value ^= value >> 30;
        value *= 0xbf58476d1ce4e5b9ULL;
        value ^= value >> 27;
        value *= 0x94d049bb133111ebULL;
        value ^= value >> 31;
        return static_cast<std::size_t>(
            value % (static_cast<std::uint64_t>(maximum) + 1));
    }

    static void validate_vector_range(std::uint64_t address)
    {
        if (address > std::numeric_limits<std::uint64_t>::max()
                - (hw::kPhysicalVectorBytes - 1)) {
            throw std::out_of_range("DDR4 vector crosses the address space");
        }
    }

    Ddr4Config config_{};
    static constexpr std::uint64_t kStoragePageBytes = 4096;
    using StoragePage = std::array<std::uint8_t, kStoragePageBytes>;
    std::unordered_map<std::uint64_t, StoragePage> pages_{};
    std::deque<Request> requests_{};
    void transfer_active_beat(
        ActiveRequest& active, std::size_t byte_count)
    {
        const auto offset = active.bytes_transferred;
        for (std::size_t byte = 0; byte < byte_count; ++byte) {
            const auto vector_offset = offset + byte;
            const auto tile = vector_offset / hw::kLanesPerTile;
            const auto lane = vector_offset % hw::kLanesPerTile;
            const auto address = active.request.address + vector_offset;
            if (active.request.operation == Operation::Read) {
                const auto page = pages_.find(
                    address / kStoragePageBytes);
                active.request.vector.payload[tile][lane] =
                    page == pages_.end() ? 0
                    : page->second[static_cast<std::size_t>(
                          address % kStoragePageBytes)];
            } else {
                pages_[address / kStoragePageBytes]
                    [static_cast<std::size_t>(
                        address % kStoragePageBytes)] =
                    active.request.vector.payload[tile][lane];
            }
        }
        active.bytes_transferred += byte_count;
        if (active.request.operation == Operation::Read)
            read_bytes_transferred_ += byte_count;
        else
            write_bytes_transferred_ += byte_count;
        last_beat_ = BeatTrace {
            active.request.id,
            active.request.operation == Operation::Write,
            active.request.address + offset,
            byte_count,
        };
    }

    std::vector<ActiveRequest> active_{};
    std::deque<ReadCompletion> read_completions_{};
    std::deque<WriteCompletion> write_completions_{};
    std::optional<BeatTrace> last_beat_{};
    RequestId next_request_id_{1};
    std::size_t cycle_{0};
    std::uint64_t bandwidth_remainder_{0};
    std::size_t round_robin_cursor_{0};
    std::uint64_t read_bytes_transferred_{0};
    std::uint64_t write_bytes_transferred_{0};
};

} // namespace ftlpu
