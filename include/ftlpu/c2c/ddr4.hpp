#pragma once

#include "ftlpu/c2c/types.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace ftlpu {

struct Ddr4Config {
    std::size_t beat_bytes{16};
    std::size_t read_latency_cycles{12};
    std::size_t write_latency_cycles{8};
    std::size_t request_queue_depth{8};

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

    void reset_execution_state()
    {
        requests_.clear();
        active_.reset();
        read_completions_.clear();
        write_completions_.clear();
        cycle_ = 0;
        next_request_id_ = 1;
        last_beat_.reset();
    }

    const Ddr4Config& config() const noexcept { return config_; }
    std::size_t cycle() const noexcept { return cycle_; }
    bool idle() const noexcept
    {
        return requests_.empty() && !active_.has_value();
    }
    const std::optional<BeatTrace>& last_beat() const noexcept
    {
        return last_beat_;
    }
    bool can_accept_request() const noexcept
    {
        return requests_.size() + (active_.has_value() ? 1U : 0U)
            < config_.request_queue_depth;
    }

    RequestId request_read(std::uint64_t address)
    {
        require_request_credit();
        validate_vector_range(address);
        const auto id = allocate_request_id();
        requests_.push_back(Request {
            id, Operation::Read, address, {}, config_.read_latency_cycles});
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
            config_.write_latency_cycles,
        });
        return id;
    }

    bool read_completion_ready(RequestId id) const noexcept
    {
        return !read_completions_.empty()
            && read_completions_.front().id == id;
    }

    ReadCompletion pop_read_completion(RequestId id)
    {
        if (!read_completion_ready(id)) {
            throw std::logic_error(
                "DDR4 read completion is not ready or is out of order");
        }
        auto completion = std::move(read_completions_.front());
        read_completions_.pop_front();
        return completion;
    }

    bool write_completion_ready(RequestId id) const noexcept
    {
        return !write_completions_.empty()
            && write_completions_.front().id == id;
    }

    WriteCompletion pop_write_completion(RequestId id)
    {
        if (!write_completion_ready(id)) {
            throw std::logic_error(
                "DDR4 write completion is not ready or is out of order");
        }
        const auto completion = write_completions_.front();
        write_completions_.pop_front();
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
                const auto it = bytes_.find(address + offset);
                vector.payload[tile][lane] =
                    it == bytes_.end() ? 0 : it->second;
            }
        }
        return vector;
    }

    void tick()
    {
        last_beat_.reset();
        if (!active_.has_value() && !requests_.empty()) {
            active_ = ActiveRequest {std::move(requests_.front()), 0};
            requests_.pop_front();
        }

        if (active_.has_value()) {
            auto& active = *active_;
            if (active.request.remaining_latency != 0) {
                --active.request.remaining_latency;
            } else {
                const auto remaining =
                    hw::kPhysicalVectorBytes - active.bytes_transferred;
                transfer_active_beat(
                    std::min(config_.beat_bytes, remaining));
                if (active.bytes_transferred == hw::kPhysicalVectorBytes) {
                    complete_active_request();
                }
            }
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

    void complete_active_request()
    {
        auto request = std::move(active_->request);
        active_.reset();
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
                bytes_[address + offset] = vector.payload[tile][lane];
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

    static void validate_vector_range(std::uint64_t address)
    {
        if (address > std::numeric_limits<std::uint64_t>::max()
                - (hw::kPhysicalVectorBytes - 1)) {
            throw std::out_of_range("DDR4 vector crosses the address space");
        }
    }

    Ddr4Config config_{};
    std::unordered_map<std::uint64_t, std::uint8_t> bytes_{};
    std::deque<Request> requests_{};
    void transfer_active_beat(std::size_t byte_count)
    {
        auto& active = *active_;
        const auto offset = active.bytes_transferred;
        for (std::size_t byte = 0; byte < byte_count; ++byte) {
            const auto vector_offset = offset + byte;
            const auto tile = vector_offset / hw::kLanesPerTile;
            const auto lane = vector_offset % hw::kLanesPerTile;
            const auto address = active.request.address + vector_offset;
            if (active.request.operation == Operation::Read) {
                const auto it = bytes_.find(address);
                active.request.vector.payload[tile][lane] =
                    it == bytes_.end() ? 0 : it->second;
            } else {
                bytes_[address] =
                    active.request.vector.payload[tile][lane];
            }
        }
        active.bytes_transferred += byte_count;
        last_beat_ = BeatTrace {
            active.request.id,
            active.request.operation == Operation::Write,
            active.request.address + offset,
            byte_count,
        };
    }

    std::optional<ActiveRequest> active_{};
    std::deque<ReadCompletion> read_completions_{};
    std::deque<WriteCompletion> write_completions_{};
    std::optional<BeatTrace> last_beat_{};
    RequestId next_request_id_{1};
    std::size_t cycle_{0};
};

} // namespace ftlpu
