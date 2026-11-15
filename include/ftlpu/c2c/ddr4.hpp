#pragma once

#include "ftlpu/c2c/types.hpp"
#include "ftlpu/core/timing.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ftlpu {

struct Ddr4Config {
    std::size_t beat_bytes{hw::kC2cBytesPerStreamPerCycle};
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

class Ddr4Model : public timing::ClockedModule {
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

    void enable_clocked_mode()
    {
        if (clock_managed_ || state_.cycle != 0 || next_.has_value()
            || !pending_requests_.empty())
            throw std::logic_error(
                "DDR4 clocked mode must be enabled before execution");
        clock_managed_ = true;
    }

    bool clocked_mode() const noexcept { return clock_managed_; }

    void low() override
    {
        if (!clock_managed_) {
            tick_immediate();
            return;
        }
        if (next_.has_value())
            throw std::logic_error("DDR4 LOW executed more than once");
        next_ = state_;
        apply_pending_inputs(*next_);
        next_memory_writes_.clear();
        tick_state(*next_);
    }

    void validate_posedge() const override
    {
        if (clock_managed_ && !next_.has_value())
            throw std::logic_error(
                "DDR4 posedge requires one complete LOW");
    }

    void posedge() override
    {
        if (!clock_managed_) return;
        validate_posedge();
        commit_memory_writes();
        state_ = std::move(*next_);
        next_.reset();
        pending_requests_.clear();
        pending_read_completion_pops_.clear();
        pending_write_completion_pops_.clear();
    }

    void high() override {}

    void configure(Ddr4Config config)
    {
        if (!idle())
            throw std::logic_error(
                "cannot reconfigure DDR4 while requests are active");
        config.validate();
        config_ = config;
        state_.bandwidth_remainder = 0;
        state_.round_robin_cursor = 0;
        state_.read_bytes_transferred = 0;
        state_.write_bytes_transferred = 0;
    }

    void reset_execution_state()
    {
        state_ = {};
        next_.reset();
        pending_requests_.clear();
        pending_read_completion_pops_.clear();
        pending_write_completion_pops_.clear();
        next_memory_writes_.clear();
    }

    const Ddr4Config& config() const noexcept { return config_; }
    std::size_t read_vector_service_cycles() const noexcept
    {
        const auto transferCycles = static_cast<std::size_t>(
            (static_cast<std::uint64_t>(hw::kPhysicalVectorBytes)
                    * config_.lpu_clock_hz
                + config_.peak_bandwidth_bytes_per_second - 1)
            / config_.peak_bandwidth_bytes_per_second);
        const auto beatCycles = (hw::kPhysicalVectorBytes + config_.beat_bytes - 1)
            / config_.beat_bytes;
        return config_.read_latency_cycles
            + config_.read_latency_jitter_cycles + std::max(transferCycles, beatCycles) + 1;
    }
    std::size_t cycle() const noexcept { return state_.cycle; }
    bool idle() const noexcept
    {
        return state_.requests.empty() && state_.active.empty()
            && pending_requests_.empty();
    }
    bool drained() const noexcept {
        return idle()
            && state_.read_completions.size()
                == pending_read_completion_pops_.size()
            && state_.write_completions.size()
                == pending_write_completion_pops_.size();
    }
    const std::optional<BeatTrace>& last_beat() const noexcept
    {
        return state_.last_beat;
    }
    std::uint64_t read_bytes_transferred() const noexcept
    {
        return state_.read_bytes_transferred;
    }
    std::uint64_t write_bytes_transferred() const noexcept
    {
        return state_.write_bytes_transferred;
    }
    std::uint64_t bytes_transferred() const noexcept
    {
        return state_.read_bytes_transferred
            + state_.write_bytes_transferred;
    }
    bool can_accept_request() const noexcept
    {
        return state_.requests.size() + state_.active.size()
            + pending_requests_.size()
            < config_.request_queue_depth;
    }

    RequestId request_read(std::uint64_t address,
        std::optional<std::size_t> transport_lane = std::nullopt)
    {
        require_input_window("read request");
        validate_transport_lane(transport_lane);
        require_request_credit();
        validate_vector_range(address);
        const auto id = allocate_request_id();
        auto request = Request {
            id, Operation::Read, address, {},
            config_.read_latency_cycles
                + latency_jitter(id, address, Operation::Read),transport_lane};
        if (clock_managed_) pending_requests_.push_back(std::move(request));
        else state_.requests.push_back(std::move(request));
        return id;
    }

    RequestId request_write(std::uint64_t address, C2cVector vector,
        std::optional<std::size_t> transport_lane = std::nullopt)
    {
        require_input_window("write request");
        validate_transport_lane(transport_lane);
        require_request_credit();
        validate_vector_range(address);
        const auto id = allocate_request_id();
        auto request = Request {
            id,
            Operation::Write,
            address,
            std::move(vector),
            config_.write_latency_cycles
                + latency_jitter(id, address, Operation::Write),
            transport_lane,
        };
        if (clock_managed_) pending_requests_.push_back(std::move(request));
        else state_.requests.push_back(std::move(request));
        return id;
    }

    bool read_completion_ready(RequestId id) const noexcept
    {
        return std::any_of(
            state_.read_completions.begin(), state_.read_completions.end(),
            [this, id](const ReadCompletion& completion) {
                return completion.id == id
                    && std::find(pending_read_completion_pops_.begin(),
                           pending_read_completion_pops_.end(), id)
                        == pending_read_completion_pops_.end();
            });
    }

    ReadCompletion pop_read_completion(RequestId id)
    {
        require_input_window("read completion pop");
        if (!read_completion_ready(id)) {
            throw std::logic_error(
                "DDR4 read completion is not ready or is out of order");
        }
        const auto it = std::find_if(
            state_.read_completions.begin(), state_.read_completions.end(),
            [id](const ReadCompletion& completion) {
                return completion.id == id;
            });
        auto completion = *it;
        if (clock_managed_)
            pending_read_completion_pops_.push_back(id);
        else
            state_.read_completions.erase(it);
        return completion;
    }

    bool write_completion_ready(RequestId id) const noexcept
    {
        return std::any_of(
            state_.write_completions.begin(), state_.write_completions.end(),
            [this, id](const WriteCompletion& completion) {
                return completion.id == id
                    && std::find(pending_write_completion_pops_.begin(),
                           pending_write_completion_pops_.end(), id)
                        == pending_write_completion_pops_.end();
            });
    }

    WriteCompletion pop_write_completion(RequestId id)
    {
        require_input_window("write completion pop");
        if (!write_completion_ready(id)) {
            throw std::logic_error(
                "DDR4 write completion is not ready or is out of order");
        }
        const auto it = std::find_if(
            state_.write_completions.begin(), state_.write_completions.end(),
            [id](const WriteCompletion& completion) {
                return completion.id == id;
            });
        const auto completion = *it;
        if (clock_managed_)
            pending_write_completion_pops_.push_back(id);
        else
            state_.write_completions.erase(it);
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
        if (clock_managed_) {
            low();
            validate_posedge();
            posedge();
            high();
            return;
        }
        tick_immediate();
    }

private:
    enum class Operation { Read, Write };
    static constexpr std::size_t kTransportLaneCount =
        2 * hw::kHemispheres * hw::kC2cStreamsPerDirection;
    static void validate_transport_lane(std::optional<std::size_t> lane)
    {
        if (lane && *lane >= kTransportLaneCount)
            throw std::out_of_range("DDR request transport lane is outside the C2C fabric");
    }

    struct Request {
        RequestId id{0};
        Operation operation{Operation::Read};
        std::uint64_t address{0};
        C2cVector vector{};
        std::size_t remaining_latency{0};
        // DMA requests identify a physical direction/lane; aggregate DDR
        // concurrency must not service that lane more than once per cycle.
        std::optional<std::size_t> transport_lane{};
    };

    struct ActiveRequest {
        Request request{};
        std::size_t bytes_transferred{0};
    };

    struct State {
        std::deque<Request> requests{};
        std::vector<ActiveRequest> active{};
        std::deque<ReadCompletion> read_completions{};
        std::deque<WriteCompletion> write_completions{};
        std::optional<BeatTrace> last_beat{};
        RequestId next_request_id{1};
        std::size_t cycle{0};
        std::uint64_t bandwidth_remainder{0};
        std::size_t round_robin_cursor{0};
        std::uint64_t read_bytes_transferred{0};
        std::uint64_t write_bytes_transferred{0};
    };

    struct PendingMemoryWrite {
        std::uint64_t address{0};
        std::uint8_t value{0};
    };

    void tick_immediate()
    {
        next_memory_writes_.clear();
        tick_state(state_);
        commit_memory_writes();
    }

    void tick_state(State& state)
    {
        state.last_beat.reset();
        while (!state.requests.empty()) {
            state.active.push_back(
                ActiveRequest {std::move(state.requests.front()), 0});
            state.requests.pop_front();
        }

        state.bandwidth_remainder +=
            config_.peak_bandwidth_bytes_per_second;
        std::uint64_t byteBudget =
            state.bandwidth_remainder / config_.lpu_clock_hz;
        state.bandwidth_remainder %= config_.lpu_clock_hz;

        std::size_t channelsUsed = 0;
        std::array<bool,kTransportLaneCount> laneUsed{};
        const std::size_t activeCount = state.active.size();
        const std::size_t start = activeCount == 0
            ? 0 : state.round_robin_cursor % activeCount;
        for (std::size_t visited = 0;
             visited < activeCount && byteBudget != 0
                && channelsUsed < config_.transfer_channels;
             ++visited) {
            auto& active = state.active[(start + visited) % activeCount];
            if (active.request.remaining_latency != 0) continue;
            if (active.request.transport_lane
                && laneUsed[*active.request.transport_lane])
                continue;
            const auto remaining =
                hw::kPhysicalVectorBytes - active.bytes_transferred;
            const auto byteCount = static_cast<std::size_t>(
                std::min<std::uint64_t>(
                    {config_.beat_bytes, remaining, byteBudget}));
            transfer_active_beat(state, active, byteCount);
            if (active.request.transport_lane)
                laneUsed[*active.request.transport_lane] = true;
            byteBudget -= byteCount;
            ++channelsUsed;
        }
        if (activeCount != 0)
            state.round_robin_cursor = (start + 1) % activeCount;
        for (auto& active : state.active) {
            if (active.request.remaining_latency != 0)
                --active.request.remaining_latency;
        }
        for (std::size_t index = state.active.size(); index-- > 0;) {
            if (state.active[index].bytes_transferred
                == hw::kPhysicalVectorBytes)
                complete_active_request(state, index);
        }
        ++state.cycle;
    }

    void complete_active_request(State& state, std::size_t index)
    {
        auto request = std::move(state.active[index].request);
        state.active.erase(state.active.begin() + index);
        if (request.operation == Operation::Read) {
            state.read_completions.push_back(ReadCompletion {
                request.id, request.address, std::move(request.vector)});
            return;
        }
        state.write_completions.push_back(
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

    void require_input_window(const char* operation) const
    {
        if (clock_managed_ && next_.has_value())
            throw std::logic_error(std::string("DDR4 ") + operation
                + " arrived after DDR4 LOW");
    }

    void apply_pending_inputs(State& state)
    {
        for (const auto id : pending_read_completion_pops_) {
            const auto found = std::find_if(
                state.read_completions.begin(),
                state.read_completions.end(),
                [id](const auto& completion) {
                    return completion.id == id;
                });
            if (found == state.read_completions.end())
                throw std::logic_error(
                    "DDR4 staged read completion disappeared");
            state.read_completions.erase(found);
        }
        for (const auto id : pending_write_completion_pops_) {
            const auto found = std::find_if(
                state.write_completions.begin(),
                state.write_completions.end(),
                [id](const auto& completion) {
                    return completion.id == id;
                });
            if (found == state.write_completions.end())
                throw std::logic_error(
                    "DDR4 staged write completion disappeared");
            state.write_completions.erase(found);
        }
        state.next_request_id +=
            static_cast<RequestId>(pending_requests_.size());
        for (const auto& request : pending_requests_)
            state.requests.push_back(request);
    }

    RequestId allocate_request_id()
    {
        const auto pending = static_cast<RequestId>(pending_requests_.size());
        if (state_.next_request_id == 0
            || pending > std::numeric_limits<RequestId>::max()
                    - state_.next_request_id) {
            throw std::overflow_error("DDR4 request ID space is exhausted");
        }
        if (clock_managed_)
            return state_.next_request_id + pending;
        return state_.next_request_id++;
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

    std::uint8_t read_storage_byte(std::uint64_t address) const
    {
        const auto staged = std::find_if(
            next_memory_writes_.rbegin(), next_memory_writes_.rend(),
            [address](const PendingMemoryWrite& write) {
                return write.address == address;
            });
        if (staged != next_memory_writes_.rend()) return staged->value;
        const auto page = pages_.find(address / kStoragePageBytes);
        return page == pages_.end() ? 0
            : page->second[static_cast<std::size_t>(
                  address % kStoragePageBytes)];
    }

    void commit_memory_writes()
    {
        for (const auto& write : next_memory_writes_)
            pages_[write.address / kStoragePageBytes]
                [static_cast<std::size_t>(
                    write.address % kStoragePageBytes)] = write.value;
        next_memory_writes_.clear();
    }

    void transfer_active_beat(
        State& state, ActiveRequest& active, std::size_t byte_count)
    {
        const auto offset = active.bytes_transferred;
        for (std::size_t byte = 0; byte < byte_count; ++byte) {
            const auto vector_offset = offset + byte;
            const auto tile = vector_offset / hw::kLanesPerTile;
            const auto lane = vector_offset % hw::kLanesPerTile;
            const auto address = active.request.address + vector_offset;
            if (active.request.operation == Operation::Read) {
                active.request.vector.payload[tile][lane] =
                    read_storage_byte(address);
            } else {
                next_memory_writes_.push_back(PendingMemoryWrite {
                    address, active.request.vector.payload[tile][lane]});
            }
        }
        active.bytes_transferred += byte_count;
        if (active.request.operation == Operation::Read)
            state.read_bytes_transferred += byte_count;
        else
            state.write_bytes_transferred += byte_count;
        state.last_beat = BeatTrace {
            active.request.id,
            active.request.operation == Operation::Write,
            active.request.address + offset,
            byte_count,
        };
    }

    State state_{};
    std::optional<State> next_{};
    std::deque<Request> pending_requests_{};
    std::vector<RequestId> pending_read_completion_pops_{};
    std::vector<RequestId> pending_write_completion_pops_{};
    std::vector<PendingMemoryWrite> next_memory_writes_{};
    bool clock_managed_{false};
};

} // namespace ftlpu
