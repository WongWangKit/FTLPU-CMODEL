#pragma once

#include "ftlpu/axi/memory_backend.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ftlpu::axi {

struct Axi4Config {
    std::size_t addr_width_bits{64};
    std::size_t data_width_bits{256};
    std::size_t id_width_bits{4};
    std::size_t max_burst_beats{32};
    std::size_t max_read_outstanding{1};
    std::size_t max_write_outstanding{4};
    std::size_t read_response_fifo_beats{16};
    std::size_t write_data_fifo_beats{16};

    void validate() const
    {
        if (addr_width_bits == 0 || addr_width_bits > 64)
            throw std::invalid_argument(
                "AXI address width must be in [1, 64]");
        if (id_width_bits == 0 || id_width_bits > 32)
            throw std::invalid_argument(
                "AXI ID width must be in [1, 32]");
        if (data_width_bits == 0 || data_width_bits % 8 != 0
            || !std::has_single_bit(data_width_bits / 8))
            throw std::invalid_argument(
                "AXI data width must contain a power-of-two byte count");
        if (max_burst_beats == 0 || max_burst_beats > 256)
            throw std::invalid_argument(
                "AXI maximum burst length must be in [1, 256]");
        if (max_read_outstanding == 0
            || max_write_outstanding == 0)
            throw std::invalid_argument(
                "AXI outstanding limits must be non-zero");
        if (read_response_fifo_beats == 0
            || write_data_fifo_beats == 0)
            throw std::invalid_argument(
                "AXI channel FIFO depths must be non-zero");
    }

    std::size_t data_bytes() const noexcept
    {
        return data_width_bits / 8;
    }
};

enum class AxiBurst : std::uint8_t {
    Fixed = 0,
    Incrementing = 1,
    Wrapping = 2,
};

enum class AxiResponse : std::uint8_t {
    Okay = 0,
    ExclusiveOkay = 1,
    SlaveError = 2,
    DecodeError = 3,
};

struct Axi4ReadInputs {
    bool arvalid{false};
    std::uint64_t araddr{0};
    std::uint8_t arlen{0};
    std::uint8_t arsize{0};
    AxiBurst arburst{AxiBurst::Incrementing};
    std::uint32_t arid{0};
    bool rready{false};
};

struct Axi4ReadOutputs {
    bool arready{false};
    bool rvalid{false};
    std::vector<std::uint8_t> rdata{};
    bool rlast{false};
    std::uint32_t rid{0};
    AxiResponse rresp{AxiResponse::Okay};
};

struct Axi4ReadCycle {
    std::size_t cycle{0};
    Axi4ReadOutputs outputs{};
    bool ar_fire{false};
    bool r_fire{false};
    std::size_t backend_bytes_serviced{0};
    std::size_t response_fifo_occupancy{0};
};

struct Axi4ReadTransactionTrace {
    std::uint64_t sequence{0};
    std::uint32_t id{0};
    std::uint64_t address{0};
    std::size_t beat_bytes{0};
    std::size_t beat_count{0};
    std::size_t byte_count{0};
    std::size_t ar_accept_cycle{0};
    std::optional<std::size_t> first_rvalid_cycle{};
    std::optional<std::size_t> first_r_handshake_cycle{};
    std::optional<std::size_t> last_r_handshake_cycle{};
};

struct Axi4ReadStatistics {
    std::size_t cycle_count{0};
    std::size_t ar_accepted{0};
    std::size_t r_beats_completed{0};
    std::uint64_t transferred_bytes{0};
    std::uint64_t backend_bytes_serviced{0};
    std::size_t outstanding_peak{0};
    std::size_t response_fifo_peak{0};
    std::size_t r_stall_cycles{0};
    std::size_t data_wait_cycles{0};

    double effective_bytes_per_cycle() const noexcept
    {
        return cycle_count == 0
            ? 0.0
            : static_cast<double>(transferred_bytes)
                / static_cast<double>(cycle_count);
    }
};

// Cycle-level AXI4 read slave. AW/W/B configuration fields are reserved in
// Axi4Config, but write channels are intentionally not implemented here.
class Axi4MemorySlave {
public:
    Axi4MemorySlave(
        Axi4Config config, ReadMemoryBackend& memory_backend)
        : config_(config)
        , memory_backend_(memory_backend)
    {
        config_.validate();
    }

    explicit Axi4MemorySlave(ReadMemoryBackend& memory_backend)
        : Axi4MemorySlave(Axi4Config {}, memory_backend)
    {
    }

    const Axi4Config& config() const noexcept { return config_; }
    std::size_t cycle() const noexcept { return cycle_; }
    std::size_t read_outstanding() const noexcept
    {
        return read_outstanding_;
    }
    std::size_t response_fifo_occupancy() const noexcept
    {
        return responses_.size();
    }
    const Axi4ReadStatistics& statistics() const noexcept
    {
        return statistics_;
    }
    const std::vector<Axi4ReadTransactionTrace>& traces() const noexcept
    {
        return traces_;
    }

    Axi4ReadOutputs evaluate() const
    {
        auto outputs = Axi4ReadOutputs {};
        outputs.arready =
            read_outstanding_ < config_.max_read_outstanding
            && memory_backend_.can_accept_read();
        outputs.rdata.assign(config_.data_bytes(), 0);
        if (responses_.empty())
            return outputs;

        const auto& response = responses_.front();
        outputs.rvalid = true;
        outputs.rdata = response.data;
        outputs.rlast = response.last;
        outputs.rid = response.id;
        outputs.rresp = response.response;
        return outputs;
    }

    // One call represents evaluate -> handshake -> commit. The returned
    // outputs are the signals observed for this cycle before state advances.
    Axi4ReadCycle tick(const Axi4ReadInputs& inputs)
    {
        auto result = Axi4ReadCycle {};
        result.cycle = cycle_;
        result.outputs = evaluate();
        result.ar_fire = inputs.arvalid && result.outputs.arready;
        result.r_fire = result.outputs.rvalid && inputs.rready;

        if (result.ar_fire)
            validate_read_request(inputs);

        if (read_outstanding_ != 0 && !result.outputs.rvalid)
            ++statistics_.data_wait_cycles;
        if (result.outputs.rvalid && !inputs.rready)
            ++statistics_.r_stall_cycles;

        if (result.outputs.rvalid) {
            auto& trace = traces_[responses_.front().trace_index];
            if (!trace.first_rvalid_cycle.has_value())
                trace.first_rvalid_cycle = cycle_;
            if (result.r_fire
                && !trace.first_r_handshake_cycle.has_value())
                trace.first_r_handshake_cycle = cycle_;
        }

        const auto occupancyAfterHandshake = responses_.size()
            - static_cast<std::size_t>(result.r_fire);
        const auto responseCredits =
            config_.read_response_fifo_beats - occupancyAfterHandshake;
        auto backendResult = memory_backend_.tick(responseCredits);

        if (result.r_fire) {
            auto response = std::move(responses_.front());
            responses_.pop_front();
            ++statistics_.r_beats_completed;
            statistics_.transferred_bytes += response.transfer_bytes;
            if (response.last) {
                if (read_outstanding_ == 0)
                    throw std::logic_error(
                        "AXI read outstanding counter underflow");
                --read_outstanding_;
                traces_[response.trace_index].last_r_handshake_cycle =
                    cycle_;
                pending_reads_.erase(response.backend_request_id);
            }
        }

        for (auto& beat : backendResult.completed_beats)
            enqueue_backend_beat(std::move(beat));
        statistics_.backend_bytes_serviced +=
            backendResult.bytes_serviced;

        if (result.ar_fire)
            accept_read_request(inputs);

        statistics_.response_fifo_peak = std::max(
            statistics_.response_fifo_peak, responses_.size());
        result.backend_bytes_serviced = backendResult.bytes_serviced;
        result.response_fifo_occupancy = responses_.size();

        ++cycle_;
        statistics_.cycle_count = cycle_;
        return result;
    }

    void reset_execution_state()
    {
        memory_backend_.reset_execution_state();
        responses_.clear();
        pending_reads_.clear();
        traces_.clear();
        cycle_ = 0;
        read_outstanding_ = 0;
        next_sequence_ = 0;
        statistics_ = {};
    }

private:
    struct PendingRead {
        std::uint32_t id{0};
        std::size_t trace_index{0};
    };

    struct ResponseBeat {
        ReadMemoryBackend::RequestId backend_request_id{0};
        std::uint32_t id{0};
        std::vector<std::uint8_t> data{};
        std::size_t transfer_bytes{0};
        bool last{false};
        AxiResponse response{AxiResponse::Okay};
        std::size_t trace_index{0};
    };

    void validate_read_request(const Axi4ReadInputs& inputs) const
    {
        if (inputs.arburst != AxiBurst::Incrementing)
            throw std::invalid_argument(
                "AXI read model currently supports only INCR bursts");

        const auto beatCount =
            static_cast<std::size_t>(inputs.arlen) + 1;
        if (beatCount > config_.max_burst_beats)
            throw std::invalid_argument(
                "AXI read burst exceeds configured maximum");
        if (inputs.arsize >= std::numeric_limits<std::size_t>::digits)
            throw std::invalid_argument("AXI ARSIZE is too large");
        const auto beatBytes = std::size_t {1} << inputs.arsize;
        if (beatBytes > config_.data_bytes())
            throw std::invalid_argument(
                "AXI ARSIZE exceeds the configured data width");
        if (inputs.araddr % beatBytes != 0)
            throw std::invalid_argument(
                "AXI read model requires beat-aligned INCR addresses");

        if (config_.addr_width_bits < 64
            && inputs.araddr
                >= (std::uint64_t {1} << config_.addr_width_bits))
            throw std::invalid_argument(
                "AXI ARADDR exceeds the configured address width");
        if (config_.id_width_bits < 32
            && inputs.arid
                >= (std::uint32_t {1} << config_.id_width_bits))
            throw std::invalid_argument(
                "AXI ARID exceeds the configured ID width");

        if (beatCount
            > std::numeric_limits<std::size_t>::max() / beatBytes)
            throw std::out_of_range("AXI read burst byte count overflows");
        const auto byteCount = beatCount * beatBytes;
        if (byteCount - 1
            > std::numeric_limits<std::uint64_t>::max()
                - inputs.araddr)
            throw std::out_of_range(
                "AXI read burst crosses the address space");
        const auto lastAddress = inputs.araddr + byteCount - 1;
        if ((inputs.araddr >> 12) != (lastAddress >> 12))
            throw std::invalid_argument(
                "AXI read burst crosses a 4 KB boundary");
    }

    void accept_read_request(const Axi4ReadInputs& inputs)
    {
        const auto beatCount =
            static_cast<std::size_t>(inputs.arlen) + 1;
        const auto beatBytes = std::size_t {1} << inputs.arsize;
        const auto traceIndex = traces_.size();
        const auto backendRequest = memory_backend_.request_read(
            inputs.araddr, beatBytes, beatCount);

        traces_.push_back(Axi4ReadTransactionTrace {
            next_sequence_++,
            inputs.arid,
            inputs.araddr,
            beatBytes,
            beatCount,
            beatBytes * beatCount,
            cycle_,
        });
        pending_reads_.emplace(
            backendRequest, PendingRead {inputs.arid, traceIndex});
        ++read_outstanding_;
        ++statistics_.ar_accepted;
        statistics_.outstanding_peak = std::max(
            statistics_.outstanding_peak, read_outstanding_);
    }

    void enqueue_backend_beat(ReadMemoryBackend::ReadBeat beat)
    {
        const auto pending = pending_reads_.find(beat.request_id);
        if (pending == pending_reads_.end())
            throw std::logic_error(
                "memory backend returned an unknown read request");
        if (responses_.size() >= config_.read_response_fifo_beats)
            throw std::logic_error(
                "AXI read response FIFO capacity was exceeded");

        auto data = std::vector<std::uint8_t>(config_.data_bytes(), 0);
        const auto lane = static_cast<std::size_t>(
            beat.address % config_.data_bytes());
        if (lane + beat.data.size() > data.size())
            throw std::logic_error(
                "memory beat crosses an AXI data-bus word");
        std::copy(
            beat.data.begin(), beat.data.end(), data.begin() + lane);
        responses_.push_back(ResponseBeat {
            beat.request_id,
            pending->second.id,
            std::move(data),
            beat.data.size(),
            beat.last,
            AxiResponse::Okay,
            pending->second.trace_index,
        });
    }

    Axi4Config config_{};
    ReadMemoryBackend& memory_backend_;
    std::deque<ResponseBeat> responses_{};
    std::unordered_map<ReadMemoryBackend::RequestId, PendingRead>
        pending_reads_{};
    std::vector<Axi4ReadTransactionTrace> traces_{};
    std::size_t cycle_{0};
    std::size_t read_outstanding_{0};
    std::uint64_t next_sequence_{0};
    Axi4ReadStatistics statistics_{};
};

} // namespace ftlpu::axi
