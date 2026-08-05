#pragma once

#include "ftlpu/core/fp16.hpp"
#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/stream_port.hpp"
#include "ftlpu/mxm/control_slice.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <stdexcept>

namespace ftlpu {

// Fixed one-cycle INT8 -> FP16 boundary in front of one MXM.  The compiler
// issues it one cycle before the matching Compute instruction.  Its control
// wave uses exactly the same vertical skew as MXM Compute, so each Superlane
// receives the converted activation on the cycle its local Compute pulse
// arrives.  There is no ready/valid handshake or backpressure path.
class MxmActivationDequantizer {
public:
    static constexpr std::size_t kLatency = 1;
    // Within each MXM's fixed 16-stream window, streams 0/1 are the first
    // FP16 activation group, streams 8..15 are the background IW window, and
    // stream 7 is that MXM's INT8 input wire.  Adjacent MXMs therefore use
    // physical stream 7 and stream 23 rather than sharing one broadcast cell.
    static constexpr std::size_t kInputStreamWithinWindow = 7;
    static constexpr std::size_t kBroadcastInputStream =
        kInputStreamWithinWindow; // source-compatible local-MXM name

    struct Endpoint {
        std::size_t column{0};
        StreamDirection input_direction{StreamDirection::East};
        std::size_t input_stream{0};
        StreamDirection output_direction{StreamDirection::West};
        std::size_t output_stream_base{0};
        bool multicast_input{false};
    };

    struct Statistics {
        std::uint64_t accepted_segments{0};
        std::uint64_t emitted_segments{0};
        std::uint64_t active_cycles{0};
    };

    explicit MxmActivationDequantizer(Endpoint endpoint)
        : endpoint_(endpoint)
    {
        if (endpoint_.input_stream >= hw::kStreamsPerDirection
            || endpoint_.output_stream_base + 2
                > hw::kStreamsPerDirection) {
            throw std::out_of_range(
                "MXM activation dequantizer stream endpoint is outside one direction");
        }
    }

    void reset()
    {
        issue_pending_ = false;
        issue_pipeline_.fill(false);
        for (auto& queue : pending_outputs_) queue.clear();
        statistics_ = {};
        cycle_ = 0;
    }

    void configure_scale(float scale)
    {
        scale_ = scale;
    }

    float scale() const noexcept
    {
        return scale_;
    }

    const Endpoint& endpoint() const noexcept
    {
        return endpoint_;
    }

    // One issue corresponds to one complete MXM activation row.  Its input
    // collection wave advances one tile per cycle (matching MEM), while the
    // registered outputs are delayed to the MXM Compute skew of 0/8/16/24.
    // Calling issue_south every cycle therefore sustains II=1 at both edges.
    void issue_south()
    {
        if (issue_pending_) {
            throw std::logic_error(
                "MXM activation dequantizer received two issues in one cycle");
        }
        issue_pending_ = true;
    }

    void evaluate(StreamRegisterFabric& fabric)
    {
        if (!fabric.cycle_open()) {
            throw std::logic_error(
                "MXM activation dequantizer requires an open SR cycle");
        }
        if (endpoint_.column >= fabric.column_count()) {
            throw std::out_of_range(
                "MXM activation dequantizer column is outside the SR fabric");
        }

        auto input = StreamInputPort(
            fabric, endpoint_.column, endpoint_.input_direction,
            "MXM activation dequantizer");
        auto output = StreamOutputPort(
            fabric, endpoint_.column, endpoint_.output_direction,
            "MXM activation dequantizer");
        auto active = false;

        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            auto& queue = pending_outputs_[tile];
            while (!queue.empty() && queue.front().ready_cycle <= cycle_) {
                output.write_segment(
                    tile, endpoint_.output_stream_base,
                    queue.front().low);
                output.write_segment(
                    tile, endpoint_.output_stream_base + 1,
                    queue.front().high);
                queue.pop_front();
                ++statistics_.emitted_segments;
                active = true;
            }

            const auto capture = tile == 0
                ? issue_pending_ : issue_pipeline_[tile - 1];
            if (!capture) continue;
            if (!input.segment_valid(tile, endpoint_.input_stream)) {
                throw std::logic_error(
                    "MXM activation dequantizer issue reached tile "
                    + std::to_string(tile)
                    + " without its INT8 segment on stream "
                    + std::to_string(endpoint_.input_stream));
            }
            const auto bytes = endpoint_.multicast_input
                ? input.consume_shared_segment(
                      tile, endpoint_.input_stream)
                : input.consume_segment(
                      tile, endpoint_.input_stream);
            auto low = StreamSegment16{};
            auto high = StreamSegment16{};
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                const auto signed_value = static_cast<std::int8_t>(
                    bytes[lane].data);
                const auto bits = Fp16::from_float(
                    static_cast<float>(signed_value) * scale_).bits();
                low[lane] = StreamCell::Valid(
                    static_cast<std::uint8_t>(bits), bytes[lane].last,
                    bytes[lane].vector_tag);
                high[lane] = StreamCell::Valid(
                    static_cast<std::uint8_t>(bits >> 8), bytes[lane].last,
                    bytes[lane].vector_tag);
            }
            const auto ready = cycle_
                + tile * (MxmControlSlice::kComputeTileLatency - 1);
            if (ready == cycle_) {
                output.write_segment(
                    tile, endpoint_.output_stream_base, low);
                output.write_segment(
                    tile, endpoint_.output_stream_base + 1, high);
                ++statistics_.emitted_segments;
            } else {
                queue.push_back(PendingOutput{ready, low, high});
            }
            ++statistics_.accepted_segments;
            active = true;
        }

        for (std::size_t stage = issue_pipeline_.size(); stage > 1; --stage) {
            issue_pipeline_[stage - 1] = issue_pipeline_[stage - 2];
        }
        issue_pipeline_[0] = issue_pending_;
        issue_pending_ = false;
        statistics_.active_cycles += active;
        ++cycle_;
    }

    const Statistics& statistics() const noexcept
    {
        return statistics_;
    }

private:
    struct PendingOutput {
        std::size_t ready_cycle{0};
        StreamSegment16 low{};
        StreamSegment16 high{};
    };

    Endpoint endpoint_{};
    float scale_{1.0f};
    bool issue_pending_{false};
    std::array<bool, MxmControlSlice::kComputePipelineSlots>
        issue_pipeline_{};
    std::array<std::deque<PendingOutput>, hw::kTileRows>
        pending_outputs_{};
    Statistics statistics_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
