#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/vxm/lane.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace ftlpu {

class VxmSuperlane {
public:
    static constexpr std::size_t kLaneCount = hw::kLanesPerTile;

    using Int8Vector = std::array<std::int8_t, kLaneCount>;
    using StreamBytes = VxmLane::StreamBytes;
    using StreamMatrix = std::array<StreamBytes, kLaneCount>;

    struct Output {
        Int8Vector values{};
        std::array<std::array<std::uint8_t, 4>, kLaneCount> byte_values{};
        std::size_t stream{0};
        std::size_t byte_count{1};
        Hemisphere hemisphere{Hemisphere::East};
    };

    void reset()
    {
        for (auto& lane : lanes_) {
            lane.reset();
        }
        output_.reset();
        outputs_.clear();
        cycle_ = 0;
    }

    void enqueue_instruction(std::size_t alu, VxmLaneAluInstruction instruction)
    {
        for (auto& lane : lanes_) {
            lane.enqueue_instruction(alu, instruction);
        }
    }

    void set_stream_inputs(Hemisphere hemisphere, const StreamMatrix& streams)
    {
        for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
            lanes_[lane].set_stream_inputs(hemisphere, streams[lane]);
        }
    }

    void set_stream_inputs(const StreamMatrix& streams)
    {
        set_stream_inputs(Hemisphere::East, streams);
    }

    void tick()
    {
        output_.reset();
        outputs_.clear();

        for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
            lanes_[lane].tick();
        }

        const auto output_count = lanes_[0].outputs().size();
        for (std::size_t lane = 1; lane < kLaneCount; ++lane) {
            if (lanes_[lane].outputs().size() != output_count) {
                throw std::logic_error("VXM superlane lanes produced different output counts");
            }
        }

        for (std::size_t output_index = 0; output_index < output_count; ++output_index) {
            auto values = Int8Vector {};
            auto byte_values = std::array<std::array<std::uint8_t, 4>, kLaneCount> {};
            const auto stream = lanes_[0].outputs()[output_index].stream;
            const auto byte_count = lanes_[0].outputs()[output_index].byte_count;
            const auto hemisphere = lanes_[0].outputs()[output_index].hemisphere;
            for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
                const auto& lane_output = lanes_[lane].outputs()[output_index];
                if (lane_output.stream != stream) {
                    throw std::logic_error("VXM superlane lanes produced outputs on different streams");
                }
                if (lane_output.byte_count != byte_count) {
                    throw std::logic_error("VXM superlane lanes produced outputs with different byte widths");
                }
                if (lane_output.hemisphere != hemisphere) {
                    throw std::logic_error("VXM superlane lanes produced outputs for different hemispheres");
                }
                values[lane] = lane_output.value;
                byte_values[lane] = lane_output.bytes;
            }
            outputs_.push_back(Output {values, byte_values, stream, byte_count, hemisphere});
        }
        if (!outputs_.empty()) {
            output_ = outputs_.front();
        }

        ++cycle_;
    }

    const std::optional<Output>& output() const
    {
        return output_;
    }

    const std::vector<Output>& outputs() const
    {
        return outputs_;
    }

    const VxmLane& lane(std::size_t index) const
    {
        check_lane(index);
        return lanes_[index];
    }

    VxmLane& lane(std::size_t index)
    {
        check_lane(index);
        return lanes_[index];
    }

    std::size_t cycle() const
    {
        return cycle_;
    }

    void print_lane_trace(std::ostream& os, std::size_t lane_index) const
    {
        lane(lane_index).print_last_trace(os);
    }

private:
    static void check_lane(std::size_t index)
    {
        if (index >= kLaneCount) {
            throw std::out_of_range("VXM superlane lane index is outside the 16-lane superlane");
        }
    }

    std::array<VxmLane, kLaneCount> lanes_{};
    std::optional<Output> output_{};
    std::vector<Output> outputs_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
