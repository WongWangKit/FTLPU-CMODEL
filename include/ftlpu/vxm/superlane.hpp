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

    using Int32Vector = std::array<std::int32_t, kLaneCount>;
    using Int8Vector = std::array<std::int8_t, kLaneCount>;
    using StreamBytes = VxmLane::StreamBytes;
    using StreamMatrix = std::array<StreamBytes, kLaneCount>;

    struct Output {
        Int8Vector values{};
        std::array<std::array<std::uint8_t, 4>, kLaneCount> byte_values{};
        std::size_t stream{0};
        std::size_t byte_count{1};
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

    void load_swiglu_program(const VxmLane::SwigluParams& params)
    {
        for (auto& lane : lanes_) {
            lane.load_swiglu_program(params);
        }
    }

    void load_pipelined_swiglu_program(
        const VxmLane::SwigluParams& params,
        std::size_t token_count,
        std::size_t gate_stream_base = 0,
        std::size_t up_stream_base = 4,
        std::size_t output_stream = 0)
    {
        for (auto& lane : lanes_) {
            lane.load_pipelined_swiglu_program(
                params,
                token_count,
                gate_stream_base,
                up_stream_base,
                output_stream);
        }
    }

    void load_pipelined_add_quant_program(
        const VxmLane::AddQuantParams& params,
        std::size_t token_count,
        std::size_t lhs_stream_base = 0,
        std::size_t rhs_stream_base = 4,
        std::size_t output_stream = 0)
    {
        for (auto& lane : lanes_) {
            lane.load_pipelined_add_quant_program(
                params,
                token_count,
                lhs_stream_base,
                rhs_stream_base,
                output_stream);
        }
    }

    void enqueue_instruction(std::size_t alu, VxmLaneAluInstruction instruction)
    {
        for (auto& lane : lanes_) {
            lane.enqueue_instruction(alu, instruction);
        }
    }

    void set_stream_inputs(const StreamMatrix& streams)
    {
        for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
            lanes_[lane].set_stream_inputs(streams[lane]);
        }
    }

    void set_swiglu_inputs(const Int32Vector& gates, const Int32Vector& ups)
    {
        for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
            lanes_[lane].set_swiglu_input(
                VxmLane::pack_int32(gates[lane]),
                VxmLane::pack_int32(ups[lane]));
        }
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
            for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
                const auto& lane_output = lanes_[lane].outputs()[output_index];
                if (lane_output.stream != stream) {
                    throw std::logic_error("VXM superlane lanes produced outputs on different streams");
                }
                if (lane_output.byte_count != byte_count) {
                    throw std::logic_error("VXM superlane lanes produced outputs with different byte widths");
                }
                values[lane] = lane_output.value;
                byte_values[lane] = lane_output.bytes;
            }
            outputs_.push_back(Output {values, byte_values, stream, byte_count});
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
