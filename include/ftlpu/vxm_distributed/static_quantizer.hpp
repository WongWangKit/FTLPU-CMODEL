#pragma once

#include "ftlpu/vxm_distributed/data_format.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ftlpu::distributed_vxm {

// Optional boundary hardware after a VXM output lane. Int8 requests enter
// this fixed one-cycle pipeline; there is no ready signal or replay path.
// Static scale/zero-point values are supplied by the compiler with the
// scheduled output configuration.
class VxmStaticQuantizer {
public:
    static constexpr std::size_t kLatency = 1;

    struct Request {
        float fp16_value{0.0f};
        float scale{1.0f};
        std::int32_t zero_point{0};
        std::size_t stream{0};
    };

    struct Result {
        std::int8_t value{0};
        std::size_t stream{0};
    };

    void reset()
    {
        pending_.clear();
    }

    std::vector<Result> tick(std::vector<Request> requests = {})
    {
        if (requests.size() > 1) {
            throw std::logic_error(
                "one VXM output quantizer accepts at most one value per cycle");
        }
        auto completed = std::move(pending_);
        pending_.clear();
        pending_.reserve(requests.size());
        for (const auto& request : requests) {
            // The converter consumes FP16 at the VXM boundary, even though
            // the C++ ALU model carries the value in a float container.
            const auto fp16 = VxmDataFormat::round_fp16_ftz(
                request.fp16_value);
            pending_.push_back(Result {
                VxmDataFormat::quantize_int8(
                    fp16, request.scale, request.zero_point),
                request.stream});
        }
        return completed;
    }

    bool idle() const noexcept
    {
        return pending_.empty();
    }

private:
    std::vector<Result> pending_{};
};

} // namespace ftlpu::distributed_vxm
