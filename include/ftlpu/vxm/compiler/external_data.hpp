#pragma once

#include "ftlpu/vxm/compiler/kernel_ir.hpp"
#include "ftlpu/vxm/superlane.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace ftlpu::vxm::compiler {

using VxmSuperlaneValues =
    std::array<float, VxmSuperlane::kLaneCount>;

// This request belongs to the simulation environment outside VXM. It models
// the compiler-scheduled producer of Stream Register data, not a dynamic
// request/ready signal implemented inside the VXM hardware.
struct VxmExternalStreamRequest {
    std::size_t issue_cycle{0};
    std::size_t required_cycle{0};
    std::size_t phase_id{0};
    std::size_t superlane{0};
    ValueId value{kInvalidValue};
    std::size_t stream_base{0};
    std::size_t element_index{0};
    bool hold{false};
    std::size_t reuse_count{1};
};

struct VxmExternalOutputEvent {
    std::size_t cycle{0};
    // The VXM result is produced at cycle. A later MEM read may be issued
    // only after sram_visible_cycle.
    std::size_t sram_visible_cycle{0};
    std::size_t phase_id{0};
    std::size_t superlane{0};
    ValueId value{kInvalidValue};
    std::size_t stream_base{0};
    std::size_t element_index{0};
};

class VxmExternalDataModel {
public:
    virtual ~VxmExternalDataModel() = default;

    // Called when the statically scheduled request is issued. The returned
    // value is held by the adapter until request.required_cycle.
    virtual VxmSuperlaneValues read(
        const VxmExternalStreamRequest& request) = 0;

    // Receives both intermediate and final VXM outputs.
    virtual void write(
        const VxmExternalOutputEvent& event,
        const VxmSuperlaneValues& values) = 0;
};

// A simple host-side value store for tests and early compiler integration.
// Layout is [ValueId][Superlane][Lane][element].
class VxmHostValueStore final : public VxmExternalDataModel {
public:
    using LaneVectors =
        std::array<std::vector<float>, VxmSuperlane::kLaneCount>;

    void set(ValueId value, std::size_t superlane, std::size_t lane,
             std::vector<float> elements)
    {
        check_lane(lane);
        auto& tiles = values_[value];
        if (tiles.size() <= superlane) tiles.resize(superlane + 1);
        tiles[superlane][lane] = std::move(elements);
    }

    const std::vector<float>& get(
        ValueId value, std::size_t superlane, std::size_t lane) const
    {
        check_lane(lane);
        const auto found = values_.find(value);
        if (found == values_.end() || found->second.size() <= superlane) {
            throw std::out_of_range(
                "VXM host value is unavailable for this Superlane");
        }
        return found->second[superlane][lane];
    }

    VxmSuperlaneValues read(
        const VxmExternalStreamRequest& request) override
    {
        auto result = VxmSuperlaneValues{};
        for (std::size_t lane = 0;
             lane < VxmSuperlane::kLaneCount; ++lane) {
            const auto& elements =
                get(request.value, request.superlane, lane);
            if (request.element_index >= elements.size()) {
                throw std::out_of_range(
                    "VXM host value element is unavailable");
            }
            result[lane] = elements[request.element_index];
        }
        return result;
    }

    void write(
        const VxmExternalOutputEvent& event,
        const VxmSuperlaneValues& lane_values) override
    {
        auto& tiles = values_[event.value];
        if (tiles.size() <= event.superlane) {
            tiles.resize(event.superlane + 1);
        }
        for (std::size_t lane = 0;
             lane < VxmSuperlane::kLaneCount; ++lane) {
            auto& elements = tiles[event.superlane][lane];
            if (elements.size() <= event.element_index) {
                elements.resize(event.element_index + 1);
            }
            elements[event.element_index] = lane_values[lane];
        }
    }

private:
    static void check_lane(std::size_t lane)
    {
        if (lane >= VxmSuperlane::kLaneCount) {
            throw std::out_of_range(
                "VXM host value lane exceeds the configured Superlane lane count");
        }
    }

    std::unordered_map<ValueId, std::vector<LaneVectors>> values_{};
};

} // namespace ftlpu::vxm::compiler
