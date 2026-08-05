#pragma once

#include "ftlpu/mxm/accumulator_slice.hpp"
#include "ftlpu/vxm/slice.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace ftlpu {

// Token-major collector at the VXM entrance.  An MXM result arrives as one
// token row whose feature values are spread over the MXM column blocks.  The
// physical data plane is banked by VXM lane: row r owns lane r, and successive
// column-block results fill successive feature positions in that lane's token
// vector.  Once every configured chain slot is complete, one feature Bundle is
// issued per cycle to the ordinary VxmInputBuffer.
//
// This is intentionally not a host-side tensor transpose.  It is the modeled
// VXM input storage/controller that converts the MXM result order into the
// fixed stream-group order consumed by the lane chains.
class MxmVxmInputBuffer {
public:
    static constexpr std::size_t kOperandsPerToken = 2;

    explicit MxmVxmInputBuffer(
        std::size_t feature_count,
        VxmChainDepth chain_depth,
        std::size_t capacity_groups = 2)
        : feature_count_(feature_count)
        , chain_depth_(static_cast<std::size_t>(chain_depth))
        , chain_count_(VxmLane::kAluCount / chain_depth_)
        , capacity_groups_(capacity_groups)
    {
        if (feature_count_ == 0
            || chain_depth_ == 0
            || VxmLane::kAluCount % chain_depth_ != 0
            || chain_count_ * chain_depth_ != VxmLane::kAluCount) {
            throw std::invalid_argument(
                "MXM-VXM input Buffer has an invalid feature/chain configuration");
        }
        if (chain_count_ * kOperandsPerToken
            > VxmInputBuffer::kGroupCount) {
            throw std::invalid_argument(
                "MXM-VXM input Buffer requires too many stream groups");
        }
        if (capacity_groups_ == 0) {
            throw std::invalid_argument(
                "MXM-VXM input Buffer capacity must be nonzero");
        }
    }

    std::size_t feature_count() const noexcept { return feature_count_; }
    std::size_t chain_depth() const noexcept { return chain_depth_; }
    std::size_t chain_count() const noexcept { return chain_count_; }
    std::size_t capacity_groups() const noexcept { return capacity_groups_; }
    std::size_t resident_groups() const noexcept { return groups_.size(); }

    void capture(
        std::size_t group,
        std::size_t chain,
        std::size_t operand,
        std::size_t feature_block,
        const MxmAccumulatorSlice::Output& output)
    {
        if (!output.final) {
            throw std::logic_error(
                "MXM-VXM input Buffer cannot capture an uncast partial sum");
        }
        if (chain >= chain_count_ || operand >= kOperandsPerToken) {
            throw std::out_of_range(
                "MXM-VXM input Buffer chain or operand is outside configuration");
        }
        if (output.row >= token_lanes()
            || output.tile >= hw::kTileRows) {
            throw std::out_of_range(
                "MXM-VXM input Buffer MXM row/tile is outside the token layout");
        }
        const auto feature_base =
            feature_block * hw::kMxmColumns
            + output.tile * hw::kLanesPerTile;
        if (feature_base + hw::kLanesPerTile > feature_count_) {
            throw std::out_of_range(
                "MXM-VXM input Buffer feature block exceeds the configured vector");
        }

        auto& bank = bank_for(group);
        for (std::size_t source_lane = 0;
             source_lane < hw::kLanesPerTile; ++source_lane) {
            const auto feature = feature_base + source_lane;
            const auto index = value_index(
                chain, operand, output.row, feature);
            if (bank.valid[index]) {
                throw std::logic_error(
                    "MXM-VXM input Buffer received the same token element twice");
            }
            bank.values[index] = MxmOutputCast::cast(
                output.values[source_lane], output.dequant_scale);
            bank.valid[index] = true;
            ++bank.fill_count;
            ++captured_values_;
        }
    }

    bool ready(std::size_t group) const
    {
        const auto found = groups_.find(group);
        return found != groups_.end()
            && found->second.fill_count == values_per_group();
    }

    // Feed one spatial VXM tile for one feature wave.  The compiler calls this
    // at start + feature + tile, matching the northward VXM instruction skew.
    void feed_feature_tile(
        std::size_t group,
        std::size_t feature,
        std::size_t tile,
        VxmSlice& vxm)
    {
        if (!ready(group)) {
            throw std::logic_error(
                "MXM-VXM input Buffer issued an incomplete token Bundle");
        }
        if (feature >= feature_count_ || tile >= hw::kTileRows) {
            throw std::out_of_range(
                "MXM-VXM input Buffer feed coordinate is outside configuration");
        }
        auto& input = vxm.input_buffer(tile);
        if (!input.empty()) {
            throw std::logic_error(
                "MXM-VXM input Buffer reached VXM tile "
                + std::to_string(tile)
                + " at feature " + std::to_string(feature)
                + " before its prior Bundle issued; state="
                + std::to_string(static_cast<int>(input.state()))
                + " fill=" + std::to_string(input.fill_count())
                + "/" + std::to_string(input.expected_count()));
        }
        input.configure(chain_count_ * kOperandsPerToken);

        const auto& bank = groups_.at(group);
        for (std::size_t chain = 0; chain < chain_count_; ++chain) {
            for (std::size_t operand = 0;
                 operand < kOperandsPerToken; ++operand) {
                auto values = VxmInputBuffer::GroupVector{};
                const auto stream_group = chain * chain_depth_ + operand;
                for (std::size_t lane = 0;
                     lane < hw::kLanesPerTile; ++lane) {
                    const auto token = tile * hw::kLanesPerTile + lane;
                    const auto bits = bank.values[value_index(
                        chain, operand, token, feature)];
                    values[lane][0] = static_cast<std::uint8_t>(bits);
                    values[lane][1] = static_cast<std::uint8_t>(bits >> 8);
                }
                input.capture_group(stream_group, values);
            }
        }
        emitted_values_ +=
            chain_count_ * kOperandsPerToken * hw::kLanesPerTile;
    }

    void release_group(std::size_t group)
    {
        if (!ready(group)) {
            throw std::logic_error(
                "MXM-VXM input Buffer cannot release an incomplete group");
        }
        groups_.erase(group);
    }

    std::uint64_t captured_values() const noexcept
    {
        return captured_values_;
    }

    std::uint64_t emitted_values() const noexcept
    {
        return emitted_values_;
    }

private:
    struct Bank {
        explicit Bank(std::size_t value_count)
            : values(value_count), valid(value_count, false)
        {
        }

        std::vector<std::uint16_t> values{};
        std::vector<bool> valid{};
        std::size_t fill_count{0};
    };

    static constexpr std::size_t token_lanes()
    {
        return hw::kTileRows * hw::kLanesPerTile;
    }

    std::size_t values_per_group() const
    {
        return chain_count_ * kOperandsPerToken
            * token_lanes() * feature_count_;
    }

    std::size_t value_index(
        std::size_t chain,
        std::size_t operand,
        std::size_t token,
        std::size_t feature) const
    {
        return (((chain * kOperandsPerToken + operand)
                    * token_lanes() + token)
                    * feature_count_ + feature);
    }

    Bank& bank_for(std::size_t group)
    {
        const auto found = groups_.find(group);
        if (found != groups_.end()) return found->second;
        if (groups_.size() >= capacity_groups_) {
            throw std::overflow_error(
                "MXM-VXM input Buffer overflow; compiler failed to spill or throttle");
        }
        return groups_.emplace(group, Bank(values_per_group()))
            .first->second;
    }

    std::size_t feature_count_{0};
    std::size_t chain_depth_{0};
    std::size_t chain_count_{0};
    std::size_t capacity_groups_{0};
    std::map<std::size_t, Bank> groups_{};
    std::uint64_t captured_values_{0};
    std::uint64_t emitted_values_{0};
};

} // namespace ftlpu
