#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/vxm_distributed/lane.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ftlpu::distributed_vxm {

// One controller serves a VXM tile. The data plane is physically banked by
// Lane: every Lane owns one Stream-byte Buffer, while stream-group capture and
// Bundle issue are controlled atomically for all 16 Lanes. Stream groups may
// arrive in different SR cycles; the controller places each complete FP16
// group at its fixed position and exposes nothing to the ALUs until the
// configured number of groups has arrived.
class VxmInputBuffer {
public:
    static constexpr std::size_t kLaneCount = hw::kLanesPerTile;
    static constexpr std::size_t kGroupBytes =
        VxmLane::kStreamGroupBytes;
    static constexpr std::size_t kGroupCount =
        VxmLane::kStreamGroupCount;

    using LaneBuffer = VxmLane::StreamBytes;
    using StreamMatrix = std::array<LaneBuffer, kLaneCount>;
    using GroupBytes = std::array<std::uint8_t, kGroupBytes>;
    using GroupVector = std::array<GroupBytes, kLaneCount>;

    enum class State {
        Empty,
        Collecting,
        Ready,
    };

    void reset()
    {
        lane_buffers_ = {};
        occupied_.fill(false);
        expected_count_ = 0;
        fill_count_ = 0;
        state_ = State::Empty;
    }

    void configure(std::size_t expected_count)
    {
        if (state_ != State::Empty) {
            throw std::logic_error(
                "VXM input Buffer cannot be reconfigured while a Bundle is active");
        }
        if (expected_count == 0 || expected_count > kGroupCount) {
            throw std::out_of_range(
                "VXM input Buffer expected count is outside 1..16");
        }
        expected_count_ = expected_count;
        state_ = State::Collecting;
    }

    void capture_group(
        std::size_t group,
        const GroupVector& values)
    {
        check_group(group);
        if (state_ != State::Collecting) {
            throw std::logic_error(
                "VXM input Buffer capture requires a collecting Bundle");
        }
        if (occupied_[group]) {
            throw std::logic_error(
                "VXM input Buffer received the same stream group twice");
        }

        const auto base = group * kGroupBytes;
        for (std::size_t lane = 0; lane < kLaneCount; ++lane) {
            for (std::size_t byte = 0; byte < kGroupBytes; ++byte) {
                lane_buffers_[lane][base + byte] = values[lane][byte];
            }
        }
        occupied_[group] = true;
        ++fill_count_;
        if (fill_count_ == expected_count_) {
            state_ = State::Ready;
        }
    }

    // Compatibility/host injection still traverses the Buffer logically. The
    // caller supplies one already-complete physical Stream Matrix.
    void load_complete(const StreamMatrix& streams)
    {
        if (state_ != State::Empty) {
            throw std::logic_error(
                "VXM input Buffer already owns an active Bundle");
        }
        lane_buffers_ = streams;
        occupied_.fill(true);
        expected_count_ = kGroupCount;
        fill_count_ = kGroupCount;
        state_ = State::Ready;
    }

    const StreamMatrix& bundle() const
    {
        if (!ready()) {
            throw std::logic_error(
                "VXM input Buffer Bundle is not ready");
        }
        return lane_buffers_;
    }

    // Static scheduling has no ALU ready/valid acknowledgement. Once the
    // compiler-scheduled issue succeeds, the whole Bundle is released at the
    // cycle edge. A failed issue is a schedule error, never a retained retry.
    void release_after_issue()
    {
        if (!ready()) {
            throw std::logic_error(
                "VXM input Buffer cannot consume an incomplete Bundle");
        }
        reset();
    }

    State state() const noexcept
    {
        return state_;
    }

    bool empty() const noexcept
    {
        return state_ == State::Empty;
    }

    bool collecting() const noexcept
    {
        return state_ == State::Collecting;
    }

    bool ready() const noexcept
    {
        return state_ == State::Ready;
    }

    bool has_group(std::size_t group) const
    {
        check_group(group);
        return occupied_[group];
    }

    std::size_t expected_count() const noexcept
    {
        return expected_count_;
    }

    std::size_t fill_count() const noexcept
    {
        return fill_count_;
    }

private:
    static void check_group(std::size_t group)
    {
        if (group >= kGroupCount) {
            throw std::out_of_range(
                "VXM input Buffer stream group is outside 0..15");
        }
    }

    StreamMatrix lane_buffers_{};
    std::array<bool, kGroupCount> occupied_{};
    std::size_t expected_count_{0};
    std::size_t fill_count_{0};
    State state_{State::Empty};
};

} // namespace ftlpu::distributed_vxm
