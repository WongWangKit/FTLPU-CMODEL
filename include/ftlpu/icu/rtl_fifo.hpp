#pragma once

#include "ftlpu/core/timing.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ftlpu {

struct RtlFifoStatistics {
    std::uint64_t samples{0};
    std::uint64_t occupancy_sum{0};
    std::size_t maximum_occupancy{0};
    std::uint64_t full_cycles{0};
    std::uint64_t empty_cycles{0};
    std::uint64_t accepted_writes{0};
    std::uint64_t rejected_writes{0};
    std::uint64_t accepted_reads{0};
    std::uint64_t stall_cycles{0};
    std::uint64_t current_backpressure_duration{0};
    std::uint64_t maximum_backpressure_duration{0};

    double average_occupancy() const noexcept
    {
        return samples == 0 ? 0.0
            : static_cast<double>(occupancy_sum)
                / static_cast<double>(samples);
    }
};

// Cycle-accurate model of rtl/queue/lpu_queue_fifo.sv. The output is a
// combinational view of Current storage. Enqueued data first becomes visible
// after POSEDGE; an empty FIFO therefore has no write-to-read bypass.
//
// full_replacement=true matches lpu_queue_fifo: a full FIFO accepts a write
// when its Current head is also accepted in the same cycle. false matches the
// ProgramLoad relay buffers, whose ready only depends on registered occupancy.
template<class T>
class RtlSynchronousFifo final : public timing::ClockedModule {
public:
    explicit RtlSynchronousFifo(
        std::size_t depth, bool full_replacement = true)
        : depth_(depth)
        , full_replacement_(full_replacement)
        , current_(make_state(depth))
    {
        if (depth_ == 0)
            throw std::invalid_argument("RTL FIFO depth must be non-zero");
    }

    RtlSynchronousFifo(const RtlSynchronousFifo&) = delete;
    RtlSynchronousFifo& operator=(const RtlSynchronousFifo&) = delete;

    std::size_t depth() const noexcept { return depth_; }
    std::size_t level_o() const noexcept { return current_.level; }
    bool empty() const noexcept { return current_.level == 0; }
    bool full() const noexcept { return current_.level == depth_; }
    bool dequeue_valid_o() const noexcept { return !empty(); }
    bool dequeue_fire() const noexcept
    {
        return dequeue_valid_o() && dequeue_ready_i_;
    }
    bool enqueue_ready_o() const noexcept
    {
        return current_.level < depth_
            || (full_replacement_ && dequeue_fire());
    }
    bool enqueue_fire() const noexcept
    {
        return enqueue_valid_i_ && enqueue_ready_o();
    }

    const T& dequeue_data_o() const
    {
        if (empty()) throw std::logic_error("RTL FIFO front on empty FIFO");
        const auto& entry = current_.storage[current_.read_pointer];
        if (!entry.has_value())
            throw std::logic_error("RTL FIFO occupied slot has no data");
        return *entry;
    }

    // Drive one cycle of combinational inputs. A valid enqueue owns its data
    // until LOW samples it. Calling drive twice before POSEDGE is an error.
    void drive(bool enqueue_valid_i, std::optional<T> enqueue_data_i,
        bool dequeue_ready_i)
    {
        if (inputs_driven_ || next_.has_value())
            throw std::logic_error("RTL FIFO inputs driven outside input window");
        if (enqueue_valid_i != enqueue_data_i.has_value())
            throw std::invalid_argument("RTL FIFO valid/data mismatch");
        enqueue_valid_i_ = enqueue_valid_i;
        enqueue_data_i_ = std::move(enqueue_data_i);
        dequeue_ready_i_ = dequeue_ready_i;
        inputs_driven_ = true;
    }

    void drive_idle() { drive(false, std::nullopt, false); }

    void low() override
    {
        if (next_.has_value())
            throw std::logic_error("RTL FIFO LOW executed more than once");
        if (!inputs_driven_) drive_idle();

        next_ = current_;
        auto& next = *next_;
        const auto write_fire = enqueue_fire();
        const auto read_fire = dequeue_fire();

        ++next.statistics.samples;
        next.statistics.occupancy_sum += current_.level;
        next.statistics.maximum_occupancy = std::max(
            next.statistics.maximum_occupancy, current_.level);
        if (empty()) ++next.statistics.empty_cycles;
        if (full()) ++next.statistics.full_cycles;
        if (enqueue_valid_i_ && !enqueue_ready_o()) {
            ++next.statistics.rejected_writes;
            ++next.statistics.current_backpressure_duration;
            next.statistics.maximum_backpressure_duration = std::max(
                next.statistics.maximum_backpressure_duration,
                next.statistics.current_backpressure_duration);
        } else {
            next.statistics.current_backpressure_duration = 0;
        }
        if (dequeue_valid_o() && !dequeue_ready_i_)
            ++next.statistics.stall_cycles;

        if (write_fire) {
            next.storage[current_.write_pointer] = *enqueue_data_i_;
            next.write_pointer = increment(current_.write_pointer);
            ++next.statistics.accepted_writes;
        }
        if (read_fire) {
            next.read_pointer = increment(current_.read_pointer);
            ++next.statistics.accepted_reads;
        }
        if (write_fire && !read_fire) ++next.level;
        if (!write_fire && read_fire) --next.level;
        next.statistics.maximum_occupancy = std::max(
            next.statistics.maximum_occupancy, next.level);
    }

    void validate_posedge() const override
    {
        if (!next_.has_value())
            throw std::logic_error("RTL FIFO POSEDGE without LOW");
        if (next_->level > depth_)
            throw std::logic_error("RTL FIFO Next occupancy exceeds depth");
    }

    void posedge() override
    {
        validate_posedge();
        current_ = std::move(*next_);
        next_.reset();
        enqueue_valid_i_ = false;
        enqueue_data_i_.reset();
        dequeue_ready_i_ = false;
        inputs_driven_ = false;
    }

    void high() override {}

    void reset()
    {
        current_ = make_state(depth_);
        next_.reset();
        enqueue_valid_i_ = false;
        enqueue_data_i_.reset();
        dequeue_ready_i_ = false;
        inputs_driven_ = false;
    }

    const RtlFifoStatistics& statistics() const noexcept
    {
        return current_.statistics;
    }

    std::size_t read_pointer() const noexcept
    {
        return current_.read_pointer;
    }
    std::size_t write_pointer() const noexcept
    {
        return current_.write_pointer;
    }

private:
    struct State {
        std::vector<std::optional<T>> storage{};
        std::size_t read_pointer{0};
        std::size_t write_pointer{0};
        std::size_t level{0};
        RtlFifoStatistics statistics{};
    };

    static State make_state(std::size_t depth)
    {
        State state;
        state.storage.resize(depth);
        return state;
    }

    std::size_t increment(std::size_t pointer) const noexcept
    {
        return pointer + 1 == depth_ ? 0 : pointer + 1;
    }

    std::size_t depth_;
    bool full_replacement_;
    State current_;
    std::optional<State> next_{};
    bool enqueue_valid_i_{false};
    std::optional<T> enqueue_data_i_{};
    bool dequeue_ready_i_{false};
    bool inputs_driven_{false};
};

} // namespace ftlpu
