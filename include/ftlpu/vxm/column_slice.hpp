#pragma once

#include "ftlpu/vxm/compact_instruction.hpp"

#include <cstddef>
#include <deque>
#include <optional>
#include <stdexcept>
#include <utility>
#include <variant>

namespace ftlpu {

// Column-local control state for one 8-lane VALU in a Superlane. Data enters
// and leaves through the explicit inter-column registers modeled by VxmLane;
// this object owns only the hardware-visible ICU/config pipeline for Cn.
class VxmColumnSlice {
public:
    static constexpr std::size_t kFifoDepth = 3;
    static constexpr std::size_t kDecodeLatency = 1;

    explicit VxmColumnSlice(std::size_t column) : column_(column)
    {
        if (column >= kVxmAluStageCount)
            throw std::out_of_range(
                "VXM column slice exceeds the configured physical VALU columns");
    }

    void reset()
    {
        queue_.clear();
        decoding_.reset();
        next_.reset();
        current_.reset();
    }

    void enqueue(VxmChainDepth depth, VxmLaneAluInstruction instruction)
    {
        if (instruction.repeat_count == 0)
            throw std::invalid_argument(
                "VXM instruction repeat_count must be non-zero");
        auto run = ConfigRun {instruction, instruction.repeat_count, depth};
        run.instruction.repeat_count = 1;
        enqueue_pending(std::move(run));
    }

    void enqueue_compact(VxmCompactInstruction instruction)
    {
        enqueue_pending(std::move(instruction));
    }

    std::optional<VxmLaneAluInstruction> issue(VxmChainDepth active_depth)
    {
        // A packet occupies the decoder for one complete control cycle.
        if (decoding_) {
            if (next_)
                throw std::logic_error(
                    "VXM column decoder completed while Next Config was occupied");
            next_ = decode_pending(std::move(*decoding_));
            decoding_.reset();
        }

        if (!current_ && next_ && next_->depth == active_depth) {
            current_ = std::move(next_);
            next_.reset();
        }

        // The column-local FIFO can prefetch while Current executes.
        if (!decoding_ && !next_ && !queue_.empty()) {
            decoding_ = std::move(queue_.front());
            queue_.pop_front();
        }

        if (!current_) return std::nullopt;
        return current_->instruction;
    }

    void consume(bool executed)
    {
        if (!executed) return;
        if (!current_ || current_->remaining == 0)
            throw std::logic_error(
                "VXM column consumed an empty Current Config Register");
        if (--current_->remaining == 0) current_.reset();
    }

    bool idle() const
    {
        return !current_ && !next_ && !decoding_ && queue_.empty();
    }

    std::size_t remaining_executions() const
    {
        std::size_t total = current_ ? current_->remaining : 0;
        if (next_) total += next_->remaining;
        if (decoding_) total += pending_remaining(*decoding_);
        for (const auto& pending : queue_)
            total += pending_remaining(pending);
        return total;
    }

    std::size_t remaining_in_current() const
    {
        return current_ ? current_->remaining : 0;
    }

    std::size_t config_entry_count() const
    {
        return queue_.size() + (decoding_ ? 1 : 0) + (next_ ? 1 : 0)
            + (current_ ? 1 : 0);
    }

    std::size_t fifo_entry_count() const { return queue_.size(); }
    bool decoding() const { return decoding_.has_value(); }
    bool next_config_ready() const { return next_.has_value(); }

    bool feedback_capture(VxmChainDepth depth) const
    {
        const ConfigRun* run = nullptr;
        if (next_ && next_->depth == depth)
            run = &*next_;
        else if (current_ && current_->depth == depth)
            run = &*current_;
        return run
            && run->instruction.lhs.kind == VxmLaneOperandKind::Feedback;
    }

    std::optional<VxmLaneAluInstruction> next_instruction() const
    {
        if (current_) return current_->instruction;
        if (next_) return next_->instruction;
        if (decoding_) return pending_instruction(*decoding_);
        if (!queue_.empty()) return pending_instruction(queue_.front());
        return std::nullopt;
    }

    bool configuration_will_issue(VxmChainDepth active_depth) const
    {
        if (current_) return true;
        if (next_) return next_->depth == active_depth;
        if (decoding_) return pending_depth(*decoding_) == active_depth;
        return false;
    }

private:
    struct ConfigRun {
        VxmLaneAluInstruction instruction {};
        std::size_t remaining {0};
        VxmChainDepth depth {VxmChainDepth::Eight};
    };

    using PendingInstruction =
        std::variant<VxmCompactInstruction, ConfigRun>;

    void enqueue_pending(PendingInstruction instruction)
    {
        if (queue_.size() >= kFifoDepth)
            throw std::overflow_error(
                "VXM column Config FIFO overflow: compiler scheduled more "
                "than 3 queued entries for one column");
        queue_.push_back(std::move(instruction));
    }

    ConfigRun decode_pending(PendingInstruction pending) const
    {
        if (auto* decoded = std::get_if<ConfigRun>(&pending))
            return std::move(*decoded);
        auto decoded = VxmCompactInstructionCodec::decode(
            column_, std::get<VxmCompactInstruction>(pending));
        auto run = ConfigRun {decoded.instruction,
            decoded.instruction.repeat_count, decoded.chain_depth};
        run.instruction.repeat_count = 1;
        return run;
    }

    std::size_t pending_remaining(const PendingInstruction& pending) const
    {
        if (const auto* decoded = std::get_if<ConfigRun>(&pending))
            return decoded->remaining;
        return VxmCompactInstructionCodec::decode(
            column_, std::get<VxmCompactInstruction>(pending))
            .instruction.repeat_count;
    }

    VxmChainDepth pending_depth(const PendingInstruction& pending) const
    {
        if (const auto* decoded = std::get_if<ConfigRun>(&pending))
            return decoded->depth;
        return VxmCompactInstructionCodec::decode(
            column_, std::get<VxmCompactInstruction>(pending)).chain_depth;
    }

    VxmLaneAluInstruction pending_instruction(
        const PendingInstruction& pending) const
    {
        if (const auto* decoded = std::get_if<ConfigRun>(&pending))
            return decoded->instruction;
        return VxmCompactInstructionCodec::decode(
            column_, std::get<VxmCompactInstruction>(pending)).instruction;
    }

    std::size_t column_ {0};
    std::deque<PendingInstruction> queue_ {};
    std::optional<PendingInstruction> decoding_ {};
    std::optional<ConfigRun> next_ {};
    std::optional<ConfigRun> current_ {};
};

} // namespace ftlpu
