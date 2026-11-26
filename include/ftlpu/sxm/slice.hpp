#pragma once
#include "ftlpu/sxm/fifo_transpose.hpp"
#include "ftlpu/sxm/transpose_architecture.hpp"
#include <ostream>
#include <memory>
namespace ftlpu {
class SxmStreamPortMap {
public:
    static SxmStreamPortMap BetweenColumns(
        std::size_t east_input,
        std::size_t east_output,
        std::size_t west_input,
        std::size_t west_output)
    {
        return SxmStreamPortMap(east_input, east_output, west_input, west_output);
    }

    static SxmStreamPortMap SameDirection(std::size_t input, std::size_t output)
    {
        return SxmStreamPortMap(input, output, input, output);
    }

    std::size_t input_column(StreamDirection direction) const noexcept
    {
        return direction == StreamDirection::East ? east_input_ : west_input_;
    }

    std::size_t output_column(StreamDirection direction) const noexcept
    {
        return direction == StreamDirection::East ? east_output_ : west_output_;
    }

    void validate_for(const StreamRegisterFabric& fabric) const
    {
        if (east_input_ >= fabric.column_count()
            || east_output_ >= fabric.column_count()
            || west_input_ >= fabric.column_count()
            || west_output_ >= fabric.column_count()) {
            throw std::out_of_range("SXM port maps outside stream-register fabric");
        }
    }

private:
    SxmStreamPortMap(
        std::size_t east_input,
        std::size_t east_output,
        std::size_t west_input,
        std::size_t west_output)
        : east_input_(east_input)
        , east_output_(east_output)
        , west_input_(west_input)
        , west_output_(west_output)
    {
    }

    std::size_t east_input_{0};
    std::size_t east_output_{0};
    std::size_t west_input_{0};
    std::size_t west_output_{0};
};

class SxmSlice {
public:
    explicit SxmSlice(SxmStreamPortMap ports, SxmTransposeArchitecture = SxmTransposeArchitecture::Fifo)
        : ports_(std::move(ports)), pipe_(std::make_unique<SxmFifoTranspose>()) {}
    const SxmStreamPortMap& ports() const { return ports_; }
    std::size_t cycle() const { return cycle_; }
    bool idle() const { return pipe_->idle() && local_jobs_.empty() && rotate_jobs_.empty(); }
    bool can_issue(const SxmInstruction& op) const {
        if (!op.is_stream_operation())
            return local_jobs_.empty() && rotate_jobs_.empty() && pipe_->can_issue(op, cycle_);
        if (!pipe_->idle()) return false;
        if (op.opcode == SxmOpcode::VectorRotate)
            return rotate_jobs_.empty() || rotate_jobs_.back().start != cycle_;
        return local_jobs_.empty() || cycle_ - local_jobs_.back().start >= hw::kLanesPerTile;
    }
    void issue(const SxmInstruction& op) {
        if (op.is_stream_operation()) {
            op.validate_stream_operation();
            if (!can_issue(op)) throw std::logic_error("SXM stream issue conflict");
            if (op.opcode == SxmOpcode::VectorRotate)
                rotate_jobs_.push_back({cycle_, op, {}});
            else
                local_jobs_.push_back({cycle_, op, {}});
        } else {
            if (!local_jobs_.empty() || !rotate_jobs_.empty())
                throw std::logic_error("SXM stream pipeline still active");
            pipe_->issue(op, cycle_);
        }
    }
    void reset() {
        pipe_ = std::make_unique<SxmFifoTranspose>();
        local_jobs_.clear();
        rotate_jobs_.clear();
        cycle_ = 0;
    }
    void evaluate(StreamRegisterFabric& fabric) {
        ports_.validate_for(fabric);
        if (!fabric.cycle_open()) throw std::logic_error("SXM requires an open SR cycle");
        evaluate_stream_operations(fabric);
        pipe_->clock(fabric, ports_, cycle_);
        ++cycle_;
    }
    void set_trace_enabled(bool) {}
    void log_cycle(std::ostream& os) const {
        os << "sxm FIFO cycle " << cycle_ << (idle() ? " idle" : " active") << '\n';
    }
private:
    using Planes = std::array<StreamTileSegment, 2>;
    struct RotateJob {
        std::size_t start;
        SxmInstruction op;
        std::array<Planes, hw::kTileRows> segments;
    };
    struct LocalTransposeJob {
        std::size_t start;
        SxmInstruction op;
        std::array<std::array<Planes, hw::kLanesPerTile>, hw::kTileRows> rows;
    };
    std::deque<RotateJob> rotate_jobs_;
    std::deque<LocalTransposeJob> local_jobs_;

    void capture_planes(StreamRegisterFabric& fabric, const SxmInstruction& op,
        std::size_t sl, Planes& data) const
    {
        for (auto port : op.src_streams) {
            const auto id = StreamId::from_packed(port.stream);
            if (!fabric.segment_valid(ports_.input_column(id.direction()), sl, id))
                throw std::logic_error("SXM missing scheduled input at SL" + std::to_string(sl)
                    + " on SR " + std::to_string(port.stream) + " cycle " + std::to_string(cycle()));
        }
        for (std::size_t p = 0; p < op.src_streams.size(); ++p) {
            const auto id = StreamId::from_packed(op.src_streams[p].stream);
            const auto column = ports_.input_column(id.direction());
            data[p] = fabric.segment(column, sl, id);
            fabric.consume_segment(column, sl, id, "SXM stream capture");
        }
    }

    void emit_planes(StreamRegisterFabric& fabric, const SxmInstruction& op,
        std::size_t sl, const Planes& data) const
    {
        for (std::size_t p = 0; p < op.dst_streams.size(); ++p) {
            const auto id = StreamId::from_packed(op.dst_streams[p].stream);
            fabric.stage_segment(ports_.output_column(id.direction()), sl, id, data[p], "SXM stream result");
        }
    }

    void evaluate_stream_operations(StreamRegisterFabric& fabric)
    {
        constexpr auto lanes = hw::kLanesPerTile;
        constexpr auto superlanes = hw::kTileRows;
        for (auto& job : local_jobs_) {
            const auto age = cycle() - job.start;
            for (std::size_t sl = 0; sl < superlanes && sl <= age; ++sl) {
                const auto beat = age - sl;
                if (beat < lanes) {
                    capture_planes(fabric, job.op, sl, job.rows[sl][beat]);
                } else if (beat < 2 * lanes) {
                    Planes result{};
                    for (std::size_t p = 0; p < job.op.dst_streams.size(); ++p)
                        for (std::size_t lane = 0; lane < lanes; ++lane)
                            result[p][lane] = job.rows[sl][lane][p][beat-lanes];
                    emit_planes(fabric, job.op, sl, result);
                }
            }
        }
        if (!local_jobs_.empty() && cycle()-local_jobs_.front().start == superlanes+2*lanes-2)
            local_jobs_.pop_front();

        // SL s samples at issue+s. The switch registers the complete vector one
        // cycle after the last input. Destination d then delays its output by d.
        for (auto& job : rotate_jobs_) {
            const auto age = cycle() - job.start;
            if (age < superlanes) {
                capture_planes(fabric, job.op, age, job.segments[age]);
            } else {
                const auto dest = age - superlanes;
                const auto src = (dest + superlanes - job.op.shift_distance) % superlanes;
                emit_planes(fabric, job.op, dest, job.segments[src]);
            }
        }
        if (!rotate_jobs_.empty() && cycle()-rotate_jobs_.front().start == 2*superlanes-1)
            rotate_jobs_.pop_front();
    }

    SxmStreamPortMap ports_;
    std::unique_ptr<SxmFifoTranspose> pipe_;
    std::size_t cycle_{};
};
} // namespace ftlpu
