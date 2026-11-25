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
    bool idle() const { return pipe_->idle(); }
    bool can_issue(const SxmInstruction& op) const { return pipe_->can_issue(op,cycle_); }
    void issue(const SxmInstruction& op) { pipe_->issue(op,cycle_); }
    void reset() { pipe_ = std::make_unique<SxmFifoTranspose>(); cycle_=0; }
    void evaluate(StreamRegisterFabric& fabric) {
        ports_.validate_for(fabric);
        if (!fabric.cycle_open()) throw std::logic_error("SXM requires an open SR cycle");
        pipe_->clock(fabric, ports_, cycle_);
        ++cycle_;
    }
    void set_trace_enabled(bool) {}
    void log_cycle(std::ostream& os) const {
        os << "sxm FIFO cycle " << cycle_ << (idle() ? " idle" : " active") << '\n';
    }
private:
    SxmStreamPortMap ports_;
    std::unique_ptr<SxmFifoTranspose> pipe_;
    std::size_t cycle_{};
};
} // namespace ftlpu
