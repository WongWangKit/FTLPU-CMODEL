#pragma once

#include "ftlpu/sxm/matrix_mem.hpp"
#include "ftlpu/vxm/lane.hpp"

namespace ftlpu {

// Explicit CModel boundary adapter, NOT additional SXM RTL storage. It collects
// one head + its mirror, each with independent LHS/RHS, and repacks low/high
// physical beats into the unchanged VxmLane wide-input API. FP32 heads sharing
// that API's bindings must be invoked separately, never silently aliased.
class SxmMatrixVxmPairAdapter {
public:
    static constexpr std::size_t T = hw::kTileRows;
    SxmMatrixVxmPairAdapter(SxmMatrixDtype dtype, std::size_t depth, std::size_t head,
                           std::size_t consumer_column = SxmMatrixMemLayout::read_column)
        : dtype_(dtype), head_(head), column_(consumer_column)
    {
        if ((depth != 2 && depth != 4 && depth != 8) || head >= 8 || head % depth ||
            (dtype != SxmMatrixDtype::BFloat16 && dtype != SxmMatrixDtype::Float32) ||
            consumer_column > SxmMatrixMemLayout::read_column)
            throw std::invalid_argument("SXM VXM pair dtype/depth/head");
    }
    std::size_t stream_base(bool mirror, bool rhs) const
    { return 2 * (head_ + (mirror ? 8 : 0)) + (rhs ? 2 : 0); }
    void capture(StreamRegisterFabric& fabric, std::size_t tile, bool mirror,
                 bool rhs, std::size_t phase)
    {
        if (!fabric.cycle_open() || tile >= T || phase >= (fp32() ? 2u : 1u))
            throw std::invalid_argument("SXM VXM capture phase/tile/cycle");
        const auto op = std::size_t(mirror) * 2 + rhs;
        auto& state = state_[op][tile];
        if (state.phases != phase || (phase && fabric.cycle() != state.low_cycle + 1))
            throw std::logic_error("SXM VXM duplicate, reordered, or delayed FP32 phase");
        const auto base = stream_base(mirror, rhs);
        if (!fabric.segment_valid(column_, tile, StreamId::West(base)) ||
            !fabric.segment_valid(column_, tile, StreamId::West(base + 1)))
            throw std::logic_error("SXM VXM row byte pair missing");
        for (std::size_t b = 0; b < 2; ++b) {
            const auto stream = StreamId::West(base + b);
            const auto bytes = fabric.segment(column_, tile, stream);
            for (std::size_t l = 0; l < 8; ++l)
                state.words[l] |= std::uint32_t(bytes[l].data) << (8 * (phase * 2 + b));
            fabric.consume_segment(column_, tile, stream, "SXM VXM phase adapter");
        }
        if (!phase) state.low_cycle = fabric.cycle();
        ++state.phases;
    }
    bool ready(std::size_t tile) const
    {
        if (tile >= T) throw std::out_of_range("SXM VXM tile");
        for (const auto& op : state_) if (op[tile].phases != (fp32() ? 2u : 1u)) return false;
        return true;
    }
    VxmLane::StreamBytes streams(std::size_t tile, std::size_t lane) const
    {
        if (!ready(tile) || lane >= 8) throw std::logic_error("SXM VXM incomplete operand bundle");
        VxmLane::StreamBytes bytes{};
        const auto kind = fp32() ? VxmLaneOperandKind::StreamFloat32 : VxmLaneOperandKind::StreamBFloat16;
        for (std::size_t op = 0; op < 4; ++op) {
            const auto base = VxmLane::fixed_input_base_for_stage(head_ + (op / 2) * 8, op % 2, kind);
            for (std::size_t b = 0; b < (fp32() ? 4u : 2u); ++b)
                bytes.at(base + b) = static_cast<std::uint8_t>(state_[op][tile].words[lane] >> (8*b));
        }
        return bytes;
    }
private:
    bool fp32() const { return dtype_ == SxmMatrixDtype::Float32; }
    struct Operand { std::array<std::uint32_t, 8> words{}; std::size_t phases{}; std::uint64_t low_cycle{}; };
    std::array<std::array<Operand, T>, 4> state_{};
    SxmMatrixDtype dtype_;
    std::size_t head_;
    std::size_t column_;
};

} // namespace ftlpu
