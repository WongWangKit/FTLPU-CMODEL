#pragma once
#include "ftlpu/core/stream_port.hpp"
#include "ftlpu/sxm/fifo_timing.hpp"
#include "ftlpu/sxm/instruction.hpp"
#include <array>
#include <deque>
#include <optional>
#include <vector>

namespace ftlpu {
// A packet carries the two parallel byte planes of one superlane segment.
class SxmFifoTranspose {
    static constexpr int S = sxm_fifo::superlanes, K = sxm_fifo::lanes;
    static_assert(S > 0 && K > 0);
    using Pair = std::array<StreamTileSegment, 2>;
    struct Packet {
        Pair data;
        int target, position;
        StreamId dest;
    };
    using Beat = std::optional<Packet>;
    using Wave = std::array<Beat, S>;
    // Depth is fixed at construction. One push/pop per clock, including bubbles.
    struct Delay {
        std::vector<Beat> slots;
        std::size_t head{};
        Beat clock(Beat in) {
            if (slots.empty()) return in;
            std::swap(in, slots[head]);
            head = (head+1) % slots.size();
            return in;
        }
    };
    struct Command {
        std::size_t start;
        StreamId src, dst;
        int first_bank;
    };
    using Bank = std::array<Pair, K>;
    std::array<std::array<Bank, 2>, S> banks_;
    std::array<Delay, S> before_, after_;
    std::array<Wave, sxm_fifo::exchange_stages> stages_{};
    std::deque<Command> commands_;
    int next_bank_{};

    // For an ingress wave emitted at t, source s and target d obey
    // t = issue + K + skew + K*(s+d) + local_column. Two overlapping
    // matrices have sums differing by at least S because issue II >= S*K.
    // Their active intervals are disjoint, even with arbitrary issue gaps.
    Wave interval_exchange(Wave input) {
        Wave output{};
        output.swap(stages_.back());
        for (int stage = sxm_fifo::exchange_stages - 1; stage >= 0; --stage) {
            auto& prev = stage == 0 ? input : stages_[stage - 1];
            auto& next = stages_[stage];
            for (int port = 0; port < S; ++port) {
                if (!prev[port]) continue;
                const auto& packet = *prev[port];
                // position holds the ingress SL on this topology. The stage
                // array index is the physical SL occupied in the current tick.
                const int dest = sxm_fifo::interval_exchange_port(
                    stage, port, packet.position + packet.target, S);
                if (next[dest]) throw std::logic_error("SXM interval exchange collision");
                if (stage + 1 == sxm_fifo::exchange_stages && dest != packet.target)
                    throw std::logic_error("SXM interval exchange target mismatch");
                next[dest].swap(prev[port]);
            }
        }
        return output;
    }

    // Set the target bits from high to low. A distance-2 switch consists of
    // two adjacent registered hops. During a multi-hop switch, each original
    // port has its own wire/register, including both central crossing paths.
    Wave exchange(Wave input) {
        if constexpr (!sxm_fifo::binary_exchange) return interval_exchange(std::move(input));
        int stage = int(stages_.size())-1;
        Wave output;
        if constexpr (S == 1) return input;
        output.swap(stages_.back());
        for (int distance=1; distance<S; distance*=2) {
            for (int hop=distance-1; hop>=0; --hop, --stage) {
                auto& prev = stage ? stages_[stage-1] : input;
                auto& next = stages_[stage];
                for (int origin=0; origin<S; ++origin) {
                    if (!prev[origin]) continue;
                    auto packet = std::move(*prev[origin]);
                    prev[origin].reset();
                    if ((origin ^ packet.target) & distance)
                        packet.position += (origin & distance) ? -1 : 1;
                    const auto port = hop+1==distance ? packet.position : origin;
                    if (next[port]) throw std::logic_error("SXM FIFO exchange register collision");
                    next[port].emplace(std::move(packet));
                }
            }
        }
        return output;
    }
public:
    SxmFifoTranspose() {
        for (int sl=0; sl<S; ++sl) {
            before_[sl].slots.resize(sxm_fifo::input_delay(sl));
            after_[sl].slots.resize(sxm_fifo::output_delay(sl));
        }
    }
    static bool accepts(const SxmInstruction& op) {
        if (op.opcode != SxmOpcode::Transpose || op.input_row != SxmInstruction::kAllInputRows ||
            op.output_row != SxmInstruction::kAllOutputRows || op.output_tile != SxmInstruction::kAllOutputTiles)
            return false;
        for (auto list : {&op.src_streams, &op.dst_streams}) {
            if (list->size()!=2) return false;
            auto lo=(*list)[0].stream, hi=(*list)[1].stream;
            if (hi>=hw::kStreams || lo>=hw::kStreams || hi!=lo+1 ||
                StreamId::from_packed(lo).direction()!=StreamId::from_packed(hi).direction()) return false;
        }
        return true;
    }
    bool idle() const { return commands_.empty(); }
    bool can_issue(const SxmInstruction& op, std::size_t t) const {
        return accepts(op) && (idle() || t >= commands_.back().start + sxm_fifo::width);
    }
    void issue(const SxmInstruction& op, std::size_t t) {
        if (!accepts(op)) throw std::invalid_argument("SXM supports only a complete FIFO transpose on two byte streams");
        if (!can_issue(op,t)) throw std::logic_error("SXM FIFO transpose input busy");
        commands_.push_back({t,StreamId::from_packed(op.src_streams[0].stream),
                               StreamId::from_packed(op.dst_streams[0].stream), next_bank_});
        next_bank_ = (next_bank_ + S) % 2;
    }
    template<class Ports>
    void clock(StreamRegisterFabric& fabric, const Ports& ports, std::size_t t) {
        if (idle()) return;
        Wave input{};
        for (const auto& cmd : commands_) {
            for (int sl=0; sl<S; ++sl) {
                if (t<cmd.start+sl) continue;
                const auto row=t-cmd.start-sl;
                if (row<sxm_fifo::width) {
                    StreamInputPort port(fabric,ports.input_column(cmd.src.direction()),cmd.src.direction(),"FIFO transpose");
                    for (int p=0;p<2;++p) if (!port.segment_valid(sl,cmd.src.index()+p))
                        throw std::logic_error("SXM FIFO transpose scheduled input missing");
                    auto& bank=banks_[sl][(cmd.first_bank + row/K)%2];
                    for (int p=0;p<2;++p) bank[row%K][p]=port.consume_segment(sl,cmd.src.index()+p);
                }
                if (row>=K && row<sxm_fifo::width+K) {
                    const int local=int(row)-K, group=local/K;
                    const auto& bank=banks_[sl][(cmd.first_bank + group)%2];
                    Packet packet{{},group,sl,cmd.dst};
                    for (int p=0;p<2;++p) for (int i=0;i<K;++i)
                        packet.data[p][i]=bank[i][p][local%K];
                    input[sl].emplace(std::move(packet));
                }
            }
        }
        for (int sl=0;sl<S;++sl) input[sl]=before_[sl].clock(std::move(input[sl]));
        auto routed=exchange(std::move(input));
        for (int sl=0;sl<S;++sl) {
            auto out=after_[sl].clock(std::move(routed[sl]));
            if (!out) continue;
            StreamOutputPort port(fabric,ports.output_column(out->dest.direction()),out->dest.direction(),"FIFO transpose");
            for (int p=0;p<2;++p) port.write_segment(sl,out->dest.index()+p,out->data[p]);
        }
        if (t+1==commands_.front().start+sxm_fifo::execution_cycles) commands_.pop_front();
    }
};
}
