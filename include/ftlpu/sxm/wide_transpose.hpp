#pragma once
#include "ftlpu/core/stream_port.hpp"
#include "ftlpu/sxm/instruction.hpp"
#include <array>
#include <deque>
#include <optional>
#include <stdexcept>

namespace ftlpu {
// One command consumes N logical rows on a byte-stream pair. SL s arrives s
// cycles after SL0. Align -> 2 SRAM tiles -> re-skew, with no Permute command.
class SxmWideTranspose {
public:
    static constexpr std::size_t N = hw::kTileRows * hw::kLanesPerTile;
    static constexpr std::size_t skew = hw::kTileRows - 1;
    // Half-open spans from the actual issue cycle, including SL skew.
    static constexpr std::size_t input_cycles = N + skew;
    static constexpr std::size_t output_offset = N + skew;
    static constexpr std::size_t execution_cycles = 2 * (N + skew);
    static constexpr std::size_t matrix_bytes = 2 * N * N * 2;
    static constexpr std::size_t fifo_bytes = hw::kLanesPerTile * hw::kTileRows * skew;
    static constexpr std::size_t payload_capacity_bytes = matrix_bytes + 2 * fifo_bytes;
    static bool accepts(const SxmInstruction& i) {
        if (i.opcode != SxmOpcode::Transpose || i.src_streams.size()!=2 || i.dst_streams.size()!=2
            || i.input_row!=SxmInstruction::kAllInputRows
            || i.output_row!=SxmInstruction::kAllOutputRows
            || i.output_tile!=SxmInstruction::kAllOutputTiles) return false;
        for (auto* streams : {&i.src_streams,&i.dst_streams}) {
            if ((*streams)[0].stream>=hw::kStreams || (*streams)[1].stream>=hw::kStreams
                || (*streams)[0]==(*streams)[1]) return false;
            if (StreamId::from_packed((*streams)[0].stream).direction()
                != StreamId::from_packed((*streams)[1].stream).direction()) return false;
        }
        return true;
    }
    bool can_issue(std::size_t cycle) const { return !last_issue_ || cycle>=*last_issue_+N; }
    bool can_issue(const SxmInstruction& i,std::size_t cycle) const {
        return accepts(i) && can_issue(cycle) && (jobs_.empty() ||
            (StreamId::from_packed(i.src_streams[0].stream).direction()==StreamId::from_packed(jobs_.front().i.src_streams[0].stream).direction()
            && StreamId::from_packed(i.dst_streams[0].stream).direction()==StreamId::from_packed(jobs_.front().i.dst_streams[0].stream).direction()));
    }
    void issue(const SxmInstruction& i,std::size_t cycle) {
        if (!accepts(i)) throw std::invalid_argument("wide SXM needs distinct input/output byte-stream pairs and a complete tile");
        if (!can_issue(cycle)) throw std::logic_error("wide SXM input initiation interval not met");
        if (!jobs_.empty() &&
            (StreamId::from_packed(i.src_streams[0].stream).direction()!=StreamId::from_packed(jobs_.front().i.src_streams[0].stream).direction()
            || StreamId::from_packed(i.dst_streams[0].stream).direction()!=StreamId::from_packed(jobs_.front().i.dst_streams[0].stream).direction()))
            throw std::logic_error("wide SXM direction change requires drain");
        jobs_.push_back({i,cycle,next_bank_}); next_bank_^=1; last_issue_=cycle;
    }
    bool idle() const { return jobs_.empty()&&input_.empty()&&output_.empty(); }
    void reset() { *this=SxmWideTranspose{}; }
    std::size_t input_buffer_bytes() const { return input_.size()*2*hw::kLanesPerTile; }
    std::size_t output_buffer_bytes() const { return output_.size()*2*hw::kLanesPerTile; }
    void evaluate(StreamRegisterFabric& fabric,std::size_t cycle,
                  std::size_t input_column,std::size_t output_column) {
        // Validate all this cycle's source segments before consuming any input.
        for (const auto& j:jobs_) for(std::size_t s=0;s<hw::kTileRows;++s) {
            if(cycle<j.start+s || cycle>=j.start+s+N) continue;
            for(auto src:j.i.src_streams) {
                auto id=StreamId::from_packed(src.stream);
                StreamInputPort p(fabric,input_column,id.direction(),"wide SXM");
                if(!p.segment_valid(s,id.index())) throw std::logic_error("wide SXM scheduled input is missing");
            }
        }
        for (const auto& j:jobs_) for(std::size_t s=0;s<hw::kTileRows;++s) {
            if(cycle<j.start+s || cycle>=j.start+s+N) continue;
            Beat b{}; b.due=cycle+skew-s; b.sl=s; b.row=cycle-j.start-s; b.bank=j.bank;
            for(std::size_t p=0;p<2;++p) {
                auto id=StreamId::from_packed(j.i.src_streams[p].stream);
                b.data[p]=fabric.segment(input_column,s,id);
                fabric.consume_segment(input_column,s,id,"wide SXM input");
            }
            input_.push_back(b);
        }
        for(auto it=input_.begin();it!=input_.end();) {
            if(it->due!=cycle){++it;continue;}
            for(std::size_t p=0;p<2;++p)for(std::size_t l=0;l<hw::kLanesPerTile;++l) {
                auto c=it->sl*hw::kLanesPerTile+l;
                // 32 banks: bank=(row+column)%N, address=row.
                auto& cell=memory_[it->bank][p][(it->row+c)%N][it->row];
                if(cell.has_value()) throw std::logic_error("wide SXM live SRAM overwrite");
                cell=it->data[p][l];
            }
            it=input_.erase(it);
        }
        for(const auto& j:jobs_) {
            const auto first=j.start+output_offset;
            if(cycle<first || cycle>=first+N)continue;
            const auto col=cycle-first;
            for(std::size_t s=0;s<hw::kTileRows;++s) {
                Beat b{};b.due=cycle+s;b.sl=s;
                for(std::size_t p=0;p<2;++p){
                    b.dst[p]=j.i.dst_streams[p];
                    for(std::size_t l=0;l<hw::kLanesPerTile;++l){
                        const auto r=s*hw::kLanesPerTile+l;
                        auto& cell=memory_[j.bank][p][(r+col)%N][r];
                        if(!cell)throw std::logic_error("wide SXM reads uninitialized SRAM");
                        b.data[p][l]=*cell;cell.reset();
                    }
                }
                output_.push_back(b);
            }
        }
        for(auto it=output_.begin();it!=output_.end();) {
            if(it->due!=cycle){++it;continue;}
            for(std::size_t p=0;p<2;++p){
                auto id=StreamId::from_packed(it->dst[p].stream);
                StreamOutputPort port(fabric,output_column,id.direction(),"wide SXM output");
                port.write_segment(it->sl,id.index(),it->data[p]);
            }
            it=output_.erase(it);
        }
        while(!jobs_.empty()&&cycle>=jobs_.front().start+execution_cycles-1)jobs_.pop_front();
        if(input_buffer_bytes()>fifo_bytes||output_buffer_bytes()>fifo_bytes)
            throw std::logic_error("wide SXM alignment FIFO capacity exceeded");
    }
    std::optional<SxmInstruction> active_instruction() const {
        if(jobs_.empty())return std::nullopt;return jobs_.front().i;
    }
private:
    struct Job { SxmInstruction i; std::size_t start,bank; };
    struct Beat { std::size_t due{},sl{},row{},bank{};std::array<StreamTileSegment,2> data{};std::array<SxmStreamId,2> dst{}; };
    using Matrix=std::array<std::array<std::optional<StreamCell>,N>,N>;
    std::array<std::array<Matrix,2>,2> memory_{};
    std::deque<Job> jobs_;
    std::deque<Beat> input_,output_;
    std::optional<std::size_t> last_issue_;
    std::size_t next_bank_{};
};
}
