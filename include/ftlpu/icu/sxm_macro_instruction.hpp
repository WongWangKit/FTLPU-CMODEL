#pragma once

#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/icu/active_macro_context.hpp"

namespace ftlpu::isa {

// Queue identity supplies the opcode. Only the actual SXM FU fields are
// carried by the packed Macro template, with no launch-domain packet header.
inline constexpr std::size_t sxm_macro_native_bits(bool transpose)
{
    return transpose ? 16 : 16 + hw::kSxmTileSelectorBits
        + SxmInstruction::kTotalLanes * hw::kSxmLaneIndexBits;
}

inline void validate_sxm_macro_streams(const SxmInstruction::StreamList& streams,
    std::size_t count)
{
    if (streams.size() != count || streams.empty()
        || streams.front().stream + count > hw::kStreams)
        throw std::invalid_argument("SXM Macro has an invalid stream range");
    for (std::size_t i = 0; i < count; ++i)
        if (streams[i].stream != streams.front().stream + i)
            throw std::invalid_argument("SXM Macro streams must be consecutive");
}

template <typename Write>
void write_sxm_macro_native(const SxmInstruction& ins, bool transpose, Write write)
{
    if (ins.opcode != (transpose ? SxmOpcode::Transpose : SxmOpcode::Permute))
        throw std::invalid_argument("SXM Macro opcode does not match its queue");
    const bool all = ins.input_row == SxmInstruction::kAllInputRows;
    const bool wide = transpose && all && ins.dst_streams.size() == 2;
    validate_sxm_macro_streams(ins.src_streams,
        transpose && (!all || wide) ? 2 : 2 * hw::kLanesPerTile);
    validate_sxm_macro_streams(ins.dst_streams, wide ? 2 : 2 * hw::kLanesPerTile);
    write(ins.src_streams.front().stream, 6);
    write(ins.dst_streams.front().stream, 6);
    if (transpose) {
        if ((!all && ins.input_row != SxmInstruction::kBroadcastInputRows
                && ins.input_row >= hw::kLanesPerTile)
            || ins.output_row != SxmInstruction::kAllOutputRows
            || ins.output_tile != SxmInstruction::kAllOutputTiles)
            throw std::invalid_argument("invalid SXM transpose selectors");
        write(all ? (wide ? 15 : hw::kLanesPerTile) : ins.input_row, 4);
    } else {
        if (!all || (ins.output_row != SxmInstruction::kAllOutputRows && ins.output_row > 15)
            || (ins.output_tile != SxmInstruction::kAllOutputTiles && ins.output_tile >= hw::kTileRows))
            throw std::invalid_argument("invalid SXM permute selectors");
        write(ins.output_row == SxmInstruction::kAllOutputRows ? hw::kLanesPerTile : ins.output_row, 4);
        write(ins.output_tile == SxmInstruction::kAllOutputTiles ? hw::kTileRows : ins.output_tile,
            hw::kSxmTileSelectorBits);
        for (auto lane : ins.permute_map) {
            if (lane >= SxmInstruction::kTotalLanes)
                throw std::invalid_argument("SXM Macro permutation is outside the vector");
            write(lane, hw::kSxmLaneIndexBits);
        }
    }
}

template <typename Read>
SxmInstruction read_sxm_macro_native(bool transpose, Read read)
{
    const auto source = read(6), destination = read(6), row = read(4);
    const auto streams = [](std::size_t first, std::size_t count) {
        SxmInstruction::StreamList result;
        for (std::size_t i = 0; i < count; ++i) result.push_back(SxmStreamId{first + i});
        return result;
    };
    SxmInstruction result;
    if (transpose) {
        const bool wide = row == 15;
        result = SxmInstruction::Transpose(
            streams(source, row == hw::kLanesPerTile ? 2 * hw::kLanesPerTile : 2),
            streams(destination, wide ? 2 : 2 * hw::kLanesPerTile));
        result.input_row = wide || row == hw::kLanesPerTile ? SxmInstruction::kAllInputRows : row;
    } else {
        const auto tile = read(hw::kSxmTileSelectorBits);
        SxmInstruction::PermuteMap map{};
        for (auto& lane : map) lane = read(hw::kSxmLaneIndexBits);
        result = SxmInstruction::Permute(streams(source, 2 * hw::kLanesPerTile),
            streams(destination, 2 * hw::kLanesPerTile), map);
        result.output_row = row == hw::kLanesPerTile ? SxmInstruction::kAllOutputRows : row;
        result.output_tile = tile == hw::kTileRows ? SxmInstruction::kAllOutputTiles : tile;
    }
    write_sxm_macro_native(result, transpose, [](auto, auto) {});
    return result;
}
} // namespace ftlpu::isa

namespace ftlpu {

template <std::size_t Bits>
struct IcuPackedSxmMacroContext {
    static constexpr std::size_t bit_count = Bits, word_count = Bits / 32;
    std::array<std::uint32_t, word_count> words{};
    std::uint64_t read(std::size_t offset, std::size_t width) const
    {
        if (width > 64 || offset + width > Bits) throw std::logic_error("SXM context read overflow");
        std::uint64_t result = 0;
        for (std::size_t i = 0; i < width; ++i)
            result |= std::uint64_t((words[(offset+i)/32] >> ((offset+i)%32)) & 1u) << i;
        return result;
    }
    void write(std::size_t offset, std::size_t width, std::uint64_t value)
    {
        if (width > 64 || offset + width > Bits || (width < 64 && (value >> width)))
            throw std::logic_error("SXM context write overflow");
        for (std::size_t i = 0; i < width; ++i) {
            auto& word = words[(offset+i)/32];
            const auto mask = std::uint32_t{1} << ((offset+i)%32);
            word = (word & ~mask) | (((value >> i) & 1u) ? mask : 0);
        }
    }
};

template <bool Transpose>
class IcuSxmActiveMacroContextCodec {
    using TimingCodec = IcuActiveMacroContextCodec256<MemInstruction>;
    static constexpr std::size_t native_bits = isa::sxm_macro_native_bits(Transpose);
    static constexpr std::size_t timing_offset = 1 + native_bits;
    static constexpr std::size_t next_offset = timing_offset + 112;
public:
    // Preserve the physical limits of the replaced SXM contexts, including
    // the pre-existing target-dependent permutation-map width.
    static constexpr std::size_t bit_count = Transpose ? 256
        : ((384 + SxmInstruction::kTotalLanes * hw::kSxmLaneIndexBits - 160
            + hw::kSxmTileSelectorBits - 3 + hw::kSxmGroupStrideBits - 2 + 31) / 32) * 32;
    using Packed = IcuPackedSxmMacroContext<bit_count>;
    using State = IcuActiveMacroContextState<SxmInstruction>;
    static constexpr std::size_t word_count = Packed::word_count;
    static_assert(next_offset + 64 <= bit_count);
    static bool valid(const Packed& p) { return (p.words[0] & 1) != 0; }
    static std::uint32_t issue_cycle(const Packed& p) { return p.read(next_offset,32); }
    static std::uint32_t release_cycle(const Packed& p) { return p.read(next_offset+32,32); }

    static Packed pack_initial(const IcuMacroSchedule& schedule, const SxmInstruction& ins)
    {
        if (schedule.induction_target != IcuInductionTarget::None
            || schedule.inner_stride || schedule.outer_stride)
            throw StaticScheduleError("SXM Macro has no address induction");
        return from_timing(TimingCodec::pack_initial(schedule, MemInstruction::Read(0,0)), ins);
    }
    static Packed pack(const State& s)
    {
        IcuActiveMacroContextState<MemInstruction> t;
        copy_timing(s,t);
        return from_timing(TimingCodec::pack(t), s.instruction);
    }
    static State unpack(const Packed& p)
    {
        IcuPackedActiveMacroContext256 t;
        t.write(0,1,1);
        for (std::size_t i=0; i<112; ++i) t.write(52+i,1,p.read(timing_offset+i,1));
        t.write(192,32,p.read(next_offset,32));
        t.write(224,32,p.read(next_offset+32,32));
        State s;
        copy_timing(TimingCodec::unpack(t),s);
        std::size_t offset = 1;
        s.instruction = isa::read_sxm_macro_native(Transpose,[&](unsigned width) {
            auto value=p.read(offset,width); offset+=width; return value;
        });
        return s;
    }
    static bool advance_after_issue(State& s)
    {
        IcuActiveMacroContextState<MemInstruction> t;
        copy_timing(s,t);
        const bool done = TimingCodec::advance_after_issue(t);
        copy_timing(t,s);
        return done;
    }
private:
    template <typename A, typename B> static void copy_timing(const A& a, B& b)
    {
        b.inner_count=a.inner_count; b.inner_remaining=a.inner_remaining;
        b.outer_remaining=a.outer_remaining; b.inner_interval=a.inner_interval;
        b.outer_residual=a.outer_residual; b.next_issue_cycle=a.next_issue_cycle;
        b.final_issue_cycle=a.final_issue_cycle;
    }
    static Packed from_timing(const IcuPackedActiveMacroContext256& t, const SxmInstruction& ins)
    {
        Packed p; p.write(0,1,1);
        std::size_t offset=1;
        isa::write_sxm_macro_native(ins,Transpose,[&](std::uint64_t value,unsigned width) {
            p.write(offset,width,value); offset+=width;
        });
        for(std::size_t i=0;i<112;++i) p.write(timing_offset+i,1,t.read(52+i,1));
        p.write(next_offset,32,t.read(192,32)); p.write(next_offset+32,32,t.read(224,32));
        return p;
    }
};
} // namespace ftlpu
