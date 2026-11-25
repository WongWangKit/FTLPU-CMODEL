#pragma once

#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/icu/active_macro_context.hpp"
#include "ftlpu/sxm/fifo_transpose.hpp"

namespace ftlpu::isa {

// Reserve the old full-tile selector (15 on LPU32) so previously timed
// binaries fail explicitly instead of consuming a different output wave.
inline constexpr unsigned kSxmFifoTransposeRow = SxmEncodingLayout::kExtended
    ? 3 * SxmEncodingLayout::kRowModeUnit - 1 : 14;
inline constexpr std::size_t sxm_macro_native_bits(bool transpose) {
    // The old queue geometry remains readable in binary headers.
    return 12 + SxmEncodingLayout::kRowBits + (transpose ? 0 : hw::kSxmTileSelectorBits
        + SxmInstruction::kTotalLanes * hw::kSxmLaneIndexBits);
}
template <typename Write>
void write_sxm_macro_native(const SxmInstruction& ins, bool transpose, Write write) {
    if (!transpose || !SxmFifoTranspose::accepts(ins))
        throw std::invalid_argument("only the FIFO full-tile SXM transpose can execute; recompile the program");
    write(ins.src_streams[0].stream,6);
    write(ins.dst_streams[0].stream,6);
    write(kSxmFifoTransposeRow,SxmEncodingLayout::kRowBits);
}
template <typename Read>
SxmInstruction read_sxm_macro_native(bool transpose, Read read) {
    if (!transpose) throw std::invalid_argument("standalone SXM Permute is retired");
    const auto src=read(6), dst=read(6);
    if (read(SxmEncodingLayout::kRowBits)!=kSxmFifoTransposeRow)
        throw std::invalid_argument("retired SXM transpose encoding; recompile for FIFO timing");
    auto result=SxmInstruction::Transpose({{src},{src+1}},{{dst},{dst+1}});
    if (!SxmFifoTranspose::accepts(result)) throw std::invalid_argument("invalid FIFO stream pair");
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
