#pragma once
#include "ftlpu/sxm/matrix.hpp"
#include <bit>
#include <cassert>
#include <functional>

namespace sxm_test {
using namespace ftlpu;
constexpr std::size_t T = hw::kTileRows;
inline std::uint32_t value(SxmMatrixDtype dtype, std::size_t m, std::size_t r, std::size_t c)
{
    // Distinct lower FP32 planes; finite values for the real VXM numeric check.
    const auto bits = 0x3e800001u + std::uint32_t(m * 0x800000 + r * 0x18000 + c * 0x301);
    return dtype == SxmMatrixDtype::Float32 ? bits : bits >> 16;
}
inline float number(SxmMatrixDtype dtype, std::uint32_t v)
{ return std::bit_cast<float>(dtype == SxmMatrixDtype::Float32 ? v : v << 16); }
inline SxmMatrixCore::Inputs inputs(const SxmMatrixStart& job, std::uint64_t age)
{
    SxmMatrixCore::Inputs in{};
    for (std::size_t s = 0; s < T; ++s) {
        if (age < 1+s || age-1-s >= job.matrices*8*T) continue;
        const auto i = age-1-s, m = i/(8*T), r = i%(8*T);
        SxmMatrixCore::Row row;
        for (std::size_t l = 0; l < 8; ++l) row.words[l] = value(job.dtype, m, r, s*8+l);
        in[s] = row;
    }
    return in;
}
inline std::size_t inverse(const SxmMatrixStart& job, std::size_t d)
{ for (std::size_t a=0;a<T;++a) if (job.tile_map[a]==d) return a; assert(false); return 0; }
inline std::uint32_t expected(const SxmMatrixStart& job, std::size_t m,
    std::size_t output_row, std::size_t d, std::size_t l)
{ return value(job.dtype,m,8*inverse(job,d)+job.lane_map[d][l],output_row); }
inline SxmMatrixStart job(SxmMatrixDtype dtype, bool mapped = false)
{
    SxmMatrixStart j; j.matrices=2; j.dtype=dtype; j.mem_base=17;
    if (mapped) for (std::size_t a=0;a<T;++a) {
        j.tile_map[a]=T-1-a;
        for (std::size_t l=0;l<8;++l) j.lane_map[a][l]=(3*l+a)%8;
    }
    return j;
}
template<class F> inline void rejects(F f)
{ bool caught=false; try {f();} catch(const std::exception&) {caught=true;} assert(caught); }
} // namespace sxm_test
