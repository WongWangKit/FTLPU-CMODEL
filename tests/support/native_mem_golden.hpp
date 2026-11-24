#pragma once
#include "ftlpu/icu/macro_bitstream_decoder.hpp"
#include <bit>
#include <vector>
namespace ftlpu::test {
// Preserve the hand-encoded opcode/schedule goldens. Only the absolute MEM
// row field between the dictionary/start and run prefix has target width.
using MemWords = std::vector<ftlpu::IcuRawImemWord96>;
inline MemWords physical_mem_image(MemWords reference, std::size_t dictionaryEntries, unsigned row)
{
    constexpr unsigned width = std::bit_width(ftlpu::hw::kSramDepthRows - 1);
    const std::size_t field = 96 + 3 + 36*dictionaryEntries + 32;
    MemWords image(reference.size() + (width>13));
    const auto set = [&](std::size_t pos) { image.at(pos/96).lanes[pos%96/32] |= 1u<<(pos%32); };
    for (std::size_t pos=0; pos<field; ++pos) if(reference.at(pos/96).bit(pos%96)) set(pos);
    for (unsigned bit=0;bit<width;++bit) if(row & (1u<<bit)) set(field+bit);
    for (std::size_t pos=field+13;pos<reference.size()*96;++pos)
        if(reference[pos/96].bit(pos%96))set(pos-13+width);
    return image;
}
}
