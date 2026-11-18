#pragma once
#include "ftlpu/icu/macro_bitstream_decoder.hpp"
#include <vector>

namespace ftlpu::test {
// Test-side assembler. The DUT consumes only the packed 96-bit iMEM image.
struct SxmMacroFixture {
    std::vector<std::pair<IcuMacroSchedule,SxmInstruction>> entries;
    void add(std::size_t cycle, std::size_t count, std::size_t interval,
             const SxmInstruction& instruction) {
        entries.push_back({{cycle,count,interval,0,1,1,0,IcuInductionTarget::None},instruction});
    }
    std::vector<IcuRawImemWord96> image(bool transpose) const {
        std::vector<IcuRawImemWord96> result(1);
        result[0].lanes[0]=7u | (static_cast<std::uint32_t>(entries.size())<<4);
        if(entries.empty()) return result;
        std::size_t offset=96;
        const auto bits=[&](std::uint64_t value,unsigned width) {
            for(unsigned i=0;i<width;++i,++offset) {
                if(result.size()<=offset/96) result.emplace_back();
                result[offset/96].lanes[(offset%96)/32]|=std::uint32_t((value>>i)&1u)<<(offset%32);
            }
        };
        bits(0,3); // No delta dictionary.
        bits(entries.front().first.start_cycle,32);
        for(std::size_t i=0;i<entries.size();++i) {
            const auto& [schedule,instruction]=entries[i];
            bits(0,1); // One template, one descriptor.
            const bool compact=schedule.inner_count<=2048 && schedule.inner_interval<=256;
            bits(!compact,1);
            if(compact) bits(schedule.inner_count==1?0:1,2);
            isa::write_sxm_macro_native(instruction,transpose,bits);
            if(compact) {
                if(schedule.inner_count>1) {
                    bits(schedule.inner_count-1,11);bits(schedule.inner_interval-1,8);bits(0,10);
                }
            } else {
                bits(schedule.inner_count,32);bits(schedule.inner_interval,32);bits(0,32);
                bits(1,32);bits(1,32);bits(0,32);bits(0,2);
            }
            if(i) {
                bits(31,5);bits(1,1); // Wide delta escape.
                bits(schedule.start_cycle-entries[i-1].first.start_cycle,32);bits(0,32);
            }
        }
        return result;
    }
    template<typename Queue> void load(Queue& queue,bool transpose) const {
        queue.load_raw_macro(transpose?IcuMacroQueueKind::SxmTranspose:IcuMacroQueueKind::SxmPermute,image(transpose));
    }
};
}
