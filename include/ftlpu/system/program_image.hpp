#pragma once
#include "ftlpu/icu/macro_bitstream_decoder.hpp"
#include <limits>
#include <tuple>
namespace ftlpu {
enum class ProgramStorage { Imem, TransportImem, Qctl, Barr, QueueEvents, BarrierEvents, RoutingDescriptors };
// Lengths describe physical words, excluding the final DMA vector padding.
struct ProgramImage {
    std::uint64_t source_address{0};
    IcuMacroQueueKind kind{IcuMacroQueueKind::Mem};
    std::size_t queue_index{0}, byte_count{0};
    IcuQueueMode mode{IcuQueueMode::Macro};
    ProgramStorage storage{ProgramStorage::Imem};
    std::size_t physical_word_bytes{0};
    std::uint32_t end_cycle{0};
    std::uint64_t generation{0};
    std::size_t word_bytes() const noexcept {
        if (physical_word_bytes) return physical_word_bytes;
        switch(storage) {
        case ProgramStorage::Qctl: return 12;
        case ProgramStorage::Barr: return 8;
        case ProgramStorage::QueueEvents: case ProgramStorage::BarrierEvents: case ProgramStorage::RoutingDescriptors: return 4;
        default: return kind==IcuMacroQueueKind::MxmLoad || kind==IcuMacroQueueKind::MxmCompute
            || kind==IcuMacroQueueKind::MxmDequant ? 16 : 12;
        }
    }
    std::size_t word_count() const noexcept {return byte_count/word_bytes();}
    std::size_t vector_count() const noexcept {return byte_count/hw::kPhysicalVectorBytes
        +(byte_count%hw::kPhysicalVectorBytes!=0);}
    auto key() const {return std::make_tuple(storage,kind,queue_index);}
};
} // namespace ftlpu
