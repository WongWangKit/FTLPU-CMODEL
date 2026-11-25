#include "ftlpu/icu/sxm_macro_instruction.hpp"
#include "ftlpu/mem/mem_array.hpp"
#include "ftlpu/sxm/slice.hpp"

#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

}

int main() try
{
    using namespace ftlpu;
    const auto ports = MemStreamPortMap::BetweenBoundaries();
    for (std::size_t slice = 0; slice < hw::kMemSliceColumns; ++slice) {
        const auto group = slice / hw::kMemSlicesPerGroup;
        require(ports.input_column(slice, StreamDirection::East) == group,
            "MEM grouping changed with physical tile count");
        require(ports.output_column(slice, StreamDirection::West) == group,
            "reverse MEM grouping changed with physical tile count");
    }
    using Codec=IcuSxmActiveMacroContextCodec<true>;
    const auto transpose=SxmInstruction::Transpose({{0},{1}},{{32},{33}});
    const IcuMacroSchedule schedule{0,3,hw::kPhysicalVectorBytes,0,1,1,0,IcuInductionTarget::None};
    require(Codec::unpack(Codec::pack_initial(schedule,transpose)).instruction.dst_streams==transpose.dst_streams,
        "FIFO transpose stream lost");
    for (std::size_t row=0; row<hw::kLanesPerTile; ++row) {
        const auto iw = MxmControlInstruction::IWColumnDirect16(1, hw::kTileRows-1, row, 16);
        const auto native = isa::encode_mxm_instruction(iw);
        const auto packed = isa::pack_mxm_load_template(native);
        if constexpr (isa::MxmEncodingLayout::kExtendedTemplates)
            require((packed >> isa::MxmEncodingLayout::kWsTemplateBits) == 0, "WS template truncated");
        const auto load = isa::decode_mxm_instruction(isa::unpack_mxm_load_template(packed));
        require(load.weight_inner_column == row && load.weight_column == iw.weight_column
            && load.weight_stream_base == 16 && load.weight_input_mode == iw.weight_input_mode,
            "MXM IW high selector or stream lost");
    }
    if constexpr (hw::kPhysicalVectorBytes == 64) {
        require(hw::kMemSliceColumns == 52 && hw::kMemSlicesPerGroup == 4
            && hw::kMemGroups == 13 && hw::kSystemStreamRegisterColumns == 16,
            "8-tile target changed the MEM/SR topology");
    }
    std::cout << "geometry_encoding_test passed: tiles=" << hw::kTileRows
        << " mem_group=" << hw::kMemSlicesPerGroup << " lanes=" << hw::kPhysicalVectorBytes
        << " sxm_words=" << Codec::word_count << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
