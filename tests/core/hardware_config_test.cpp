#include "ftlpu/core/hardware_config.hpp"
#include "ftlpu/core/fp16.hpp"
#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/mem/sram.hpp"
#include "ftlpu/mxm/array.hpp"
#include "ftlpu/sxm/distributor.hpp"
#include "ftlpu/sxm/unit_group.hpp"
#include "ftlpu/vxm/alu.hpp"

#include <cassert>
#include <cstdint>
#include <sstream>

namespace {

template <typename Config>
consteval bool common_config_contract()
{
    using D = ftlpu::hw::ConfigDerived<Config>;
    return ftlpu::hw::valid_hardware_config<Config>()
        && D::stream_vector_bytes == Config::sram_row_bytes
        && Config::sram_slice_capacity_bytes == 5 * 1024 * 1024 / 2
        && D::ifetch_packet_count == 40
        && Config::icu_iq_capacity_bytes / Config::ifetch_block_bytes >= 1;
}

static_assert(common_config_contract<ftlpu::hw::GroqLikeConfig>());
static_assert(common_config_contract<ftlpu::hw::TransformerEvalConfig>());
static_assert(
    ftlpu::hw::ConfigDerived<ftlpu::hw::GroqLikeConfig>::
        mxm_weight_block_load_cycles == 1);
static_assert(
    ftlpu::hw::ConfigDerived<ftlpu::hw::TransformerEvalConfig>::
        mxm_weight_block_load_cycles == 1);

} // namespace

int main()
{
#ifdef FTLPU_TRANSFORMER_EVAL_CONFIG
    static_assert(ftlpu::hw::kTileRows == 4);
    static_assert(ftlpu::hw::kLanesPerTile == 8);
    static_assert(ftlpu::hw::kPhysicalVectorBytes == 32);
    static_assert(ftlpu::hw::kMxmRows == 32);
    static_assert(ftlpu::hw::kMxmColumns == 32);
    static_assert(ftlpu::hw::kMxmCount == 4);
    static_assert(ftlpu::hw::kWestMxmCount == 2);
    static_assert(ftlpu::hw::kEastMxmCount == 2);
    static_assert(ftlpu::hw::kSramWordsPerBank == 40960);
    static_assert(ftlpu::hw::kSramSliceBytes == 5 * 1024 * 1024 / 2);
    static_assert(ftlpu::hw::kIcuFetchVectorCount == 20);
    static_assert(ftlpu::MemLocalWordAddress13::kBits == 17);
    static_assert(ftlpu::MemSliceByteAddress17::kBits == 20);
    static_assert(ftlpu::MemGlobalAddress24::kBits == 27);

    ftlpu::MxmSupercell fp16_cell;
    ftlpu::MxmSupercell::InputVector fp16_weights{};
    const auto one_and_half = ftlpu::Fp16::from_float(1.5f).bits();
    for (std::size_t lane = 0;
         lane < ftlpu::hw::kLanesPerTile;
         ++lane) {
        for (std::size_t column = 0;
             column < ftlpu::hw::kMxmSupercellColumns;
             ++column) {
            fp16_weights[lane][2 * column] =
                ftlpu::MxmSupercell::InputWord {
                    static_cast<std::uint8_t>(one_and_half & 0xffu), false};
            fp16_weights[lane][2 * column + 1] =
                ftlpu::MxmSupercell::InputWord {
                    static_cast<std::uint8_t>(one_and_half >> 8),
                    column + 1 == ftlpu::hw::kMxmSupercellColumns};
        }
    }
    std::ostringstream fp16_log;
    fp16_cell.set_input(fp16_weights);
    fp16_cell.issue(ftlpu::MxmInstruction::IW(0));
    fp16_cell.tick(fp16_log);
    assert(fp16_cell.weight(0, 0, 0) == 1.5f);
    ftlpu::MxmSupercell::ActivationData fp16_activation{};
    fp16_activation.fill(2.0f);
    const auto fp16_partial =
        fp16_cell.compute_partial(fp16_activation, 0);
    assert(fp16_partial[0] == 24.0f);
#else
    static_assert(ftlpu::hw::kTileRows == 20);
    static_assert(ftlpu::hw::kLanesPerTile == 16);
    static_assert(ftlpu::hw::kPhysicalVectorBytes == 320);
    static_assert(ftlpu::hw::kMxmRows == 320);
    static_assert(ftlpu::hw::kMxmCount == 2);
    static_assert(ftlpu::hw::kSramWordsPerBank == 4096);
    static_assert(ftlpu::hw::kIcuFetchVectorCount == 2);
    static_assert(ftlpu::MemLocalWordAddress13::kBits == 13);
    static_assert(ftlpu::MemSliceByteAddress17::kBits == 17);
    static_assert(ftlpu::MemGlobalAddress24::kBits == 24);
#endif

    // MEM: exercise the last configured SRAM row and its MEM ISA round trip.
    ftlpu::SramSlice sram;
    const auto last = ftlpu::MemLocalWordAddress13::FromFields(
        ftlpu::hw::kSramBanksPerTileBlock - 1,
        ftlpu::hw::kSramWordsPerBank - 1);
    ftlpu::StreamPayloadVector320 vector{};
    vector.back().back() = 0xa5;
    sram.write_vector(last, vector);
    assert(sram.read_vector(last) == vector);

    const auto mem = ftlpu::MemInstruction::Read(last, ftlpu::StreamId::East(3));
    assert(ftlpu::isa::decode_mem_instruction(
        ftlpu::isa::encode_mem_instruction(mem)).address == last);

    // MXM, SXM and VXM objects all derive their dimensions from ActiveConfig.
    ftlpu::MxmArray mxm;
    mxm.reset();
    (void)mxm.cell(
        ftlpu::hw::kMxmSupercellsPerPlane - 1,
        ftlpu::hw::kMxmSupercellsPerPlane - 1);

    const auto lane_map = ftlpu::Distribute16::identity_map();
    ftlpu::SxmUnitGroup<std::uint8_t> sxm;
    sxm.issue(ftlpu::SxmInstruction::Distribute({0}, {1}, lane_map));
    ftlpu::SxmUnitGroup<std::uint8_t>::StreamState inputs{};
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        inputs[0][lane] = ftlpu::StreamWord<std::uint8_t> {
            static_cast<std::uint8_t>(lane), lane + 1 == ftlpu::hw::kLanesPerTile};
    }
    const auto sxm_result = sxm.evaluate(inputs);
    assert(sxm_result.consumed[0]);
    assert(sxm_result.produced[1]);

    ftlpu::VxmAlu::Vector vxm_input{};
    vxm_input.fill(2.0f);
    const auto vxm_output = ftlpu::VxmAlu::execute(
        ftlpu::VxmAluInstruction {ftlpu::VxmAluOpcode::Square},
        vxm_input);
    for (const auto value : vxm_output) {
        assert(value == 4.0f);
    }
    return 0;
}
