#include "ftlpu/core/hardware_config.hpp"
#include "ftlpu/core/fp16.hpp"
#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/mem/sram.hpp"
#include "ftlpu/mxm/array.hpp"
#include "ftlpu/sxm/distributor.hpp"
#include "ftlpu/sxm/unit_group.hpp"

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
        && Config::icu_fetch_latency_cycles == 1
        && Config::icu_mxm_instruction_bits == 32
        && Config::icu_vxm_instruction_bits == 96
        && Config::icu_mem_imem_depth >= Config::icu_mem_iq_depth;
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
    static_assert(ftlpu::hw::kIcuFetchLatencyCycles == 1);
    static_assert(ftlpu::hw::kIcuMxmInstructionBits == 32);
    static_assert(ftlpu::MemLocalWordAddress13::kBits == 17);
    static_assert(ftlpu::MemSliceByteAddress17::kBits == 20);
    static_assert(ftlpu::MemGlobalAddress24::kBits == 27);

    ftlpu::MxmSupercell fp16_cell;
    ftlpu::MxmSupercell::InputVector encoded_weights{};
    const auto one_and_half = ftlpu::Fp16::from_float(1.5f).bits();
    std::ostringstream fp16_log;
    if constexpr (
        ftlpu::MxmSupercell::kRequiresWeightDequantization) {
        ftlpu::MxmSupercell::InputVector scales{};
        for (std::size_t lane = 0;
             lane < ftlpu::hw::kLanesPerTile; ++lane) {
            for (std::size_t column = 0;
                 column < ftlpu::hw::kMxmSupercellColumns; ++column) {
                scales[lane][2 * column] =
                    ftlpu::MxmSupercell::InputWord {
                        static_cast<std::uint8_t>(one_and_half & 0xffu),
                        false};
                scales[lane][2 * column + 1] =
                    ftlpu::MxmSupercell::InputWord {
                        static_cast<std::uint8_t>(one_and_half >> 8),
                        column + 1
                            == ftlpu::hw::kMxmSupercellColumns};
                encoded_weights[lane][column] =
                    ftlpu::MxmSupercell::InputWord {
                        std::uint8_t{1},
                        column + 1
                            == ftlpu::hw::kMxmSupercellColumns};
            }
        }
        fp16_cell.set_input(scales);
        fp16_cell.issue(ftlpu::MxmInstruction::LoadScales(0));
        fp16_cell.tick(fp16_log);
    } else {
        for (std::size_t lane = 0;
             lane < ftlpu::hw::kLanesPerTile; ++lane) {
            for (std::size_t column = 0;
                 column < ftlpu::hw::kMxmSupercellColumns; ++column) {
                encoded_weights[lane][2 * column] =
                    ftlpu::MxmSupercell::InputWord {
                        static_cast<std::uint8_t>(one_and_half & 0xffu),
                        false};
                encoded_weights[lane][2 * column + 1] =
                    ftlpu::MxmSupercell::InputWord {
                        static_cast<std::uint8_t>(one_and_half >> 8),
                        column + 1
                            == ftlpu::hw::kMxmSupercellColumns};
            }
        }
    }
    fp16_cell.set_input(encoded_weights);
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
    static_assert(ftlpu::hw::kIcuFetchLatencyCycles == 1);
    static_assert(ftlpu::hw::kIcuVxmInstructionBits == 96);
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

    // MXM and SXM objects derive their dimensions from ActiveConfig.
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

    return 0;
}
