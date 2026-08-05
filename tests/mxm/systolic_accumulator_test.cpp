#include "ftlpu/mxm/mxm.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <sstream>

namespace {

ftlpu::MxmArray::InputVector uniform_weights(std::int8_t value)
{
    ftlpu::MxmArray::InputVector input{};
    for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
        for (std::size_t stream = 0; stream < ftlpu::hw::kMxmLoadStreamsPerCycle; ++stream) {
            input[lane][stream] = ftlpu::MxmSupercell::InputWord {
                value,
                stream + 1 == ftlpu::hw::kMxmLoadStreamsPerCycle,
            };
        }
    }
    return input;
}

void load_uniform_weight_buffer(ftlpu::Mxm& mxm, std::size_t buffer, std::int8_t value)
{
    std::ostringstream ignored;
    const auto input = uniform_weights(value);
    for (std::size_t row = 0; row < ftlpu::hw::kMxmSupercellsPerPlane; ++row) {
        for (std::size_t column = 0; column < ftlpu::hw::kMxmSupercellsPerPlane; ++column) {
            mxm.array().tick_cell_iw_load(row, column, buffer, input, ignored);
        }
    }
}

void stage_uniform_activation(ftlpu::StreamRegisterFabric& fabric, std::int8_t value)
{
    const auto column = ftlpu::hw::kMemBoundaryStreamRegisterColumns - 1;
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            fabric.initialize_cell(
                column,
                tile,
                lane,
                ftlpu::StreamId::East(0),
                ftlpu::StreamCell::Valid(
                    static_cast<std::uint8_t>(value),
                    lane + 1 == ftlpu::hw::kLanesPerTile));
        }
    }
}

struct RunResult {
    std::size_t outputs{0};
    std::size_t first_output_cycle{static_cast<std::size_t>(-1)};
    std::size_t last_output_cycle{0};
};

RunResult run_adjacent_k_blocks(
    ftlpu::Mxm& mxm,
    ftlpu::StreamRegisterFabric& fabric,
    ftlpu::MxmControlInstruction first,
    ftlpu::MxmControlInstruction second)
{
    // No bubble: both tokens enter MAC row zero on adjacent cycles. The
    // 16-stage broadcast-MAC array keeps II=1 while different MAC rows work
    // on different token rows.
    mxm.control().issue_south(first);
    mxm.control().issue_south(second);
    RunResult result{};
    constexpr std::size_t kCycles =
        ftlpu::hw::kMxmK
        + ftlpu::hw::kMxmSupercellsPerPlane + 5;
    for (std::size_t cycle = 0; cycle < kCycles; ++cycle) {
        stage_uniform_activation(fabric, 1);
        fabric.begin_cycle();
        mxm.evaluate(fabric, 0);
        if (!mxm.last_outputs().empty()) {
            if (result.outputs == 0) {
                result.first_output_cycle = cycle;
            }
            result.last_output_cycle = cycle;
            for (const auto& output : mxm.last_outputs()) {
                for (const auto value : output.values) {
                    assert(value == 320 || value == 640);
                }
                assert(
                    output.accumulator_mode
                    == ftlpu::MxmAccumulatorMode::DirectFinal);
                ++result.outputs;
            }
        }
        fabric.commit_cycle();
    }
    return result;
}

} // namespace

int main()
{
    // Verify the internal diagonal wavefront explicitly. At cycle 15, MAC
    // row 0 consumes token row 15 while MAC row 15 finishes token row 0.
    auto skewed = std::make_unique<ftlpu::Mxm>();
    auto skewed_fabric = std::make_unique<ftlpu::StreamRegisterFabric>(
        ftlpu::hw::kMemBoundaryStreamRegisterColumns);
    load_uniform_weight_buffer(*skewed, 0, 1);
    skewed->control().issue_south(
        ftlpu::MxmControlInstruction::ComputeAccumulating(
            0, 0,
            ftlpu::MxmAccumulatorMode::DirectFinal,
            0, ftlpu::MxmPairMode::Independent, true));
    for (std::size_t token = 1;
         token < ftlpu::MxmSupercell::kMacPipelineStages;
         ++token) {
        skewed->control().issue_south(
            ftlpu::MxmControlInstruction::Compute(0, 0));
    }
    for (std::size_t cycle = 0;
         cycle < ftlpu::MxmSupercell::kMacPipelineStages;
         ++cycle) {
        stage_uniform_activation(*skewed_fabric, 1);
        skewed_fabric->begin_cycle();
        skewed->evaluate(*skewed_fabric, 0);
        skewed_fabric->commit_cycle();
    }
    for (std::size_t mac_row = 0;
         mac_row < ftlpu::MxmSupercell::kMacPipelineStages;
         ++mac_row) {
        assert(
            skewed->mac_stage_token_row(0, 0, mac_row).value()
            == ftlpu::MxmSupercell::kMacPipelineStages
                - 1 - mac_row);
    }
    stage_uniform_activation(*skewed_fabric, 1);
    skewed_fabric->begin_cycle();
    skewed->evaluate(*skewed_fabric, 0);
    skewed_fabric->commit_cycle();
    assert(skewed->mac_stage_token_row(1, 0, 0).value() == 0);

    auto mxm = std::make_unique<ftlpu::Mxm>();
    auto fabric = std::make_unique<ftlpu::StreamRegisterFabric>(
        ftlpu::hw::kMemBoundaryStreamRegisterColumns);
    load_uniform_weight_buffer(*mxm, 0, 1);
    load_uniform_weight_buffer(*mxm, 1, 2);

    // MXM now ends at the raw K-block boundary: adjacent blocks leave as 320
    // and 640. The shared hemisphere accumulator owns all cross-block state.
    const auto raw = run_adjacent_k_blocks(
        *mxm,
        *fabric,
        ftlpu::MxmControlInstruction::ComputeAccumulating(
            0, 0,
            ftlpu::MxmAccumulatorMode::DirectFinal,
            0, ftlpu::MxmPairMode::Independent, true),
        ftlpu::MxmControlInstruction::ComputeAccumulating(
            1, 2,
            ftlpu::MxmAccumulatorMode::DirectFinal,
            0, ftlpu::MxmPairMode::Independent, true));
    assert(raw.outputs == 2 * ftlpu::hw::kMxmSupercellsPerPlane);
    assert(
        raw.first_output_cycle
        == ftlpu::hw::kMxmK - 1);
    assert(
        raw.last_output_cycle
        == ftlpu::hw::kMxmK
            + ftlpu::hw::kMxmSupercellsPerPlane - 1);

    return 0;
}
