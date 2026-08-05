#include "ftlpu/mxm/accumulator_slice.hpp"

#include <cassert>
#include <cstdint>
#include <vector>

namespace {

ftlpu::Mxm::ColumnOutput output(
    std::int32_t value,
    ftlpu::MxmAccumulatorMode mode,
    ftlpu::MxmPairMode pair = ftlpu::MxmPairMode::Independent,
    std::size_t stream = 0)
{
    auto result = ftlpu::Mxm::ColumnOutput {
        7, 0, {}, stream, 4, mode, pair};
    result.values.fill(value);
    return result;
}

void empty_cycle(
    ftlpu::MxmAccumulatorSlice& accumulator,
    ftlpu::StreamRegisterFabric& fabric)
{
    fabric.begin_cycle();
    accumulator.evaluate(fabric, {});
    fabric.commit_cycle();
}

} // namespace

int main()
{
    constexpr auto kColumn =
        ftlpu::hw::kMemBoundaryStreamRegisterColumns - 1;
    auto fabric = ftlpu::StreamRegisterFabric(
        ftlpu::hw::kMemBoundaryStreamRegisterColumns);
    auto accumulator = ftlpu::MxmAccumulatorSlice(
        {kColumn, ftlpu::StreamDirection::West});

    // A single-result context stays entirely in ACC0; no MEM bank exists.
    fabric.begin_cycle();
    accumulator.evaluate(
        fabric,
        {output(100, ftlpu::MxmAccumulatorMode::LocalStart)});
    fabric.commit_cycle();
    assert(accumulator.local_row(0, 0).value() == 7);
    assert(accumulator.local_value(0, 0, 3) == 100);

    fabric.begin_cycle();
    accumulator.evaluate(
        fabric,
        {output(23, ftlpu::MxmAccumulatorMode::LocalFinalize)});
    fabric.commit_cycle();
    assert(!accumulator.local_row(0, 0).has_value());
    empty_cycle(accumulator, fabric);
    empty_cycle(accumulator, fabric);
    assert(accumulator.last_outputs().size() == 1);
    assert(accumulator.last_outputs()[0].final);
    assert(accumulator.last_outputs()[0].values[0] == 123);

    // Multi-context accumulation uses normal four-byte int32 SR/MEM traffic;
    // MEM itself performs only compiler-scheduled Read and Write operations.
    fabric.begin_cycle();
    accumulator.evaluate(
        fabric,
        {output(
            30, ftlpu::MxmAccumulatorMode::MemoryStart,
            ftlpu::MxmPairMode::Independent, 4)});
    fabric.commit_cycle();
    empty_cycle(accumulator, fabric);
    // Model the compiler-scheduled MEM read returning the stored int32
    // partial on the direction opposite to ACC result output.
    for (std::size_t byte = 0; byte < sizeof(std::int32_t); ++byte) {
        for (std::size_t lane = 0;
             lane < ftlpu::hw::kLanesPerTile;
             ++lane) {
            fabric.initialize_cell(
                kColumn,
                0,
                lane,
                ftlpu::StreamId::East(4 + byte),
                ftlpu::StreamCell::Valid(
                    byte == 0 ? 30 : 0));
        }
    }
    fabric.begin_cycle();
    accumulator.evaluate(
        fabric,
        {output(
            12, ftlpu::MxmAccumulatorMode::MemoryFinalize,
            ftlpu::MxmPairMode::Independent, 0)});
    fabric.commit_cycle();
    empty_cycle(accumulator, fabric);
    empty_cycle(accumulator, fabric);
    assert(accumulator.last_outputs().size() == 1);
    assert(accumulator.last_outputs()[0].values[4] == 42);

    // Fixed merge wiring: ACC0 combines the two MXMs; ACC1 is the next stage.
    const auto lhs = output(
        40, ftlpu::MxmAccumulatorMode::DirectFinal,
        ftlpu::MxmPairMode::Merge, 2);
    const auto rhs = output(
        2, ftlpu::MxmAccumulatorMode::DirectFinal,
        ftlpu::MxmPairMode::Merge, 2);
    fabric.begin_cycle();
    accumulator.evaluate(fabric, {lhs}, {rhs});
    fabric.commit_cycle();
    empty_cycle(accumulator, fabric);
    empty_cycle(accumulator, fabric);
    empty_cycle(accumulator, fabric);
    assert(accumulator.last_outputs().size() == 1);
    assert(accumulator.last_outputs()[0].values[5] == 42);

    const auto expected = ftlpu::MxmOutputCast::bytes(42);
    assert(
        fabric.cell(
            kColumn, 0, 5, ftlpu::StreamId::West(2)).data
        == expected[0]);
    assert(
        fabric.cell(
            kColumn, 0, 5, ftlpu::StreamId::West(3)).data
        == expected[1]);

    // The compiler writes activation and weight scales into the fixed path
    // register. Their product is applied combinationally in the one-cycle
    // final cast; partial sums remain unscaled int32 values.
    auto scaled_fabric = ftlpu::StreamRegisterFabric(
        ftlpu::hw::kMemBoundaryStreamRegisterColumns);
    auto scaled_accumulator = ftlpu::MxmAccumulatorSlice(
        {kColumn, ftlpu::StreamDirection::West});
    scaled_accumulator.configure_output_dequant_scale(0, 0.25f, 0.5f);
    assert(scaled_accumulator.output_dequant_scale(0) == 0.125f);
    scaled_fabric.begin_cycle();
    scaled_accumulator.evaluate(
        scaled_fabric,
        {output(40, ftlpu::MxmAccumulatorMode::DirectFinal)});
    scaled_fabric.commit_cycle();
    empty_cycle(scaled_accumulator, scaled_fabric);
    empty_cycle(scaled_accumulator, scaled_fabric);
    const auto scaled_expected = ftlpu::MxmOutputCast::bytes(40, 0.125f);
    assert(
        scaled_fabric.cell(
            kColumn, 0, 0, ftlpu::StreamId::West(0)).data
        == scaled_expected[0]);
    assert(
        scaled_fabric.cell(
            kColumn, 0, 0, ftlpu::StreamId::West(1)).data
        == scaled_expected[1]);

    return 0;
}
