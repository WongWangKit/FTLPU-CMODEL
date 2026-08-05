#include "ftlpu/mem/mem_array.hpp"
#include "ftlpu/sxm/slice.hpp"
#include "ftlpu/vxm/static_quantizer.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kMemSlice = 0;
constexpr std::size_t kAddressBase = 64;
const auto kDataStream = ftlpu::StreamId::East(0);
const auto kTransposeStream = ftlpu::StreamId::East(1);
const auto kOutputStream = ftlpu::StreamId::East(2);

std::int8_t source_value(
    std::size_t tile,
    std::size_t row,
    std::size_t lane)
{
    const auto linear =
        tile * ftlpu::hw::kLanesPerTile * ftlpu::hw::kLanesPerTile
        + row * ftlpu::hw::kLanesPerTile + lane;
    return static_cast<std::int8_t>(
        static_cast<int>(linear % 127) - 63);
}

ftlpu::StreamPayloadSegment16 advance_quantizers(
    std::array<ftlpu::VxmStaticQuantizer,
               ftlpu::hw::kLanesPerTile>& quantizers,
    std::optional<std::pair<std::size_t, std::size_t>> next)
{
    auto segment = ftlpu::StreamPayloadSegment16{};
    for (std::size_t lane = 0;
         lane < ftlpu::hw::kLanesPerTile;
         ++lane) {
        auto requests = std::vector<ftlpu::VxmStaticQuantizer::Request>{};
        if (next.has_value()) {
            requests.push_back({
                static_cast<float>(source_value(
                    next->first, next->second, lane)),
                1.0f,
                0,
                0});
        }
        const auto completed = quantizers[lane].tick(
            std::move(requests));
        if (!completed.empty()) {
            assert(completed.size() == 1);
            segment[lane] = static_cast<std::uint8_t>(
                completed.front().value);
        }
    }
    return segment;
}

void evaluate_cycle(
    ftlpu::StreamRegisterFabric& fabric,
    ftlpu::MemArrayModel& mem,
    ftlpu::SxmSlice& sxm,
    std::optional<std::pair<std::size_t,
                            ftlpu::StreamPayloadSegment16>> vxm_output = {})
{
    fabric.begin_cycle();
    mem.evaluate(fabric);
    sxm.evaluate(fabric);
    fabric.stage_linear_links();
    if (vxm_output.has_value()) {
        fabric.stage_payload_segment(
            0,
            vxm_output->first,
            kDataStream,
            vxm_output->second,
            0,
            "VXM static quantizer");
    }
    fabric.commit_cycle();
}

} // namespace

int main()
{
    auto fabric = ftlpu::StreamRegisterFabric(
        ftlpu::hw::kMemBoundaryStreamRegisterColumns);
    auto mem = ftlpu::MemArrayModel{};
    auto sxm = ftlpu::SxmSlice(
        ftlpu::SxmStreamPortMap::SameDirection(1, 2));
    auto quantizers = std::array<
        ftlpu::VxmStaticQuantizer,
        ftlpu::hw::kLanesPerTile>{};

    // Each VXM row is quantized, written into SR0, and consumed by a
    // compiler-scheduled MEM Write one cycle later. The Write instruction
    // and tile data advance north in the same wave.
    for (std::size_t row = 0;
         row < ftlpu::hw::kLanesPerTile;
         ++row) {
        static_cast<void>(advance_quantizers(
            quantizers, std::pair {std::size_t {0}, row}));
        for (std::size_t tile = 0;
             tile < ftlpu::hw::kTileRows;
             ++tile) {
            const auto next = tile + 1 < ftlpu::hw::kTileRows
                ? std::optional {std::pair {tile + 1, row}}
                : std::nullopt;
            const auto output = advance_quantizers(
                quantizers, next);
            if (tile == 1) {
                mem.enqueue_instruction(
                    kMemSlice,
                    ftlpu::MemInstruction::Write(
                        kAddressBase + row, kDataStream));
            }
            evaluate_cycle(
                fabric, mem, sxm,
                std::pair {tile, output});
        }
        if (ftlpu::hw::kTileRows == 1) {
            mem.enqueue_instruction(
                kMemSlice,
                ftlpu::MemInstruction::Write(
                    kAddressBase + row, kDataStream));
        }
        evaluate_cycle(fabric, mem, sxm);
    }

    for (std::size_t tile = 0;
         tile < ftlpu::hw::kTileRows;
         ++tile) {
        for (std::size_t row = 0;
             row < ftlpu::hw::kLanesPerTile;
             ++row) {
            for (std::size_t lane = 0;
                 lane < ftlpu::hw::kLanesPerTile;
                 ++lane) {
                assert(mem.sram_lane_byte(
                           kMemSlice,
                           tile,
                           kAddressBase + row,
                           lane)
                    == static_cast<std::uint8_t>(
                        source_value(tile, row, lane)));
            }
        }
    }

    const auto transpose = ftlpu::SxmInstruction::Transpose(
        {{kDataStream.packed()}},
        {{kTransposeStream.packed()}},
        1);
    const auto permute = ftlpu::SxmInstruction::Permute(
        {{kTransposeStream.packed()}},
        {{kOutputStream.packed()}},
        ftlpu::Permute320::identity_map(),
        ftlpu::SxmWeightLayout::VectorColumns,
        1);

    // MEM rows are read on consecutive cycles. Transpose starts one cycle
    // later, exactly when the corresponding read reaches its SR boundary.
    const auto last_capture_cycle =
        ftlpu::hw::kLanesPerTile + ftlpu::hw::kTileRows - 1;
    for (std::size_t cycle = 0;
         cycle <= last_capture_cycle;
         ++cycle) {
        if (cycle < ftlpu::hw::kLanesPerTile) {
            mem.enqueue_instruction(
                kMemSlice,
                ftlpu::MemInstruction::Read(
                    kAddressBase + cycle, kDataStream));
        }
        if (cycle > 0
            && cycle <= ftlpu::hw::kLanesPerTile) {
            sxm.issue(transpose);
        }
        evaluate_cycle(fabric, mem, sxm);
    }
    evaluate_cycle(fabric, mem, sxm); // registered Transpose boundary

    for (std::size_t column = 0;
         column < ftlpu::hw::kLanesPerTile;
         ++column) {
        sxm.issue(permute);
        evaluate_cycle(fabric, mem, sxm);
        for (std::size_t tile = 0;
             tile < ftlpu::hw::kTileRows;
             ++tile) {
            for (std::size_t lane = 0;
                 lane < ftlpu::hw::kLanesPerTile;
                 ++lane) {
                assert(fabric.cell(
                           2, tile, lane, kOutputStream).data
                    == static_cast<std::uint8_t>(
                        source_value(tile, lane, column)));
            }
        }
    }
    return 0;
}
