#include "ftlpu/workload/design_space.hpp"

#include <cassert>
#include <cmath>

int main()
{
    using namespace ftlpu::workload;

    const auto exact = evaluate_gemm({"exact", 128, 1536, 576}, 32);
    assert(exact.m_blocks == 4);
    assert(exact.n_blocks == 48);
    assert(exact.k_blocks == 18);
    assert(exact.tail_rows == 0);
    assert(exact.tail_cols == 0);
    assert(exact.tail_k == 0);
    assert(exact.pe_fill_ratio() == 1.0);
    assert(exact.design.fp16_weight_bytes_per_array == 2048);
    assert(exact.design.fp16_weight_load_cycles == 16);
    assert(exact.design.aggregate_streams == 32);
    assert(exact.design.stream_deficit == 0);

    const auto tail = evaluate_gemm({"tail", 130, 577, 1537}, 32);
    assert(tail.m_blocks == 5);
    assert(tail.n_blocks == 19);
    assert(tail.k_blocks == 49);
    assert(tail.tail_rows == 30);
    assert(tail.tail_cols == 31);
    assert(tail.tail_k == 31);
    assert(tail.pe_fill_ratio() > 0.75);
    assert(tail.pe_fill_ratio() < 0.76);

    const auto tiles = plan_zero_padded_tiles(
        {"small_tail", 33, 35, 37}, 32, 32, 32);
    assert(tiles.size() == 8);
    assert(tiles.front().valid_rows == 32);
    assert(tiles.front().valid_cols == 32);
    assert(tiles.front().valid_k == 32);
    assert(tiles.back().m0 == 32);
    assert(tiles.back().n0 == 32);
    assert(tiles.back().k0 == 32);
    assert(tiles.back().valid_rows == 1);
    assert(tiles.back().valid_cols == 3);
    assert(tiles.back().valid_k == 5);

    const auto groq = sram_geometry<ftlpu::hw::GroqLikeConfig>();
    const auto transformer =
        sram_geometry<ftlpu::hw::TransformerEvalConfig>();
    assert(groq.slice_capacity_bytes == transformer.slice_capacity_bytes);
    assert(groq.row_bytes == 320);
    assert(groq.rows_per_bank == 4096);
    assert(transformer.row_bytes == 32);
    assert(transformer.rows_per_bank == 40960);

    const auto conflict = evaluate_gemm(
        {"conflict", 32, 32, 32},
        32,
        WeightLoadModel {2, 8, 16, 32, 4});
    assert(conflict.design.aggregate_streams == 64);
    assert(conflict.design.stream_deficit == 32);
    return 0;
}
