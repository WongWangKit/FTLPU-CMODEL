#pragma once

#include "ftlpu/core/hardware_config.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace ftlpu::workload {

struct TransformerDimensions {
    std::size_t seq_len{128};
    std::size_t hidden{576};
    std::size_t ffn_dim{1536};
    std::size_t head_dim{64};
    std::size_t kv_hidden{192};
    std::size_t decode_m{1};
};

struct GemmShape {
    std::string_view name{};
    std::size_t m{0};
    std::size_t n{0};
    std::size_t k{0};
};

inline std::vector<GemmShape> transformer_gemms(
    const TransformerDimensions& dims = {})
{
    return {
        {"ffn_gate_up_prefill", dims.seq_len, dims.ffn_dim, dims.hidden},
        {"ffn_down_prefill", dims.seq_len, dims.hidden, dims.ffn_dim},
        {"attention_qk", dims.seq_len, dims.seq_len, dims.head_dim},
        {"attention_pv", dims.seq_len, dims.head_dim, dims.seq_len},
        {"ffn_gate_up_decode", dims.decode_m, dims.ffn_dim, dims.hidden},
    };
}

constexpr std::size_t ceil_div(std::size_t value, std::size_t divisor)
{
    return (value + divisor - 1) / divisor;
}

constexpr std::size_t round_up(std::size_t value, std::size_t multiple)
{
    return ceil_div(value, multiple) * multiple;
}

struct GemmTile {
    std::size_t m0{0};
    std::size_t n0{0};
    std::size_t k0{0};
    std::size_t valid_rows{0};
    std::size_t valid_cols{0};
    std::size_t valid_k{0};
};

// Tail contract for phase one: the planner emits explicit valid extents.
// Producers zero-fill invalid rows/columns/K lanes. Consumers never write
// output elements outside valid_rows/valid_cols.
inline std::vector<GemmTile> plan_zero_padded_tiles(
    GemmShape shape,
    std::size_t block_m,
    std::size_t block_n,
    std::size_t block_k)
{
    std::vector<GemmTile> result{};
    result.reserve(
        ceil_div(shape.m, block_m)
        * ceil_div(shape.n, block_n)
        * ceil_div(shape.k, block_k));
    for (std::size_t m0 = 0; m0 < shape.m; m0 += block_m) {
        for (std::size_t n0 = 0; n0 < shape.n; n0 += block_n) {
            for (std::size_t k0 = 0; k0 < shape.k; k0 += block_k) {
                result.push_back(GemmTile {
                    m0,
                    n0,
                    k0,
                    std::min(block_m, shape.m - m0),
                    std::min(block_n, shape.n - n0),
                    std::min(block_k, shape.k - k0),
                });
            }
        }
    }
    return result;
}

struct DesignPoint {
    std::size_t array_edge{0};
    std::size_t fp16_weight_bytes_per_array{0};
    std::size_t fp16_weight_load_cycles{0};
    std::size_t streams_per_mxm{0};
    std::size_t concurrent_mxms_per_hemisphere{0};
    std::size_t aggregate_streams{0};
    std::size_t stream_capacity{0};
    std::size_t stream_deficit{0};
};

struct GemmEvaluation {
    GemmShape shape{};
    DesignPoint design{};
    std::size_t m_blocks{0};
    std::size_t n_blocks{0};
    std::size_t k_blocks{0};
    std::size_t padded_m{0};
    std::size_t padded_n{0};
    std::size_t padded_k{0};
    std::size_t tail_rows{0};
    std::size_t tail_cols{0};
    std::size_t tail_k{0};
    std::uint64_t useful_macs{0};
    std::uint64_t scheduled_macs{0};

    double pe_fill_ratio() const
    {
        return scheduled_macs == 0
            ? 0.0
            : static_cast<double>(useful_macs)
                / static_cast<double>(scheduled_macs);
    }
};

struct WeightLoadModel {
    std::size_t fp16_bytes_per_value{2};
    std::size_t lanes_per_stream{8};
    std::size_t streams_per_mxm{16};
    std::size_t streams_per_direction{32};
    std::size_t concurrent_mxms_per_hemisphere{2};

    constexpr std::size_t bytes_per_mxm_cycle() const
    {
        return lanes_per_stream * streams_per_mxm;
    }
};

inline GemmEvaluation evaluate_gemm(
    GemmShape shape,
    std::size_t array_edge,
    WeightLoadModel load = {})
{
    const auto padded_m = round_up(shape.m, array_edge);
    const auto padded_n = round_up(shape.n, array_edge);
    const auto padded_k = round_up(shape.k, array_edge);
    const auto weight_bytes =
        array_edge * array_edge * load.fp16_bytes_per_value;
    const auto aggregate_streams =
        load.streams_per_mxm * load.concurrent_mxms_per_hemisphere;

    return GemmEvaluation {
        shape,
        DesignPoint {
            array_edge,
            weight_bytes,
            ceil_div(weight_bytes, load.bytes_per_mxm_cycle()),
            load.streams_per_mxm,
            load.concurrent_mxms_per_hemisphere,
            aggregate_streams,
            load.streams_per_direction,
            aggregate_streams > load.streams_per_direction
                ? aggregate_streams - load.streams_per_direction
                : 0,
        },
        ceil_div(shape.m, array_edge),
        ceil_div(shape.n, array_edge),
        ceil_div(shape.k, array_edge),
        padded_m,
        padded_n,
        padded_k,
        padded_m - shape.m,
        padded_n - shape.n,
        padded_k - shape.k,
        static_cast<std::uint64_t>(shape.m) * shape.n * shape.k,
        static_cast<std::uint64_t>(padded_m) * padded_n * padded_k,
    };
}

struct SramGeometry {
    std::size_t row_bytes{0};
    std::size_t banks{0};
    std::size_t rows_per_bank{0};
    std::size_t slice_capacity_bytes{0};
};

template <typename Config>
constexpr SramGeometry sram_geometry()
{
    return {
        Config::sram_row_bytes,
        Config::sram_banks_per_slice,
        Config::sram_bank_depth_rows,
        Config::sram_slice_capacity_bytes,
    };
}

} // namespace ftlpu::workload
