#pragma once

#include <cstddef>

namespace ftlpu::hw {

// Architecture inputs are deliberately semantic rather than inferred from
// tile/lane products. ConfigDerived contains only relationships that are true
// for every supported design point.
struct GroqLikeConfig {
    static constexpr std::size_t vector_tile_count = 20;
    static constexpr std::size_t lanes_per_tile = 16;
    static constexpr std::size_t stream_vector_lanes =
        vector_tile_count * lanes_per_tile;
    static constexpr std::size_t lane_element_bytes = 1;

    static constexpr std::size_t streams_per_direction = 32;
    static constexpr std::size_t stream_register_bytes = 1;

    static constexpr std::size_t mem_slice_count = 44;
    static constexpr std::size_t mem_slices_per_group = 4;
    static constexpr std::size_t mem_read_lanes_per_cycle = 16;
    static constexpr std::size_t mem_write_lanes_per_cycle = 16;

    static constexpr std::size_t mxm_m = stream_vector_lanes;
    static constexpr std::size_t mxm_n = stream_vector_lanes;
    static constexpr std::size_t mxm_k = stream_vector_lanes;
    static constexpr std::size_t mxm_block_rows = lanes_per_tile;
    static constexpr std::size_t mxm_block_columns = lanes_per_tile;
    static constexpr std::size_t mxm_count = 2;
    static constexpr std::size_t hemisphere_count = 2;
    static constexpr std::size_t west_mxm_count = 0;
    static constexpr std::size_t east_mxm_count = 2;
    static constexpr std::size_t mxm_weight_bytes_per_value = 1;
    static constexpr std::size_t mxm_stored_weight_bytes_per_value = 1;
    static constexpr std::size_t mxm_activation_bytes_per_value = 1;
    static constexpr std::size_t mxm_weight_load_streams = 16;

    // SRAM capacity is an input. It is not inferred by keeping depth constant
    // while changing the vector width.
    static constexpr std::size_t sram_banks_per_slice = 2;
    static constexpr std::size_t sram_bank_depth_rows = 4096;
    static constexpr std::size_t sram_row_bytes =
        stream_vector_lanes * lane_element_bytes;
    static constexpr std::size_t sram_slice_capacity_bytes =
        2 * 4096 * 320;

    static constexpr std::size_t sxm_operation_lanes = 16;

    // Distributed ICU storage. Widths describe one local function/control
    // word; depths are independently configurable per function family.
    static constexpr std::size_t instruction_packet_bytes = 16;
    static constexpr std::size_t icu_fetch_latency_cycles = 1;
    static constexpr std::size_t icu_vxm_instruction_bits = 96;
    static constexpr std::size_t icu_mem_instruction_bits = 96;
    static constexpr std::size_t icu_mxm_instruction_bits = 32;
    static constexpr std::size_t icu_sxm_instruction_bits = 96;
    static constexpr std::size_t icu_vxm_imem_depth = 2048;
    static constexpr std::size_t icu_mem_imem_depth = 4096;
    static constexpr std::size_t icu_mxm_imem_depth = 2048;
    static constexpr std::size_t icu_sxm_imem_depth = 2048;
    static constexpr std::size_t icu_vxm_iq_depth = 16;
    static constexpr std::size_t icu_mem_iq_depth = 16;
    static constexpr std::size_t icu_mxm_iq_depth = 16;
    static constexpr std::size_t icu_sxm_iq_depth = 16;
};

struct TransformerEvalConfig {
    static constexpr std::size_t vector_tile_count = 4;
    static constexpr std::size_t lanes_per_tile = 8;
    static constexpr std::size_t stream_vector_lanes =
        vector_tile_count * lanes_per_tile;
    static constexpr std::size_t lane_element_bytes = 1;

    static constexpr std::size_t streams_per_direction = 32;
    static constexpr std::size_t stream_register_bytes = 1;

    static constexpr std::size_t mem_slice_count = 44;
    static constexpr std::size_t mem_slices_per_group = 4;
    static constexpr std::size_t mem_read_lanes_per_cycle = 8;
    static constexpr std::size_t mem_write_lanes_per_cycle = 8;

    static constexpr std::size_t mxm_m = stream_vector_lanes;
    static constexpr std::size_t mxm_n = stream_vector_lanes;
    static constexpr std::size_t mxm_k = stream_vector_lanes;
    static constexpr std::size_t mxm_block_rows = lanes_per_tile;
    static constexpr std::size_t mxm_block_columns = lanes_per_tile;
    static constexpr std::size_t mxm_count = 4;
    static constexpr std::size_t hemisphere_count = 2;
    static constexpr std::size_t west_mxm_count = 2;
    static constexpr std::size_t east_mxm_count = 2;
    static constexpr std::size_t mxm_weight_bytes_per_value = 2;
    static constexpr std::size_t mxm_stored_weight_bytes_per_value = 1;
    static constexpr std::size_t mxm_activation_bytes_per_value = 2;
    static constexpr std::size_t mxm_weight_load_streams = 16;

    // Keep 2.5 MiB per MEM slice explicitly. The narrower 32-byte vector row
    // therefore increases depth instead of silently shrinking capacity.
    static constexpr std::size_t sram_banks_per_slice = 2;
    static constexpr std::size_t sram_bank_depth_rows = 40960;
    static constexpr std::size_t sram_row_bytes =
        stream_vector_lanes * lane_element_bytes;
    static constexpr std::size_t sram_slice_capacity_bytes =
        2 * 40960 * 32;

    static constexpr std::size_t sxm_operation_lanes = 16;

    static constexpr std::size_t instruction_packet_bytes = 16;
    static constexpr std::size_t icu_fetch_latency_cycles = 1;
    static constexpr std::size_t icu_vxm_instruction_bits = 96;
    static constexpr std::size_t icu_mem_instruction_bits = 96;
    static constexpr std::size_t icu_mxm_instruction_bits = 32;
    static constexpr std::size_t icu_sxm_instruction_bits = 96;
    static constexpr std::size_t icu_vxm_imem_depth = 2048;
    static constexpr std::size_t icu_mem_imem_depth = 4096;
    static constexpr std::size_t icu_mxm_imem_depth = 2048;
    static constexpr std::size_t icu_sxm_imem_depth = 2048;
    static constexpr std::size_t icu_vxm_iq_depth = 16;
    static constexpr std::size_t icu_mem_iq_depth = 16;
    static constexpr std::size_t icu_mxm_iq_depth = 16;
    static constexpr std::size_t icu_sxm_iq_depth = 16;
};

template <typename Config>
struct ConfigDerived {
    static constexpr std::size_t stream_vector_bytes =
        Config::stream_vector_lanes * Config::lane_element_bytes;
    static constexpr std::size_t mem_group_count =
        Config::mem_slice_count / Config::mem_slices_per_group;
    static constexpr std::size_t mem_boundary_columns =
        mem_group_count + 1;
    static constexpr std::size_t mxm_block_row_count =
        Config::vector_tile_count;
    static constexpr std::size_t mxm_block_column_count =
        Config::vector_tile_count;
    static constexpr std::size_t mxm_weight_load_bytes_per_cycle =
        Config::lanes_per_tile
        * (Config::mxm_block_columns
            * Config::mxm_stored_weight_bytes_per_value)
        * Config::stream_register_bytes;
    static constexpr std::size_t mxm_weight_block_bytes =
        Config::mxm_block_rows
        * Config::mxm_block_columns
        * Config::mxm_stored_weight_bytes_per_value;
    static constexpr std::size_t mxm_stored_weight_load_streams =
        Config::mxm_block_columns
        * Config::mxm_stored_weight_bytes_per_value;
    static constexpr std::size_t mxm_weight_scale_streams =
        Config::mxm_stored_weight_bytes_per_value
                == Config::mxm_weight_bytes_per_value
        ? 0
        : Config::mxm_block_columns * 2;
    static constexpr std::size_t mxm_weight_block_load_cycles =
        mxm_weight_block_bytes / mxm_weight_load_bytes_per_cycle;
    static constexpr std::size_t mem_read_bytes_per_cycle =
        Config::mem_read_lanes_per_cycle * Config::stream_register_bytes;
    static constexpr std::size_t mem_write_bytes_per_cycle =
        Config::mem_write_lanes_per_cycle * Config::stream_register_bytes;
    static constexpr std::size_t sram_bank_capacity_bytes =
        Config::sram_bank_depth_rows * Config::sram_row_bytes;
};

template <typename Config>
consteval bool valid_hardware_config()
{
    using D = ConfigDerived<Config>;
    return Config::vector_tile_count * Config::lanes_per_tile
            == Config::stream_vector_lanes
        && Config::sram_row_bytes == D::stream_vector_bytes
        && Config::mxm_block_rows == Config::lanes_per_tile
        && Config::mxm_block_columns == Config::lanes_per_tile
        && Config::mxm_m == Config::stream_vector_lanes
        && Config::mxm_n == Config::stream_vector_lanes
        && Config::mxm_k == Config::stream_vector_lanes
        && Config::mem_slice_count % Config::mem_slices_per_group == 0
        && Config::mxm_m % Config::mxm_block_rows == 0
        && Config::mxm_n % Config::mxm_block_columns == 0
        && Config::mxm_k % Config::mxm_block_rows == 0
        && Config::mxm_weight_load_streams
            <= Config::streams_per_direction
        && D::mxm_stored_weight_load_streams
            <= Config::mxm_weight_load_streams
        && D::mxm_weight_scale_streams
            <= Config::mxm_weight_load_streams
        && Config::mxm_activation_bytes_per_value
            <= Config::streams_per_direction
        && D::mxm_weight_block_bytes
            % D::mxm_weight_load_bytes_per_cycle == 0
        && Config::west_mxm_count + Config::east_mxm_count
            == Config::mxm_count
        && Config::west_mxm_count
            * Config::mxm_weight_load_streams
            <= Config::streams_per_direction
        && Config::east_mxm_count
            * Config::mxm_weight_load_streams
            <= Config::streams_per_direction
        && Config::sram_banks_per_slice * D::sram_bank_capacity_bytes
            == Config::sram_slice_capacity_bytes
        && Config::instruction_packet_bytes > 0
        && Config::icu_fetch_latency_cycles > 0
        && Config::icu_vxm_instruction_bits > 0
        && Config::icu_mem_instruction_bits > 0
        && Config::icu_mxm_instruction_bits > 0
        && Config::icu_sxm_instruction_bits > 0
        && Config::icu_vxm_imem_depth >= Config::icu_vxm_iq_depth
        && Config::icu_mem_imem_depth >= Config::icu_mem_iq_depth
        && Config::icu_mxm_imem_depth >= Config::icu_mxm_iq_depth
        && Config::icu_sxm_imem_depth >= Config::icu_sxm_iq_depth;
}

static_assert(
    valid_hardware_config<GroqLikeConfig>(),
    "GroqLikeConfig violates vector, MXM, SRAM, hemisphere, or ICU constraints");
static_assert(
    valid_hardware_config<TransformerEvalConfig>(),
    "TransformerEvalConfig violates vector, MXM, SRAM, hemisphere, or ICU constraints");
static_assert(
    GroqLikeConfig::sram_slice_capacity_bytes
        == TransformerEvalConfig::sram_slice_capacity_bytes,
    "Transformer evaluation must not shrink SRAM capacity when vector rows narrow");

} // namespace ftlpu::hw
