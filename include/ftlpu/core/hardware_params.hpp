#pragma once

#include "ftlpu/core/hardware_config.hpp"

#include <cstddef>

namespace ftlpu::hw {

#ifdef FTLPU_TRANSFORMER_EVAL_CONFIG
using ActiveConfig = TransformerEvalConfig;
#else
using ActiveConfig = GroqLikeConfig;
#endif
using ActiveDerived = ConfigDerived<ActiveConfig>;

constexpr std::size_t kTileRows = ActiveConfig::vector_tile_count;
constexpr std::size_t kLanesPerTile = ActiveConfig::lanes_per_tile;
constexpr std::size_t kPhysicalVectorBytes =
    ActiveDerived::stream_vector_bytes;

// Stream identity is 0..31 plus a direction.  kStreams is retained as the
// packed ISA selector count (E0..E31, W0..W31).
constexpr std::size_t kStreamsPerDirection =
    ActiveConfig::streams_per_direction;
constexpr std::size_t kEastStreams = kStreamsPerDirection;
constexpr std::size_t kWestStreams = kStreamsPerDirection;
constexpr std::size_t kStreams = kEastStreams + kWestStreams;
constexpr std::size_t kStreamRegisterBytes =
    ActiveConfig::stream_register_bytes;

// MEM/SRAM geometry for one modeled hemisphere.
constexpr std::size_t kMemSliceColumns = ActiveConfig::mem_slice_count;
constexpr std::size_t kMemSlicesPerGroup =
    ActiveConfig::mem_slices_per_group;
constexpr std::size_t kMemGroups = ActiveDerived::mem_group_count;
constexpr std::size_t kMemBoundaryStreamRegisterColumns =
    ActiveDerived::mem_boundary_columns;
// The two edge-most four-slice groups are the architectural FP32
// accumulator stores used by MXM reductions. One instruction on the group
// base owns all four byte slices for that cycle.
constexpr std::size_t kAccumulatorMemGroupCount = 2;
constexpr std::size_t kWestAccumulatorMemGroup =
    kMemGroups - kAccumulatorMemGroupCount;
constexpr std::size_t kEastAccumulatorMemGroup =
    kMemGroups - 1;
constexpr std::size_t kWestAccumulatorMemSliceBase =
    kWestAccumulatorMemGroup * kMemSlicesPerGroup;
constexpr std::size_t kEastAccumulatorMemSliceBase =
    kEastAccumulatorMemGroup * kMemSlicesPerGroup;

// Compatibility names used by the existing code.  New code should use the
// MEM-specific names above instead of assuming that the whole chip has only
// twelve SR columns.
constexpr std::size_t kSliceColumns = kMemSliceColumns;
constexpr std::size_t kSlicesPerGroup = kMemSlicesPerGroup;
constexpr std::size_t kSliceGroups = kMemGroups;
constexpr std::size_t kStreamRegisterColumns = kMemBoundaryStreamRegisterColumns;

// Figure-4 eastward path count supplied by the architecture study.  The MEM
// region uses a mapped subset of these physical columns; it does not own them.
constexpr std::size_t kEastPathStreamRegisterColumns = 21;

constexpr std::size_t kMemLanesPerCycle =
    ActiveConfig::mem_read_lanes_per_cycle;
constexpr std::size_t kMemReadBytesPerCycle =
    ActiveDerived::mem_read_bytes_per_cycle;
constexpr std::size_t kMemWriteBytesPerCycle =
    ActiveDerived::mem_write_bytes_per_cycle;

constexpr std::size_t kMxmRows = ActiveConfig::mxm_m;
constexpr std::size_t kMxmColumns = ActiveConfig::mxm_n;
constexpr std::size_t kMxmK = ActiveConfig::mxm_k;
constexpr std::size_t kMxmSupercellRows =
    ActiveConfig::mxm_block_rows;
constexpr std::size_t kMxmSupercellColumns =
    ActiveConfig::mxm_block_columns;
constexpr std::size_t kMxmSupercellsPerPlane =
    ActiveDerived::mxm_block_row_count;
constexpr std::size_t kMxmAccumulatorBanks =
    ActiveConfig::mxm_accumulator_banks;
constexpr std::size_t kMxmCount = ActiveConfig::mxm_count;
constexpr std::size_t kWestMxmCount = ActiveConfig::west_mxm_count;
constexpr std::size_t kEastMxmCount = ActiveConfig::east_mxm_count;
constexpr std::size_t kMxmWeightBytesPerValue =
    ActiveConfig::mxm_weight_bytes_per_value;
constexpr std::size_t kMxmStoredWeightBytesPerValue =
    ActiveConfig::mxm_stored_weight_bytes_per_value;
constexpr std::size_t kMxmActivationBytesPerValue =
    ActiveConfig::mxm_activation_bytes_per_value;
constexpr std::size_t kMxmLoadStreamsPerCycle =
    ActiveConfig::mxm_weight_load_streams;
constexpr std::size_t kMxmStoredWeightLoadStreams =
    ActiveDerived::mxm_stored_weight_load_streams;
constexpr std::size_t kMxmWeightScaleStreams =
    ActiveDerived::mxm_weight_scale_streams;
constexpr std::size_t kMxmLoadBytesPerCycle =
    ActiveDerived::mxm_weight_load_bytes_per_cycle;

// CModel scheduling/storage parameters.  These values make the model finite
// and testable; they are not claims about finalized hardware capacities.
constexpr std::size_t kEncodedInstructionPacketBytes =
    ActiveConfig::ifetch_packet_bytes;
constexpr std::size_t kIcuFetchVectorCount =
    ActiveDerived::ifetch_vector_count;
constexpr std::size_t kIcuFetchBufferBytes =
    ActiveConfig::ifetch_block_bytes;
constexpr std::size_t kIcuFetchPackets =
    ActiveDerived::ifetch_packet_count;
constexpr std::size_t kIcuIqCapacityBytes =
    ActiveConfig::icu_iq_capacity_bytes;
constexpr std::size_t kIcuBarrierLatencyCycles = 35;

constexpr std::size_t kSxmConcurrentStreamOps =
    ActiveConfig::sxm_operation_lanes;
constexpr std::size_t kVxmLaneCount = ActiveConfig::vxm_lane_count;
constexpr std::size_t kVxmPipelineStages =
    ActiveConfig::vxm_pipeline_stages;
constexpr std::size_t kVxmAluCount = ActiveConfig::vxm_alu_count;

constexpr std::size_t kHemispheres = ActiveConfig::hemisphere_count;
constexpr std::size_t kPublicMemSlices = kMemSliceColumns * kHemispheres;
constexpr std::size_t kSramTileBlocksPerSlice = kTileRows;
constexpr std::size_t kSramBanksPerTileBlock =
    ActiveConfig::sram_banks_per_slice;
constexpr std::size_t kSramWordsPerBank =
    ActiveConfig::sram_bank_depth_rows;
constexpr std::size_t kSramWordBytes = kLanesPerTile;
constexpr std::size_t kMemLocalWordAddressCount =
    kSramBanksPerTileBlock * kSramWordsPerBank;
constexpr std::size_t kSramBankBytes = kSramWordsPerBank * kSramWordBytes;
constexpr std::size_t kSramTileBlockBytes =
    kSramBanksPerTileBlock * kSramBankBytes;
constexpr std::size_t kSramSliceBytes =
    kSramTileBlocksPerSlice * kSramTileBlockBytes;
constexpr std::size_t kModeledSramTileBlocks =
    kMemSliceColumns * kSramTileBlocksPerSlice;
constexpr std::size_t kPublicSramTileBlocks =
    kPublicMemSlices * kSramTileBlocksPerSlice;
constexpr std::size_t kModeledSramBanks =
    kModeledSramTileBlocks * kSramBanksPerTileBlock;
constexpr std::size_t kPublicSramBanks =
    kPublicSramTileBlocks * kSramBanksPerTileBlock;
constexpr std::size_t kTotalSramBytes = kMemSliceColumns * kSramSliceBytes;
constexpr std::size_t kPublicTotalSramBytes = kPublicMemSlices * kSramSliceBytes;

static_assert(
    valid_hardware_config<ActiveConfig>(),
    "active hardware configuration violates vector, MXM, SRAM, VXM, "
    "hemisphere bandwidth, or IFetch constraints");
static_assert(
    kPhysicalVectorBytes
        == kTileRows * kLanesPerTile * ActiveConfig::lane_element_bytes,
    "stream vector width must equal the configured tile/lane storage width");
static_assert(
    kMemSliceColumns % kMemSlicesPerGroup == 0,
    "MEM slice count must be divisible by the MEM group size");
static_assert(
    kMemGroups >= kAccumulatorMemGroupCount,
    "MEM topology needs two four-slice FP32 accumulator groups");
static_assert(
    kMxmRows == kMxmSupercellRows * kMxmSupercellsPerPlane,
    "MXM M dimension must be exactly tiled by physical supercell rows");
static_assert(
    kMxmColumns == kMxmSupercellColumns
        * ActiveDerived::mxm_block_column_count,
    "MXM N dimension must be exactly tiled by physical supercell columns");
static_assert(
    kMxmLoadStreamsPerCycle <= kStreamsPerDirection,
    "one MXM weight port cannot consume more streams than one direction owns");
static_assert(
    kWestMxmCount * kMxmLoadStreamsPerCycle <= kStreamsPerDirection,
    "west-hemisphere MXMs oversubscribe the configured weight-load streams");
static_assert(
    kEastMxmCount * kMxmLoadStreamsPerCycle <= kStreamsPerDirection,
    "east-hemisphere MXMs oversubscribe the configured weight-load streams");
static_assert(
    kSramSliceBytes == ActiveConfig::sram_slice_capacity_bytes,
    "SRAM slice capacity must remain an explicit configuration input");
static_assert(
    kIcuFetchBufferBytes
            == kIcuFetchVectorCount * kPhysicalVectorBytes
        && kIcuFetchBufferBytes
            == kIcuFetchPackets * kEncodedInstructionPacketBytes,
    "IFetch block must contain an integral number of stream vectors and packets");
static_assert(
    kVxmLaneCount == kLanesPerTile,
    "current SR-facing VXM contract requires one VXM lane per tile lane");
static_assert(
    kVxmPipelineStages == kVxmAluCount,
    "current VXM pipeline implementation requires one stage per ALU queue");

#ifndef FTLPU_TRANSFORMER_EVAL_CONFIG
static_assert(kPhysicalVectorBytes == 320);
static_assert(kMemBoundaryStreamRegisterColumns == 12);
static_assert(kPublicMemSlices == 88);
static_assert(kSramTileBlocksPerSlice == 20);
static_assert(kSramBanksPerTileBlock == 2);
static_assert(kSramWordsPerBank == 4096);
static_assert(kSramWordBytes == 16);
static_assert(kMemLocalWordAddressCount == 8192);
static_assert(kMxmLoadBytesPerCycle == 256);
static_assert(kSramBankBytes == 64 * 1024);
static_assert(kSramTileBlockBytes == 128 * 1024);
static_assert(kSramSliceBytes == 5 * 1024 * 1024 / 2);
static_assert(kTotalSramBytes == 110 * 1024 * 1024);
static_assert(kPublicTotalSramBytes == 220 * 1024 * 1024);
#endif

} // namespace ftlpu::hw
