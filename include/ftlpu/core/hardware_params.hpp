#pragma once

#include "ftlpu/core/hardware_config.hpp"

#include <cstddef>
#include <string_view>

namespace ftlpu::hw {

constexpr std::size_t kHardwareSchemaVersion = config::kSchemaVersion;
constexpr std::string_view kTargetName = config::kTargetName;
constexpr std::size_t kHemispheres = config::kHemispheres;
constexpr std::size_t kTileRows = config::kTilesPerSlice;
constexpr std::size_t kLanesPerTile = config::kLanesPerTile;
constexpr std::size_t kMemBytesPerLane = config::kMemBytesPerLane;
constexpr std::size_t kPhysicalVectorBytes =
    kTileRows * kLanesPerTile * kMemBytesPerLane;
constexpr std::size_t kReferenceVectorBytes = 320;
constexpr std::size_t kVectorWidthScale =
    kReferenceVectorBytes / kPhysicalVectorBytes;

// Stream identity is 0..31 plus a direction.  kStreams is retained as the
// packed ISA selector count (E0..E31, W0..W31).
constexpr std::size_t kStreams = config::kStreamRegistersPerLane;
constexpr std::size_t kStreamsPerDirection = kStreams / 2;
constexpr std::size_t kEastStreams = kStreamsPerDirection;
constexpr std::size_t kWestStreams = kStreamsPerDirection;
constexpr std::size_t kStreamRegisterBytes =
    config::kBytesPerStreamPerLane;

// C2C exposes independent external transport lanes. Each lane carries one
// complete 32-byte vector per cycle; the default receive path then injects the
// vector into an ordinary E/W stream for on-chip transport and MEM consumption.
constexpr std::size_t kC2cStreamsPerDirection = kLanesPerTile;
constexpr std::size_t kC2cBytesPerStreamPerCycle = kPhysicalVectorBytes;
constexpr std::size_t kC2cBytesPerDirectionPerCycle =
    kC2cStreamsPerDirection * kC2cBytesPerStreamPerCycle;

// MEM/SRAM geometry for one modeled hemisphere.
constexpr std::size_t kMemSliceColumns = config::kMemSlicesPerHemisphere;
constexpr std::size_t kMemBanksPerSlice = config::kMemBanksPerSlice;
constexpr std::size_t kMemSlicesPerGroup = kTileRows;
constexpr std::size_t kMemGroups = kMemSliceColumns / kMemSlicesPerGroup;
constexpr std::size_t kMemBoundaryStreamRegisterColumns = kMemGroups + 1;
constexpr std::size_t kC2cToSxmStreamRegisterColumns = 1;
constexpr std::size_t kSxmToMxmStreamRegisterColumns = 1;
constexpr std::size_t kSystemStreamRegisterColumns =
    kMemBoundaryStreamRegisterColumns
    + kC2cToSxmStreamRegisterColumns
    + kSxmToMxmStreamRegisterColumns;
constexpr std::size_t kMemWestBoundaryStreamRegisterColumn = 0;
constexpr std::size_t kMemEastBoundaryStreamRegisterColumn =
    kMemBoundaryStreamRegisterColumns - 1;
constexpr std::size_t kC2cSxmBoundaryStreamRegisterColumn =
    kMemEastBoundaryStreamRegisterColumn + kC2cToSxmStreamRegisterColumns;
constexpr std::size_t kMxmBoundaryStreamRegisterColumn =
    kSystemStreamRegisterColumns - 1;

// Compatibility names used by the existing code.  New code should use the
// MEM-specific names above instead of assuming that the whole chip has only
// a fixed number of SR columns.
constexpr std::size_t kSliceColumns = kMemSliceColumns;
constexpr std::size_t kSlicesPerGroup = kMemSlicesPerGroup;
constexpr std::size_t kSliceGroups = kMemGroups;
constexpr std::size_t kStreamRegisterColumns = kSystemStreamRegisterColumns;

// Figure-4 eastward path count supplied by the architecture study.  The MEM
// region uses a mapped subset of these physical columns; it does not own them.
constexpr std::size_t kEastPathStreamRegisterColumns = 21;

constexpr std::size_t kMemLanesPerCycle = kLanesPerTile;
constexpr std::size_t kMemReadBytesPerCycle = kMemLanesPerCycle * kStreamRegisterBytes;
constexpr std::size_t kMemWriteBytesPerCycle = kMemLanesPerCycle * kStreamRegisterBytes;

constexpr std::size_t kMxmRows = kPhysicalVectorBytes;
constexpr std::size_t kMxmColumns = kPhysicalVectorBytes;
constexpr std::size_t kMxmSupercellRows = kLanesPerTile;
constexpr std::size_t kMxmSupercellColumns = kLanesPerTile;
constexpr std::size_t kMxmSupercellsPerPlane = kTileRows;
constexpr std::size_t kMxmWeightBytesPerValue = 2;
constexpr std::size_t kMxmLoadStreamsPerCycle = kMxmSupercellColumns * kMxmWeightBytesPerValue;
constexpr std::size_t kMxmColumnLoadStreamsPerCycle = kMxmWeightBytesPerValue;
constexpr std::size_t kMxmInt8WeightStreamsPerCycle =
    kMxmSupercellColumns;
constexpr std::size_t kMxmInt8LoadStreamsPerCycle =
    kMxmInt8WeightStreamsPerCycle;
constexpr std::size_t kMxmInt8ColumnLoadStreamsPerCycle = 1;
constexpr std::size_t kMxmLoadStreamStride = kMxmLoadStreamsPerCycle;
constexpr std::size_t kMxmInt8LoadStreamStride =
    kMxmInt8LoadStreamsPerCycle;
constexpr std::size_t kMxmActivationStreamsPerVector = 2;
constexpr std::size_t kMxmLoadBytesPerCycle = kLanesPerTile * kMxmLoadStreamsPerCycle * kStreamRegisterBytes;
// One partial-sum block contains a complete 32x32 FP32 output tile.
constexpr std::size_t kMxmAccumulatorBlockCount =
    config::kMxmAccumulatorContexts;
constexpr std::size_t kMxmAccumulatorRows =
    kMxmAccumulatorBlockCount * kMxmRows;
constexpr std::size_t kMxmAccumulatorBytes =
    kMxmAccumulatorRows * kMxmColumns * sizeof(float);

constexpr bool kMxmSupportsNative4x4 = config::kMxmSupportsNative4x4;
constexpr bool kMxmSupportsLinear1x16 = config::kMxmSupportsLinear1x16;

constexpr std::size_t kVxmAlusPerDirection =
    config::kVxmAlusPerDirection;
constexpr std::size_t kVxmAluCount = 2 * kVxmAlusPerDirection;
constexpr std::size_t kSxmConcurrentStreamOps = 2 * kLanesPerTile;

// Distributed ICU geometry. Every functional queue owns local instruction
// memory and a finite prefetch IQ; instruction fetch never consumes MEM/SR
// bandwidth.
constexpr std::size_t kIcuFetchLatencyCycles = 1;
constexpr std::size_t kIcuBarrierLatencyCycles = 35;
constexpr std::size_t kIcuVxmInstructionBits = 96;
constexpr std::size_t kIcuMemInstructionBits = 96;
constexpr std::size_t kIcuMxmInstructionBits = 128;
constexpr std::size_t kIcuSxmInstructionBits = 96;
constexpr std::size_t kIcuC2cInstructionBits = 96;
constexpr std::size_t kIcuC2cDmaInstructionBits = 192;
constexpr std::size_t kIcuVxmImemDepth = 2048;
// Existing whole-layer schedules are emitted as one flat program image. The
// runtime frontend still exposes only a 16-entry IQ and one fetch per cycle.
constexpr std::size_t kIcuMemImemDepth = 131072;
// A Qwen2.5-1.5B seq128 decoder layer needs just over 32K compressed
// compute commands in each MXM queue. Keep the next power-of-two capacity so
// one complete layer can remain resident without runtime i-MEM paging.
constexpr std::size_t kIcuMxmImemDepth = 65536;
constexpr std::size_t kIcuSxmImemDepth = 2048;
constexpr std::size_t kIcuC2cImemDepth = 32768;
constexpr std::size_t kIcuVxmIqDepth = 16;
constexpr std::size_t kIcuMemIqDepth = 16;
constexpr std::size_t kIcuMxmIqDepth = 16;
constexpr std::size_t kIcuSxmIqDepth = 16;
constexpr std::size_t kIcuC2cIqDepth = 16;

constexpr std::size_t kMxmsPerHemisphere = 2;
constexpr std::size_t kMxmCount = kHemispheres * kMxmsPerHemisphere;
constexpr std::size_t kModeledSramBlocks =
    kMemSliceColumns * kMemBanksPerSlice;
constexpr std::size_t kPublicSramBlocks =
    kHemispheres * kModeledSramBlocks;
// Every MEM slice owns two independent single-port vector-wide SRAM banks.
constexpr std::size_t kSramBlocksPerSlice = kMemBanksPerSlice;
constexpr std::size_t kSramBlocks = kModeledSramBlocks;
constexpr std::size_t kSramRowBytes = kPhysicalVectorBytes;
constexpr std::size_t kSramDepthRows = config::kMemRowsPerBank;
constexpr std::size_t kMemSliceDepthRows =
    kSramDepthRows * kMemBanksPerSlice;
// Compatibility alias for code that historically called a vector row a word.
constexpr std::size_t kSramDepthWords = kSramDepthRows;
constexpr std::size_t kSramBlockBytes = kSramRowBytes * kSramDepthRows;
constexpr std::size_t kMemSliceBytes =
    kSramBlocksPerSlice * kSramBlockBytes;
constexpr std::size_t kReferenceMemSliceBytes =
    kReferenceVectorBytes * kMemSliceDepthRows;
constexpr std::size_t kTotalSramBytes = kSramBlocks * kSramBlockBytes;
constexpr std::size_t kPublicTotalSramBytes = kPublicSramBlocks * kSramBlockBytes;

static_assert(kHardwareSchemaVersion == 1);
static_assert(!kTargetName.empty());
static_assert(kHemispheres > 0);
static_assert(kTileRows > 0);
static_assert(kLanesPerTile > 0);
static_assert(kStreams > 0 && kStreams <= 64 && kStreams % 2 == 0);
static_assert(kStreamRegisterBytes == kMemBytesPerLane);
static_assert(kReferenceVectorBytes % kPhysicalVectorBytes == 0);
static_assert(kMemSliceColumns % kMemSlicesPerGroup == 0);
static_assert(kModeledSramBlocks
    == kMemSliceColumns * kMemBanksPerSlice);
static_assert(kEastStreams + kWestStreams == kStreams);
static_assert(kMxmRows == kMxmSupercellRows * kMxmSupercellsPerPlane);
static_assert(kMxmColumns == kMxmSupercellColumns * kMxmSupercellsPerPlane);
static_assert(kMxmInt8ColumnLoadStreamsPerCycle == 1);
static_assert(kMxmLoadStreamsPerCycle <= kStreamsPerDirection);
static_assert(kMxmInt8LoadStreamsPerCycle <= kStreamsPerDirection);
static_assert(kMxmAccumulatorBlockCount > 0);
static_assert(kMxmAccumulatorBlockCount <= 256,
    "the 13-bit MXM accumulator address supports at most 256 blocks");
static_assert(kMxmAccumulatorRows
    == kMxmAccumulatorBlockCount * kMxmRows);
static_assert(kSxmConcurrentStreamOps == 16);
static_assert(kIcuVxmImemDepth >= kIcuVxmIqDepth);
static_assert(kIcuMemImemDepth >= kIcuMemIqDepth);
static_assert(kIcuMxmImemDepth >= kIcuMxmIqDepth);
static_assert(kIcuSxmImemDepth >= kIcuSxmIqDepth);
static_assert(kIcuC2cImemDepth >= kIcuC2cIqDepth);
static_assert(kMemSliceBytes
    == kPhysicalVectorBytes * kMemSliceDepthRows);
static_assert(kReferenceMemSliceBytes
    == kReferenceVectorBytes * kMemSliceDepthRows);
static_assert(kMemSliceBytes * kVectorWidthScale
    == kReferenceMemSliceBytes);

} // namespace ftlpu::hw
