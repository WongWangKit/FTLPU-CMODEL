#pragma once

#include <cstddef>

namespace ftlpu::hw {

constexpr std::size_t kTileRows = 20;
constexpr std::size_t kLanesPerTile = 16;
constexpr std::size_t kPhysicalVectorBytes = kTileRows * kLanesPerTile;

// Stream identity is 0..31 plus a direction.  kStreams is retained as the
// packed ISA selector count (E0..E31, W0..W31).
constexpr std::size_t kStreamsPerDirection = 32;
constexpr std::size_t kEastStreams = kStreamsPerDirection;
constexpr std::size_t kWestStreams = kStreamsPerDirection;
constexpr std::size_t kStreams = kEastStreams + kWestStreams;
constexpr std::size_t kStreamRegisterBytes = 1;

// MEM/SRAM geometry for one modeled hemisphere.
constexpr std::size_t kMemSliceColumns = 44;
constexpr std::size_t kMemSlicesPerGroup = 4;
constexpr std::size_t kMemGroups = kMemSliceColumns / kMemSlicesPerGroup;
constexpr std::size_t kMemBoundaryStreamRegisterColumns = kMemGroups + 1;

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

constexpr std::size_t kMemLanesPerCycle = kLanesPerTile;
constexpr std::size_t kMemReadBytesPerCycle = kMemLanesPerCycle * kStreamRegisterBytes;
constexpr std::size_t kMemWriteBytesPerCycle = kMemLanesPerCycle * kStreamRegisterBytes;

constexpr std::size_t kMxmRows = 320;
constexpr std::size_t kMxmColumns = 320;
constexpr std::size_t kMxmSupercellRows = 16;
constexpr std::size_t kMxmSupercellColumns = 16;
constexpr std::size_t kMxmSupercellsPerPlane = 20;
constexpr std::size_t kMxmLoadStreamsPerCycle = 16;
constexpr std::size_t kMxmLoadBytesPerCycle = kLanesPerTile * kMxmLoadStreamsPerCycle * kStreamRegisterBytes;

constexpr std::size_t kSxmConcurrentStreamOps = 16;

constexpr std::size_t kHemispheres = 2;
constexpr std::size_t kPublicSramBlocks = 88;
constexpr std::size_t kModeledSramBlocks = kPublicSramBlocks / kHemispheres;
constexpr std::size_t kSramBlocksPerSlice = 1;
constexpr std::size_t kSramBlocks = kModeledSramBlocks;
constexpr std::size_t kSramRowBytes = kPhysicalVectorBytes;
constexpr std::size_t kSramDepthRows = 8192;
// Compatibility alias for code that historically called a 320-byte row a word.
constexpr std::size_t kSramDepthWords = kSramDepthRows;
constexpr std::size_t kSramBlockBytes = kSramRowBytes * kSramDepthRows;
constexpr std::size_t kTotalSramBytes = kSramBlocks * kSramBlockBytes;
constexpr std::size_t kPublicTotalSramBytes = kPublicSramBlocks * kSramBlockBytes;

static_assert(kPhysicalVectorBytes == 320);
static_assert(kMemSliceColumns % kMemSlicesPerGroup == 0);
static_assert(kMemBoundaryStreamRegisterColumns == 12);
static_assert(kModeledSramBlocks == kMemSliceColumns);
static_assert(kPublicSramBlocks == 88);
static_assert(kSramBlocks == 44);
static_assert(kEastStreams + kWestStreams == kStreams);
static_assert(kLanesPerTile == 16);
static_assert(kMemReadBytesPerCycle == 16);
static_assert(kMemWriteBytesPerCycle == 16);
static_assert(kMxmRows == kMxmSupercellRows * kMxmSupercellsPerPlane);
static_assert(kMxmColumns == kMxmSupercellColumns * kMxmSupercellsPerPlane);
static_assert(kMxmLoadStreamsPerCycle == kMxmSupercellColumns);
static_assert(kMxmLoadBytesPerCycle == 256);
static_assert(kSxmConcurrentStreamOps == 16);
static_assert(kSramBlockBytes == 5 * 1024 * 1024 / 2);
static_assert(kTotalSramBytes == 110 * 1024 * 1024);
static_assert(kPublicTotalSramBytes == 220 * 1024 * 1024);

} // namespace ftlpu::hw
