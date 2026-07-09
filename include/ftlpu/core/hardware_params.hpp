#pragma once

#include <cstddef>

namespace ftlpu::hw {

constexpr std::size_t kTileRows = 20;
constexpr std::size_t kSliceColumns = 44;
constexpr std::size_t kSlicesPerGroup = 4;
constexpr std::size_t kSliceGroups = kSliceColumns / kSlicesPerGroup;

constexpr std::size_t kStreamRegisterColumns = kSliceGroups + 1;
constexpr std::size_t kLanesPerTile = 16;
constexpr std::size_t kStreams = 64;
constexpr std::size_t kEastStreams = 32;
constexpr std::size_t kWestStreams = 32;
constexpr std::size_t kStreamRegisterBytes = 1;
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
constexpr std::size_t kPhysicalVectorBytes = 320;
constexpr std::size_t kSramDepthWords = 8192;
constexpr std::size_t kSramBlockBytes = kPhysicalVectorBytes * kSramDepthWords;
constexpr std::size_t kTotalSramBytes = kSramBlocks * kSramBlockBytes;
constexpr std::size_t kPublicTotalSramBytes = kPublicSramBlocks * kSramBlockBytes;

static_assert(kSliceColumns % kSlicesPerGroup == 0);
static_assert(kStreamRegisterColumns == 12);
static_assert(kModeledSramBlocks == kSliceColumns);
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
