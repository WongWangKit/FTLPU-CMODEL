#include "ftlpu/sxm/slice.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace {

constexpr std::size_t kAttentionSize = 32;
constexpr std::size_t kBlockSize = ftlpu::hw::kLanesPerTile;
constexpr std::size_t kBlocks = kAttentionSize / kBlockSize;

using Matrix = std::array<std::array<std::int32_t, kAttentionSize>, kAttentionSize>;
using Plane = ftlpu::SxmSlice::StreamVector<std::int32_t>;
using Chunks = std::array<std::array<std::array<std::int32_t, kBlockSize>, kBlocks>, kAttentionSize>;
using UnitGroup = ftlpu::SxmUnitGroup<std::int32_t>;

std::int32_t value(std::size_t query, std::size_t key)
{
    return static_cast<std::int32_t>(query * 1000 + key);
}

ftlpu::SxmInstruction::StreamList stream_range(std::size_t first, std::size_t count)
{
    auto streams = ftlpu::SxmInstruction::StreamList {};
    streams.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        streams.push_back(ftlpu::SxmStreamId {first + index});
    }
    return streams;
}

} // namespace

int main()
{
    static_assert(kAttentionSize % kBlockSize == 0);

    Matrix column_stream_input{};
    for (std::size_t query = 0; query < kAttentionSize; ++query) {
        for (std::size_t key = 0; key < kAttentionSize; ++key) {
            column_stream_input[query][key] = value(query, key);
        }
    }

    // Sixteen key columns are aligned onto sixteen source streams. Within one
    // query superlane, stream is the key-local index and lane is query-local.
    // Transpose sg16 turns that into one 16-key row chunk per query-local stream.
    Chunks transposed_chunks{};
    UnitGroup transpose_unit;
    for (std::size_t query_block = 0; query_block < kBlocks; ++query_block) {
        for (std::size_t key_block = 0; key_block < kBlocks; ++key_block) {
            UnitGroup::StreamState input{};
            for (std::size_t key_local = 0; key_local < kBlockSize; ++key_local) {
                for (std::size_t query_local = 0; query_local < kBlockSize; ++query_local) {
                    const auto query = query_block * kBlockSize + query_local;
                    const auto key = key_block * kBlockSize + key_local;
                    input[2 * key_local][query_local] = UnitGroup::Word {
                        column_stream_input[query][key],
                        query_local + 1 == kBlockSize,
                    };
                    input[2 * key_local + 1][query_local] = UnitGroup::Word {
                        0,
                        query_local + 1 == kBlockSize,
                    };
                }
            }

            transpose_unit.issue(ftlpu::SxmInstruction::Transpose(
                stream_range(0, 2 * kBlockSize),
                stream_range(2 * kBlockSize, 2 * kBlockSize)));
            const auto transposed = transpose_unit.evaluate(input);
            transpose_unit.complete_cycle();
            for (std::size_t query_local = 0; query_local < kBlockSize; ++query_local) {
                const auto query = query_block * kBlockSize + query_local;
                for (std::size_t key_local = 0; key_local < kBlockSize; ++key_local) {
                    const auto& output =
                        transposed.outputs[2 * kBlockSize + 2 * query_local][key_local];
                    assert(output.has_value());
                    transposed_chunks[query][key_block][key_local] = output->data;
                }
            }
        }
    }

    // Keep all 32 keys. The staging buffer selects one physical vector per
    // invocation (two on LPU16); a permute source cannot span invocations.
    constexpr auto planeBlocks=ftlpu::Permute320::kTotalLanes/kBlockSize;
    static_assert(planeBlocks>0);
    Matrix row_stream_output{};
    for (std::size_t query = 0; query < kAttentionSize; ++query) {
        for(std::size_t first=0;first<kBlocks;first+=planeBlocks){
            const auto count=std::min(planeBlocks,kBlocks-first);
            Plane permute_input{};auto map=ftlpu::Permute320::identity_map();
            for(std::size_t k=0;k<count;k++){
                permute_input[count-1-k]=transposed_chunks[query][first+k];
                for(std::size_t n=0;n<kBlockSize;n++)map[k*kBlockSize+n]=(count-1-k)*kBlockSize+n;
            }
            const auto permuted=ftlpu::SxmSlice::permute(permute_input,map);
            for(std::size_t n=0;n<count*kBlockSize;n++)
                row_stream_output[query][first*kBlockSize+n]=permuted[n/kBlockSize][n%kBlockSize];
            for(std::size_t n=count*kBlockSize;n<ftlpu::Permute320::kTotalLanes;n++)
                assert(permuted[n/kBlockSize][n%kBlockSize]==0);
        }
    }

    for (std::size_t query = 0; query < kAttentionSize; ++query) {
        for (std::size_t key = 0; key < kAttentionSize; ++key) {
            assert(row_stream_output[query][key] == column_stream_input[query][key]);
        }
    }

    return 0;
}
