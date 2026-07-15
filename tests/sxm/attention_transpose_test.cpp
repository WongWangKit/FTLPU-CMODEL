#include "ftlpu/sxm/slice.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace {

constexpr std::size_t kAttentionSize = 160;
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
                    input[key_local][query_local] = UnitGroup::Word {
                        column_stream_input[query][key],
                        query_local + 1 == kBlockSize,
                    };
                }
            }

            transpose_unit.issue(ftlpu::SxmInstruction::Transpose(
                stream_range(0, kBlockSize),
                stream_range(kBlockSize, kBlockSize)));
            const auto transposed = transpose_unit.evaluate(input);
            transpose_unit.complete_cycle();
            for (std::size_t query_local = 0; query_local < kBlockSize; ++query_local) {
                const auto query = query_block * kBlockSize + query_local;
                for (std::size_t key_local = 0; key_local < kBlockSize; ++key_local) {
                    const auto& output = transposed.outputs[kBlockSize + query_local][key_local];
                    assert(output.has_value());
                    transposed_chunks[query][key_block][key_local] = output->data;
                }
            }
        }
    }

    // Staging supplies the ten chunks for one query in reverse block order.
    // Permute map restores key order in lanes 0..159; lanes 160..319 are the
    // zero padding required by the 320-wide MXM activation input.
    auto map = ftlpu::Permute320::identity_map();
    for (std::size_t output = 0; output < kAttentionSize; ++output) {
        const auto key_block = output / kBlockSize;
        const auto key_local = output % kBlockSize;
        map[output] = (kBlocks - 1 - key_block) * kBlockSize + key_local;
    }

    Matrix row_stream_output{};
    for (std::size_t query = 0; query < kAttentionSize; ++query) {
        Plane permute_input{};
        for (std::size_t key_block = 0; key_block < kBlocks; ++key_block) {
            const auto input_block = kBlocks - 1 - key_block;
            permute_input[input_block] = transposed_chunks[query][key_block];
        }

        const auto permuted = ftlpu::SxmSlice::permute(permute_input, map);
        for (std::size_t key = 0; key < kAttentionSize; ++key) {
            row_stream_output[query][key] = permuted[key / kBlockSize][key % kBlockSize];
        }
        for (std::size_t lane = kAttentionSize; lane < ftlpu::Permute320::kTotalLanes; ++lane) {
            assert(permuted[lane / kBlockSize][lane % kBlockSize] == 0);
        }
    }

    for (std::size_t query = 0; query < kAttentionSize; ++query) {
        for (std::size_t key = 0; key < kAttentionSize; ++key) {
            assert(row_stream_output[query][key] == column_stream_input[query][key]);
        }
    }

    return 0;
}
