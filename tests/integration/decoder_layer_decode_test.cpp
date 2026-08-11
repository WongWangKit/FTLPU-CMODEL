#include "ftlpu/core/fp16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kPrefillLength = 128;
constexpr std::size_t kDecodePosition = kPrefillLength;
constexpr std::size_t kSequenceLength = kPrefillLength + 1;
constexpr std::size_t kHidden = 32;
constexpr std::size_t kIntermediate = 64;
constexpr float kEpsilon = 1.0e-5f;
const float kAttentionScale =
    1.0f / std::sqrt(static_cast<float>(kHidden));

constexpr std::array<std::size_t, 16> kWeightSlices {
    0, 1, 2, 3, 4, 5, 6, 7,
    8, 9, 10, 11, 12, 13, 14, 15,
};
constexpr std::array<std::size_t, 2> kColumnSlices {16, 17};
constexpr std::array<std::size_t, 2> kKeyCacheSlices {40, 41};
constexpr std::array<std::size_t, 2> kValueCacheSlices {42, 43};
constexpr std::array<std::size_t, 2> kActivationSlices {50, 51};

constexpr std::size_t kWeightAddressBase = 64;
constexpr std::size_t kColumnAddress = 96;
constexpr std::size_t kActivationAddress = 128;
constexpr std::size_t kAccumulatorAddress =
    ftlpu::hw::kMxmAccumulatorRows / 2;

using Vector = std::array<float, kHidden>;
using Matrix = std::array<Vector, kHidden>;
using IntermediateVector = std::array<float, kIntermediate>;

float fp16(float value)
{
    return ftlpu::Fp16::from_float(value).to_float();
}

std::uint16_t fp16_bits(float value)
{
    return ftlpu::Fp16::from_float(value).bits();
}

std::size_t east_read_to_mxm_latency(std::size_t slice)
{
    return ftlpu::hw::kMemGroups + 2
        - slice / ftlpu::hw::kMemSlicesPerGroup;
}

std::size_t mem_queue(std::size_t slice)
{
    return ftlpu::InstructionControlUnit::mem_queue(
        ftlpu::Hemisphere::East, slice);
}

class OfflineSchedule {
public:
    explicit OfflineSchedule(ftlpu::InstructionControlUnit& icu)
        : icu_(icu)
    {
    }

    void mem_at(
        std::size_t queue,
        std::size_t cycle,
        ftlpu::MemInstruction instruction)
    {
        pad(mem_[queue], cycle, [&](std::size_t count) {
            icu_.enqueue_mem_nop(queue, count);
        });
        icu_.enqueue_mem(queue, instruction);
        advance(mem_[queue], cycle + 1);
    }

    void mem_repeat_at(
        std::size_t queue,
        std::size_t cycle,
        ftlpu::MemInstruction instruction,
        std::size_t count,
        std::int64_t address_stride)
    {
        mem_at(queue, cycle, instruction);
        if (count > 1) {
            icu_.enqueue_mem_repeat(
                queue, count - 1, 1, address_stride);
        }
        advance(mem_[queue], cycle + count);
    }

    void mxm_load_at(
        std::size_t cycle,
        ftlpu::MxmControlInstruction instruction)
    {
        pad(mxm_load_, cycle, [&](std::size_t count) {
            icu_.enqueue_mxm_load_nop(0, count);
        });
        icu_.enqueue_mxm(0, instruction);
        advance(mxm_load_, cycle + 1);
    }

    void mxm_compute_at(
        std::size_t cycle,
        ftlpu::MxmControlInstruction instruction,
        std::size_t count = 1)
    {
        pad(mxm_compute_, cycle, [&](std::size_t gap) {
            icu_.enqueue_mxm_compute_nop(0, gap);
        });
        icu_.enqueue_mxm(0, instruction);
        if (count > 1) {
            icu_.enqueue_mxm_compute_repeat(0, count - 1, 1);
        }
        advance(mxm_compute_, cycle + count);
    }

    std::size_t end_cycle() const
    {
        return end_cycle_;
    }

private:
    template <typename Emit>
    static void pad(std::size_t cursor, std::size_t cycle, Emit emit)
    {
        if (cycle < cursor) {
            throw std::logic_error(
                "decode-layer offline schedule overlaps an ICU queue");
        }
        emit(cycle - cursor);
    }

    void advance(std::size_t& cursor, std::size_t next)
    {
        cursor = next;
        end_cycle_ = std::max(end_cycle_, next);
    }

    ftlpu::InstructionControlUnit& icu_;
    std::array<
        std::size_t,
        ftlpu::InstructionControlUnit::kMemQueues> mem_{};
    std::size_t mxm_load_{0};
    std::size_t mxm_compute_{0};
    std::size_t end_cycle_{0};
};

class DecodeHarness {
public:
    struct TraceEvent {
        std::size_t start{0};
        std::size_t end{0};
        std::string resource{};
        std::string detail{};
    };

    DecodeHarness()
    {
        initialize_prefill_cache();
    }

    Vector gemv(
        const Vector& input,
        const Matrix& weights,
        const std::string& label)
    {
        system_.reset_execution_state();
        initialize_weight_tile(weights);
        initialize_vector(kActivationSlices, kActivationAddress, input);

        OfflineSchedule schedule(system_.icu());
        constexpr std::size_t kLoadStart = 20;
        for (std::size_t block = 0;
             block < ftlpu::hw::kMxmSupercellsPerPlane;
             ++block) {
            const auto iw_cycle = kLoadStart + block;
            for (std::size_t stream = 0;
                 stream < ftlpu::hw::kMxmLoadStreamsPerCycle;
                 ++stream) {
                const auto slice = kWeightSlices[stream];
                schedule.mem_at(
                    mem_queue(slice),
                    iw_cycle - east_read_to_mxm_latency(slice),
                    ftlpu::MemInstruction::Read(
                        kWeightAddressBase + block,
                        ftlpu::StreamId::East(stream)));
            }
            schedule.mxm_load_at(
                iw_cycle,
                ftlpu::MxmControlInstruction::IWDirect16(0, block));
        }

        constexpr std::size_t kComputeCycle = kLoadStart + 10;
        for (std::size_t byte = 0;
             byte < kActivationSlices.size();
             ++byte) {
            const auto slice = kActivationSlices[byte];
            schedule.mem_at(
                mem_queue(slice),
                kComputeCycle - east_read_to_mxm_latency(slice),
                ftlpu::MemInstruction::Read(
                    kActivationAddress,
                    ftlpu::StreamId::East(byte)));
        }
        schedule.mxm_compute_at(
            kComputeCycle,
            ftlpu::MxmControlInstruction::Compute(
                0,
                0,
                0,
                kAccumulatorAddress,
                1,
                ftlpu::MxmAccumulatorDestination::Sram));

        const auto phase_start = total_cycles_;
        trace(
            phase_start + kLoadStart,
            phase_start + kLoadStart + ftlpu::hw::kMxmSupercellsPerPlane,
            "MXM.E0.Load",
            label + ": 4 x IW wide");
        trace(
            phase_start + kComputeCycle,
            phase_start + kComputeCycle + 1,
            "MXM.E0.Compute",
            label + ": one decode activation");
        run(schedule.end_cycle() + 16, label);

        Vector output{};
        for (std::size_t column = 0; column < kHidden; ++column) {
            output[column] =
                system_.mxm_unit(0).accumulator().value(
                    kAccumulatorAddress, column);
        }
        require_close(output, reference_gemv(input, weights), label);
        return output;
    }

    std::array<float, kSequenceLength> qk_scores(
        const Vector& query)
    {
        system_.reset_execution_state();
        initialize_weight_tile(Matrix{});
        initialize_vector(kColumnSlices, kColumnAddress, query);

        OfflineSchedule schedule(system_.icu());
        constexpr std::size_t kLoadStart = 20;
        for (std::size_t block = 0;
             block < ftlpu::hw::kMxmSupercellsPerPlane;
             ++block) {
            const auto iw_cycle = kLoadStart + block;
            for (std::size_t stream = 0;
                 stream < ftlpu::hw::kMxmLoadStreamsPerCycle;
                 ++stream) {
                const auto slice = kWeightSlices[stream];
                schedule.mem_at(
                    mem_queue(slice),
                    iw_cycle - east_read_to_mxm_latency(slice),
                    ftlpu::MemInstruction::Read(
                        kWeightAddressBase + block,
                        ftlpu::StreamId::East(stream)));
            }
            schedule.mxm_load_at(
                iw_cycle,
                ftlpu::MxmControlInstruction::IWDirect16(0, block));
        }

        constexpr std::size_t kColumnCycle = kLoadStart + 8;
        for (std::size_t byte = 0; byte < kColumnSlices.size(); ++byte) {
            const auto slice = kColumnSlices[byte];
            schedule.mem_at(
                mem_queue(slice),
                kColumnCycle - east_read_to_mxm_latency(slice),
                ftlpu::MemInstruction::Read(
                    kColumnAddress,
                    ftlpu::StreamId::East(byte)));
        }
        schedule.mxm_load_at(
            kColumnCycle,
            ftlpu::MxmControlInstruction::IWColumnDirect16(0, 0, 0));

        constexpr std::size_t kComputeStart = kColumnCycle + 8;
        for (std::size_t block_start = 0;
             block_start < kSequenceLength;
             block_start += ftlpu::hw::kMxmRows) {
            const auto count = std::min(
                ftlpu::hw::kMxmRows,
                kSequenceLength - block_start);
            const auto compute_cycle = kComputeStart + block_start;
            for (std::size_t byte = 0;
                 byte < kKeyCacheSlices.size();
                 ++byte) {
                const auto slice = kKeyCacheSlices[byte];
                schedule.mem_repeat_at(
                    mem_queue(slice),
                    compute_cycle - east_read_to_mxm_latency(slice),
                    ftlpu::MemInstruction::Read(
                        block_start,
                        ftlpu::StreamId::East(byte)),
                    count,
                    1);
            }
            schedule.mxm_compute_at(
                compute_cycle,
                ftlpu::MxmControlInstruction::Compute(
                    0,
                    0,
                    0,
                    kAccumulatorAddress + block_start,
                    1,
                    ftlpu::MxmAccumulatorDestination::Sram),
                count);
        }

        const auto phase_start = total_cycles_;
        trace(
            phase_start + kLoadStart,
            phase_start + kLoadStart + ftlpu::hw::kMxmSupercellsPerPlane,
            "MXM.E0.Load",
            "QK zero template: 4 x IW wide");
        trace(
            phase_start + kColumnCycle,
            phase_start + kColumnCycle + ftlpu::hw::kTileRows,
            "MXM.E0.Load",
            "decode Q: IWColumn inner=0 streams=2");
        trace(
            phase_start + kComputeStart,
            phase_start + kComputeStart + kSequenceLength,
            "MEM.E.Read",
            "stream K cache rows 0..128");
        trace(
            phase_start + kComputeStart,
            phase_start + kComputeStart + kSequenceLength,
            "MXM.E0.Compute",
            "QK decode: 129 cached K rows");
        run(schedule.end_cycle() + 16, "QK decode");

        std::array<float, kSequenceLength> scores{};
        for (std::size_t token = 0; token < kSequenceLength; ++token) {
            scores[token] =
                system_.mxm_unit(0).accumulator().value(
                    kAccumulatorAddress + token, 0);
            auto expected = 0.0f;
            for (std::size_t dimension = 0;
                 dimension < kHidden;
                 ++dimension) {
                expected += query[dimension]
                    * read_cache_value(
                        kKeyCacheSlices, token, dimension);
            }
            if (std::fabs(scores[token] - expected) > 1.0e-5f) {
                throw std::runtime_error(
                    "QK score mismatch at token " + std::to_string(token));
            }
        }
        return scores;
    }

    void append_cache(
        const Vector& key,
        const Vector& value)
    {
        initialize_vector(
            kKeyCacheSlices, kDecodePosition, key);
        initialize_vector(
            kValueCacheSlices, kDecodePosition, value);
        for (std::size_t dimension = 0;
             dimension < kHidden;
             ++dimension) {
            if (read_cache_value(
                    kKeyCacheSlices,
                    kDecodePosition,
                    dimension)
                    != key[dimension]
                || read_cache_value(
                    kValueCacheSlices,
                    kDecodePosition,
                    dimension)
                    != value[dimension]) {
                throw std::runtime_error(
                    "decode KV cache append mismatch");
            }
        }
    }

    Vector value_cache_row(std::size_t token) const
    {
        Vector value{};
        for (std::size_t dimension = 0;
             dimension < kHidden;
             ++dimension) {
            value[dimension] = read_cache_value(
                kValueCacheSlices, token, dimension);
        }
        return value;
    }

    static Vector reference_gemv(
        const Vector& input,
        const Matrix& weights)
    {
        Vector output{};
        for (std::size_t column = 0; column < kHidden; ++column) {
            for (std::size_t row = 0; row < kHidden; ++row) {
                output[column] += input[row] * weights[row][column];
            }
        }
        return output;
    }

private:
    void initialize_prefill_cache()
    {
        for (std::size_t token = 0;
             token < kPrefillLength;
             ++token) {
            Vector key{};
            Vector value{};
            for (std::size_t dimension = 0;
                 dimension < kHidden;
                 ++dimension) {
                key[dimension] = fp16(
                    static_cast<float>(
                        static_cast<int>((token * 5 + dimension * 3) % 29)
                        - 14)
                    * 0.03125f);
                value[dimension] = fp16(
                    static_cast<float>(
                        static_cast<int>((token * 7 + dimension * 11) % 31)
                        - 15)
                    * 0.025f);
            }
            initialize_vector(kKeyCacheSlices, token, key);
            initialize_vector(kValueCacheSlices, token, value);
        }
    }

    void initialize_weight_tile(const Matrix& weights)
    {
        for (std::size_t block = 0;
             block < ftlpu::hw::kMxmSupercellsPerPlane;
             ++block) {
            for (std::size_t tile = 0;
                 tile < ftlpu::hw::kTileRows;
                 ++tile) {
                for (std::size_t lane = 0;
                     lane < ftlpu::hw::kLanesPerTile;
                     ++lane) {
                    const auto row =
                        tile * ftlpu::hw::kLanesPerTile + lane;
                    for (std::size_t local_column = 0;
                         local_column < ftlpu::hw::kMxmSupercellColumns;
                         ++local_column) {
                        const auto column =
                            block * ftlpu::hw::kMxmSupercellColumns
                            + local_column;
                        const auto bits =
                            fp16_bits(weights[row][column]);
                        system_.initialize_mem_sram_lane_byte(
                            kWeightSlices[local_column * 2],
                            tile,
                            kWeightAddressBase + block,
                            lane,
                            static_cast<std::uint8_t>(bits & 0xffu));
                        system_.initialize_mem_sram_lane_byte(
                            kWeightSlices[local_column * 2 + 1],
                            tile,
                            kWeightAddressBase + block,
                            lane,
                            static_cast<std::uint8_t>(bits >> 8));
                    }
                }
            }
        }
    }

    void initialize_vector(
        const std::array<std::size_t, 2>& slices,
        std::size_t address,
        const Vector& values)
    {
        for (std::size_t tile = 0;
             tile < ftlpu::hw::kTileRows;
             ++tile) {
            for (std::size_t lane = 0;
                 lane < ftlpu::hw::kLanesPerTile;
                 ++lane) {
                const auto index =
                    tile * ftlpu::hw::kLanesPerTile + lane;
                const auto bits = fp16_bits(values[index]);
                system_.initialize_mem_sram_lane_byte(
                    slices[0],
                    tile,
                    address,
                    lane,
                    static_cast<std::uint8_t>(bits & 0xffu));
                system_.initialize_mem_sram_lane_byte(
                    slices[1],
                    tile,
                    address,
                    lane,
                    static_cast<std::uint8_t>(bits >> 8));
            }
        }
    }

    float read_cache_value(
        const std::array<std::size_t, 2>& slices,
        std::size_t token,
        std::size_t dimension) const
    {
        const auto tile = dimension / ftlpu::hw::kLanesPerTile;
        const auto lane = dimension % ftlpu::hw::kLanesPerTile;
        const auto low = system_.read_mem_sram_lane_byte(
            slices[0], tile, token, lane);
        const auto high = system_.read_mem_sram_lane_byte(
            slices[1], tile, token, lane);
        return ftlpu::Fp16::from_bits(
            static_cast<std::uint16_t>(low)
            | (static_cast<std::uint16_t>(high) << 8))
            .to_float();
    }

    void run(std::size_t cycles, const std::string& label)
    {
        for (std::size_t cycle = 0; cycle < cycles; ++cycle) {
            try {
                system_.tick({});
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    label + " cycle " + std::to_string(cycle)
                    + ": " + error.what());
            }
        }
        total_cycles_ += cycles;
    }

    static void require_close(
        const Vector& actual,
        const Vector& expected,
        const std::string& label)
    {
        for (std::size_t index = 0;
             index < actual.size();
             ++index) {
            if (std::fabs(actual[index] - expected[index]) > 1.0e-5f) {
                throw std::runtime_error(
                    label + " mismatch at " + std::to_string(index)
                    + " actual=" + std::to_string(actual[index])
                    + " expected=" + std::to_string(expected[index]));
            }
        }
    }

public:
    std::size_t total_cycles() const
    {
        return total_cycles_;
    }

    void mark_reference(const std::string& operation)
    {
        trace(
            total_cycles_,
            total_cycles_ + 1,
            "Reference." + operation,
            "functional composition; VXM timing covered by dedicated tests");
    }

    void write_trace_csv(const std::string& path) const
    {
        const auto output_path = std::filesystem::path(path);
        if (output_path.has_parent_path()) {
            std::filesystem::create_directories(
                output_path.parent_path());
        }
        auto output = std::ofstream(output_path, std::ios::trunc);
        if (!output) {
            throw std::runtime_error(
                "cannot open decoder schedule trace: " + path);
        }
        output << "start,end,resource,detail\n";
        for (const auto& event : trace_events_) {
            output << event.start << ',' << event.end << ','
                   << csv_field(event.resource) << ','
                   << csv_field(event.detail) << '\n';
        }
    }

private:
    static std::string csv_field(const std::string& value)
    {
        auto output = std::string{"\""};
        for (const auto ch : value) {
            if (ch == '"') {
                output += '"';
            }
            output += ch;
        }
        output += '"';
        return output;
    }

    void trace(
        std::size_t start,
        std::size_t end,
        std::string resource,
        std::string detail)
    {
        trace_events_.push_back(
            TraceEvent{
                start,
                end,
                std::move(resource),
                std::move(detail)});
    }

    ftlpu::TspSliceSystem system_{};
    std::size_t total_cycles_{0};
    std::vector<TraceEvent> trace_events_{};
};

Matrix make_weights(std::size_t tag)
{
    Matrix weights{};
    for (std::size_t row = 0; row < kHidden; ++row) {
        for (std::size_t column = 0; column < kHidden; ++column) {
            const auto raw = static_cast<int>(
                (tag * 17 + row * 7 + column * 11) % 41)
                - 20;
            weights[row][column] = fp16(
                static_cast<float>(raw) * 0.00625f);
        }
    }
    return weights;
}

Vector rmsnorm(const Vector& input, const Vector& gamma)
{
    auto sum = 0.0f;
    for (const auto value : input) {
        sum += value * value;
    }
    const auto inverse = 1.0f / std::sqrt(
        sum / static_cast<float>(kHidden) + kEpsilon);
    Vector output{};
    for (std::size_t index = 0; index < kHidden; ++index) {
        output[index] = fp16(input[index] * inverse * gamma[index]);
    }
    return output;
}

void apply_rope(Vector& vector, std::size_t position)
{
    const auto original = vector;
    for (std::size_t pair = 0; pair < kHidden / 2; ++pair) {
        const auto low = pair;
        const auto high = pair + kHidden / 2;
        const auto frequency = 1.0f / std::pow(
            100000.0f,
            static_cast<float>(2 * pair)
                / static_cast<float>(kHidden));
        const auto angle = static_cast<float>(position) * frequency;
        const auto cosine = fp16(std::cos(angle));
        const auto sine = fp16(std::sin(angle));
        vector[low] = fp16(
            original[low] * cosine - original[high] * sine);
        vector[high] = fp16(
            original[high] * cosine + original[low] * sine);
    }
}

std::array<float, kSequenceLength> softmax(
    const std::array<float, kSequenceLength>& scores)
{
    std::array<float, kSequenceLength> probabilities{};
    auto maximum = -std::numeric_limits<float>::infinity();
    for (const auto score : scores) {
        maximum = std::max(maximum, score * kAttentionScale);
    }
    auto sum = 0.0f;
    for (std::size_t token = 0; token < kSequenceLength; ++token) {
        probabilities[token] = std::exp(
            scores[token] * kAttentionScale - maximum);
        sum += probabilities[token];
    }
    for (auto& probability : probabilities) {
        probability = fp16(probability / sum);
    }
    return probabilities;
}

Vector add_and_cast(const Vector& lhs, const Vector& rhs)
{
    Vector output{};
    for (std::size_t index = 0; index < kHidden; ++index) {
        output[index] = fp16(lhs[index] + rhs[index]);
    }
    return output;
}

Vector cast_fp16(const Vector& input)
{
    Vector output{};
    std::transform(
        input.begin(),
        input.end(),
        output.begin(),
        [](float value) { return fp16(value); });
    return output;
}

} // namespace

int main() try
{
    DecodeHarness harness;
    harness.mark_reference("KVCache.Prefill128Ready");

    Vector input{};
    Vector attention_gamma{};
    Vector ffn_gamma{};
    for (std::size_t index = 0; index < kHidden; ++index) {
        input[index] = fp16(
            static_cast<float>(static_cast<int>(index % 13) - 6)
            * 0.0625f);
        attention_gamma[index] = fp16(
            0.875f + static_cast<float>(index % 5) * 0.015625f);
        ffn_gamma[index] = fp16(
            0.8125f + static_cast<float>(index % 7) * 0.015625f);
    }

    const auto wq = make_weights(1);
    const auto wk = make_weights(2);
    const auto wv = make_weights(3);
    const auto wo = make_weights(4);
    const std::array<Matrix, 2> wgate {
        make_weights(5), make_weights(6)};
    const std::array<Matrix, 2> wup {
        make_weights(7), make_weights(8)};
    const std::array<Matrix, 2> wdown {
        make_weights(9), make_weights(10)};

    harness.mark_reference("RMSNorm.Attention");
    const auto attention_input = rmsnorm(input, attention_gamma);
    auto query = harness.gemv(attention_input, wq, "Q projection");
    auto key = harness.gemv(attention_input, wk, "K projection");
    const auto value = cast_fp16(
        harness.gemv(attention_input, wv, "V projection"));
    harness.mark_reference("RoPE");
    apply_rope(query, kDecodePosition);
    apply_rope(key, kDecodePosition);
    harness.append_cache(key, value);
    harness.mark_reference("KVCache.AppendToken128");

    const auto scores = harness.qk_scores(query);
    harness.mark_reference("Softmax");
    const auto probabilities = softmax(scores);

    Vector context{};
    for (std::size_t block = 0;
         block < (kSequenceLength + kHidden - 1) / kHidden;
         ++block) {
        Vector probability_block{};
        Matrix value_block{};
        for (std::size_t row = 0; row < kHidden; ++row) {
            const auto token = block * kHidden + row;
            if (token >= kSequenceLength) {
                continue;
            }
            probability_block[row] = probabilities[token];
            value_block[row] = harness.value_cache_row(token);
        }
        const auto partial = harness.gemv(
            probability_block,
            value_block,
            "P x V block " + std::to_string(block));
        for (std::size_t dimension = 0;
             dimension < kHidden;
             ++dimension) {
            context[dimension] += partial[dimension];
        }
    }
    for (auto& value_out : context) {
        value_out = fp16(value_out);
    }

    const auto attention_output =
        harness.gemv(context, wo, "O projection");
    harness.mark_reference("Residual.Attention");
    const auto post_attention =
        add_and_cast(input, attention_output);
    harness.mark_reference("RMSNorm.FFN");
    const auto ffn_input = rmsnorm(post_attention, ffn_gamma);

    IntermediateVector gate{};
    IntermediateVector up{};
    for (std::size_t block = 0; block < 2; ++block) {
        const auto gate_block = harness.gemv(
            ffn_input, wgate[block],
            "Gate projection " + std::to_string(block));
        const auto up_block = harness.gemv(
            ffn_input, wup[block],
            "Up projection " + std::to_string(block));
        for (std::size_t column = 0; column < kHidden; ++column) {
            gate[block * kHidden + column] = gate_block[column];
            up[block * kHidden + column] = up_block[column];
        }
    }

    harness.mark_reference("SwiGLU");
    IntermediateVector swiglu{};
    for (std::size_t index = 0; index < kIntermediate; ++index) {
        const auto sigmoid = 1.0f / (1.0f + std::exp(-gate[index]));
        swiglu[index] = fp16(gate[index] * sigmoid * up[index]);
    }

    Vector down{};
    for (std::size_t block = 0; block < 2; ++block) {
        Vector activation{};
        for (std::size_t row = 0; row < kHidden; ++row) {
            activation[row] = swiglu[block * kHidden + row];
        }
        const auto partial = harness.gemv(
            activation,
            wdown[block],
            "Down projection " + std::to_string(block));
        for (std::size_t column = 0; column < kHidden; ++column) {
            down[column] += partial[column];
        }
    }
    for (auto& value_out : down) {
        value_out = fp16(value_out);
    }
    harness.mark_reference("Residual.FFN");
    const auto output = add_and_cast(post_attention, down);

    auto checksum = 0.0f;
    for (const auto value_out : output) {
        if (!std::isfinite(value_out)) {
            throw std::runtime_error(
                "decoder layer produced a non-finite output");
        }
        checksum += value_out;
    }

    auto trace_path = std::string{};
    if (const auto* configured =
            std::getenv("FTLPU_SCHEDULE_TRACE")) {
        trace_path = configured;
        harness.write_trace_csv(trace_path);
    }

    std::cout
        << "decoder layer decode passed: prefill_kv=128, decode_position=128"
        << ", hidden=32, head_dim=32, intermediate=64"
        << ", qk=IWColumn(2-stream)+129 K rows"
        << ", mapped_cycles=" << harness.total_cycles()
        << ", checksum=" << checksum;
    if (!trace_path.empty()) {
        std::cout << ", schedule_csv=" << trace_path;
    }
    std::cout << '\n';
    return 0;
}
catch (const std::exception& error)
{
    std::cerr << "decoder layer decode test failed: "
              << error.what() << '\n';
    return 1;
}
