#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/stream_port.hpp"
#include "ftlpu/mxm/mxm.hpp"
#include "ftlpu/mxm/output_cast.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace ftlpu {

// One fixed accumulator datapath is shared by the two MXMs in a hemisphere.
// ACC0/ACC1 are scalar adders replicated over all lanes.  Independent mode
// assigns one adder to each MXM.  Merge mode fixes ACC0 to MXM0+MXM1 and ACC1
// to ACC0+old.  Only short-lived operands/results live here; multi-context
// partial sums are explicitly spilled to ordinary MEM by compiler scheduling.
class MxmAccumulatorSlice {
public:
    using Value = Mxm::Accumulator;
    using Values = Mxm::ResultValues;

    struct Endpoint {
        std::size_t column;
        // Result/partial outputs move from MXM toward MEM/VXM. A compiler-
        // scheduled old partial returns from MEM on the opposite direction.
        StreamDirection output_direction;
    };

    struct Output {
        std::size_t row{0};
        std::size_t tile{0};
        std::size_t stream_base{0};
        bool final{false};
        Values values{};
        float dequant_scale{1.0f};
    };

    MxmAccumulatorSlice()
        : MxmAccumulatorSlice(Endpoint {0, StreamDirection::West})
    {
    }

    explicit MxmAccumulatorSlice(Endpoint endpoint)
        : endpoint_(endpoint)
    {
        reset();
    }

    void set_endpoint(Endpoint endpoint)
    {
        endpoint_ = endpoint;
    }

    const Endpoint& endpoint() const noexcept
    {
        return endpoint_;
    }

    // Compiler-managed static scale registers. Path 0/1 correspond to the
    // two fixed accumulator egress paths in this hemisphere. Merge output
    // uses path 1 after ACC0 combines the two MXMs.
    void configure_output_dequant_scale(
        std::size_t path,
        float activation_scale,
        float weight_scale)
    {
        if (path >= output_dequant_scales_.size()) {
            throw std::out_of_range(
                "MXM output scale path is outside the two fixed ACC paths");
        }
        output_dequant_scales_[path] =
            MxmOutputCast::combined_dequant_scale(
                activation_scale, weight_scale);
    }

    float output_dequant_scale(std::size_t path) const
    {
        if (path >= output_dequant_scales_.size()) {
            throw std::out_of_range(
                "MXM output scale path is outside the two fixed ACC paths");
        }
        return output_dequant_scales_[path];
    }

    void reset()
    {
        for (auto& path : local_values_) {
            for (auto& tile : path) {
                tile.fill(Value{});
            }
        }
        for (auto& path : local_rows_) {
            for (auto& row : path) {
                row.reset();
            }
        }
        pending_.clear();
        last_outputs_.clear();
        output_dequant_scales_.fill(1.0f);
    }

    void evaluate(
        StreamRegisterFabric& fabric,
        const std::vector<Mxm::ColumnOutput>& first,
        const std::vector<Mxm::ColumnOutput>& second = {})
    {
        if (!fabric.cycle_open()) {
            throw std::logic_error(
                "MXM accumulator evaluate requires an open SR cycle");
        }
        if (endpoint_.column >= fabric.column_count()) {
            throw std::out_of_range(
                "MXM accumulator endpoint is outside the SR fabric");
        }

        last_outputs_.clear();
        advance_and_emit(fabric);

        std::vector<bool> second_used(second.size(), false);
        for (const auto& output : first) {
            if (output.pair_mode == MxmPairMode::Independent) {
                accept_independent(fabric, 0, output);
                continue;
            }
            const auto match = find_match(output, second, second_used);
            second_used[match] = true;
            accept_merge(fabric, output, second[match]);
        }
        for (std::size_t index = 0; index < second.size(); ++index) {
            if (second_used[index]) {
                continue;
            }
            if (second[index].pair_mode == MxmPairMode::Merge) {
                throw std::logic_error(
                    "MXM merge result arrived without its fixed MXM0 partner");
            }
            accept_independent(fabric, 1, second[index]);
        }
    }

    const std::vector<Output>& last_outputs() const noexcept
    {
        return last_outputs_;
    }

    std::optional<std::size_t> local_row(
        std::size_t path, std::size_t tile) const
    {
        check_path_tile(path, tile);
        return local_rows_[path][tile];
    }

    Value local_value(
        std::size_t path,
        std::size_t tile,
        std::size_t lane) const
    {
        check_path_tile(path, tile);
        if (lane >= hw::kLanesPerTile) {
            throw std::out_of_range("MXM accumulator lane is outside the tile");
        }
        return local_values_[path][tile][lane];
    }

private:
    struct PendingOutput {
        std::size_t remaining_cycles{0};
        Output output{};
    };

    static void check_path_tile(std::size_t path, std::size_t tile)
    {
        if (path >= 2 || tile >= hw::kTileRows) {
            throw std::out_of_range(
                "MXM accumulator path or tile is outside the fixed datapath");
        }
    }

    static Value checked_add(Value lhs, Value rhs)
    {
        if constexpr (std::is_integral_v<Value>) {
            const auto wide = static_cast<std::int64_t>(lhs)
                + static_cast<std::int64_t>(rhs);
            if (wide < std::numeric_limits<std::int32_t>::min()
                || wide > std::numeric_limits<std::int32_t>::max()) {
                throw std::overflow_error(
                    "MXM int32 accumulator overflow; quantization/compiler range contract was violated");
            }
            return static_cast<Value>(wide);
        } else {
            return lhs + rhs;
        }
    }

    static std::size_t find_match(
        const Mxm::ColumnOutput& lhs,
        const std::vector<Mxm::ColumnOutput>& rhs,
        const std::vector<bool>& used)
    {
        for (std::size_t index = 0; index < rhs.size(); ++index) {
            if (!used[index]
                && rhs[index].row == lhs.row
                && rhs[index].column_block == lhs.column_block) {
                return index;
            }
        }
        throw std::logic_error(
            "MXM merge result arrived without its fixed MXM1 partner");
    }

    static void require_same_merge_control(
        const Mxm::ColumnOutput& lhs,
        const Mxm::ColumnOutput& rhs)
    {
        if (rhs.pair_mode != MxmPairMode::Merge
            || lhs.accumulator_mode != rhs.accumulator_mode
            || lhs.output_stream_base != rhs.output_stream_base
            || lhs.partial_stream_base != rhs.partial_stream_base) {
            throw std::logic_error(
                "paired MXMs reached the shared accumulator with different control");
        }
    }

    void accept_independent(
        StreamRegisterFabric& fabric,
        std::size_t path,
        const Mxm::ColumnOutput& input)
    {
        auto values = input.values;
        apply_old_operand(
            fabric, path, input.column_block, input.row,
            input.accumulator_mode, input.partial_stream_base,
            values);
        complete_operation(input, path, std::move(values), 1);
    }

    void accept_merge(
        StreamRegisterFabric& fabric,
        const Mxm::ColumnOutput& first,
        const Mxm::ColumnOutput& second)
    {
        require_same_merge_control(first, second);
        auto values = Values{};
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            values[lane] = checked_add(first.values[lane], second.values[lane]);
        }
        // The merged value uses ACC1/local path 1 for its optional old
        // operand. ACC0 is permanently the cross-MXM combine stage.
        apply_old_operand(
            fabric, 1, first.column_block, first.row,
            first.accumulator_mode, first.partial_stream_base,
            values);
        complete_operation(first, 1, std::move(values), 2);
    }

    void apply_old_operand(
        StreamRegisterFabric& fabric,
        std::size_t path,
        std::size_t tile,
        std::size_t row,
        MxmAccumulatorMode mode,
        std::size_t partial_stream_base,
        Values& values)
    {
        check_path_tile(path, tile);
        if (mode == MxmAccumulatorMode::LocalStart
            || mode == MxmAccumulatorMode::MemoryStart
            || mode == MxmAccumulatorMode::DirectFinal) {
            return;
        }
        if (mode == MxmAccumulatorMode::LocalAccumulate
            || mode == MxmAccumulatorMode::LocalFinalize) {
            if (!local_rows_[path][tile].has_value()
                || *local_rows_[path][tile] != row) {
                throw std::logic_error(
                    "MXM local accumulator does not contain the requested token row");
            }
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                values[lane] = checked_add(
                    values[lane], local_values_[path][tile][lane]);
            }
            return;
        }
        if (mode == MxmAccumulatorMode::MemoryAccumulate
            || mode == MxmAccumulatorMode::MemoryFinalize) {
            const auto old = consume_memory_partial(
                fabric, tile, partial_stream_base);
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                values[lane] = checked_add(values[lane], old[lane]);
            }
            return;
        }
        throw std::logic_error("unknown MXM accumulator mode");
    }

    void complete_operation(
        const Mxm::ColumnOutput& control,
        std::size_t local_path,
        Values values,
        std::size_t adder_latency)
    {
        const auto mode = control.accumulator_mode;
        if (mode == MxmAccumulatorMode::LocalStart
            || mode == MxmAccumulatorMode::LocalAccumulate) {
            local_values_[local_path][control.column_block] = values;
            local_rows_[local_path][control.column_block] = control.row;
            return;
        }
        if (mode == MxmAccumulatorMode::LocalFinalize) {
            local_rows_[local_path][control.column_block].reset();
        }

        const auto final = MxmControlInstruction::produces_final_output(mode);
        pending_.push_back(PendingOutput {
            adder_latency + (final ? MxmOutputCast::kLatency : 0),
            Output {
                control.row,
                control.column_block,
                control.output_stream_base,
                final,
                std::move(values),
                output_dequant_scales_[local_path]}});
    }

    Values consume_memory_partial(
        StreamRegisterFabric& fabric,
        std::size_t tile,
        std::size_t stream_base) const
    {
        StreamInputPort input(
            fabric,
            endpoint_.column,
            endpoint_.output_direction
                    == StreamDirection::East
                ? StreamDirection::West
                : StreamDirection::East,
            "MXM accumulator old partial");
        for (std::size_t byte = 0; byte < sizeof(std::int32_t); ++byte) {
            if (!input.segment_valid(tile, stream_base + byte)) {
                throw std::logic_error(
                    "MXM accumulator fired before the compiler-scheduled MEM partial arrived");
            }
        }
        std::array<StreamSegment16, sizeof(std::int32_t)> segments{};
        for (std::size_t byte = 0; byte < segments.size(); ++byte) {
            segments[byte] = input.consume_segment(tile, stream_base + byte);
        }
        auto result = Values{};
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            std::uint32_t raw = 0;
            for (std::size_t byte = 0; byte < segments.size(); ++byte) {
                raw |= static_cast<std::uint32_t>(segments[byte][lane].data)
                    << (8 * byte);
            }
            if constexpr (std::is_integral_v<Value>) {
                result[lane] = std::bit_cast<std::int32_t>(raw);
            } else {
                result[lane] = static_cast<Value>(
                    std::bit_cast<std::int32_t>(raw));
            }
        }
        return result;
    }

    void advance_and_emit(StreamRegisterFabric& fabric)
    {
        auto remaining = std::vector<PendingOutput>{};
        for (auto& pending : pending_) {
            if (pending.remaining_cycles > 0) {
                --pending.remaining_cycles;
            }
            if (pending.remaining_cycles == 0) {
                emit(fabric, pending.output);
            } else {
                remaining.push_back(std::move(pending));
            }
        }
        pending_ = std::move(remaining);
    }

    void emit(StreamRegisterFabric& fabric, const Output& output)
    {
        const auto byte_count = output.final
            ? MxmOutputCast::kOutputBytes
            : sizeof(std::int32_t);
        StreamOutputPort port(
            fabric, endpoint_.column, endpoint_.output_direction,
            output.final ? "MXM accumulator final" : "MXM accumulator partial");
        std::array<StreamSegment16, sizeof(std::int32_t)> segments{};
        const auto tag = UINT64_C(0x4143430000000000)
            | static_cast<std::uint64_t>(output.row);
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            if (output.final) {
                const auto bytes = MxmOutputCast::bytes(
                    output.values[lane], output.dequant_scale);
                for (std::size_t byte = 0; byte < bytes.size(); ++byte) {
                    segments[byte][lane] = StreamCell::Valid(
                        bytes[byte], lane + 1 == hw::kLanesPerTile, tag);
                }
            } else {
                const auto value = to_int32(output.values[lane]);
                const auto raw = std::bit_cast<std::uint32_t>(value);
                for (std::size_t byte = 0; byte < sizeof(raw); ++byte) {
                    segments[byte][lane] = StreamCell::Valid(
                        static_cast<std::uint8_t>(raw >> (8 * byte)),
                        lane + 1 == hw::kLanesPerTile, tag);
                }
            }
        }
        for (std::size_t byte = 0; byte < byte_count; ++byte) {
            port.write_segment(
                output.tile, output.stream_base + byte, segments[byte]);
        }
        last_outputs_.push_back(output);
    }

    static std::int32_t to_int32(Value value)
    {
        if constexpr (std::is_integral_v<Value>) {
            return static_cast<std::int32_t>(value);
        } else {
            if (value < static_cast<Value>(std::numeric_limits<std::int32_t>::min())
                || value > static_cast<Value>(std::numeric_limits<std::int32_t>::max())) {
                throw std::overflow_error(
                    "MXM partial cannot be represented as int32 for MEM spill");
            }
            return static_cast<std::int32_t>(value);
        }
    }

    Endpoint endpoint_{0, StreamDirection::West};
    std::array<
        std::array<Values, hw::kTileRows>, 2> local_values_{};
    std::array<
        std::array<std::optional<std::size_t>, hw::kTileRows>, 2> local_rows_{};
    std::vector<PendingOutput> pending_{};
    std::vector<Output> last_outputs_{};
    std::array<float, 2> output_dequant_scales_ {1.0f, 1.0f};
};

} // namespace ftlpu
