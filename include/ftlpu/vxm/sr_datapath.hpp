#pragma once

#include "ftlpu/vxm/sr_instruction.hpp"
#include "ftlpu/vxm/stream_fabric.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace ftlpu {

// Cycle model for the configured physical VALU columns. All inter-column state lives
// in VxmStreamFabric. Pipelines retain only FU latency metadata and local
// accumulator state; they do not create a second stream namespace.
class VxmSrDatapath {
public:
    static constexpr std::size_t kColumns = hw::kVxmAluCount;
    static constexpr std::size_t kTiles = hw::kTileRows;
    static constexpr std::size_t kLanes = hw::kLanesPerTile;
    using TileInstructions =
        std::array<std::optional<VxmSrInstruction>, kColumns>;
    using Instructions = std::array<TileInstructions, kTiles>;
    using ExecutionMask = std::array<std::array<bool, kColumns>, kTiles>;

    VxmSrDatapath()
        : special_alu_(std::make_shared<VxmSpecialAlu>())
    {}

    void reset()
    {
        fabric_.registers().reset();
        for (auto& tile : basic_pipelines_)
            for (auto& lane : tile)
                for (auto& pipeline : lane) pipeline.reset();
        for (auto& tile : special_pipelines_)
            for (auto& lane : tile)
                for (auto& pipeline : lane) pipeline.reset();
        for (auto& tile : accumulators_)
            for (auto& lane : tile)
                for (auto& value : lane) value.reset();
    }

    VxmStreamFabric& fabric() noexcept { return fabric_; }
    const VxmStreamFabric& fabric() const noexcept { return fabric_; }

    void configure_special_lut(VxmSpecialAluOpcode opcode,
        VxmLutConfig config, std::vector<VxmLutEntry> entries)
    {
        special_alu_->configure_lut(opcode, config, std::move(entries));
    }

    // Caller may stage ingress values after begin_cycle() and before execute().
    void begin_cycle() { fabric_.registers().begin_cycle(); }

    ExecutionMask execute(const Instructions& issued)
    {
        auto executed = ExecutionMask {};
        for (std::size_t tile = 0; tile < kTiles; ++tile) {
            for (std::size_t column = 0; column < kColumns; ++column) {
                if (!issued[tile][column]) continue;
                const auto& instruction = *issued[tile][column];
                if (!operands_ready(tile, column, instruction)) continue;
                for (std::size_t lane = 0; lane < kLanes; ++lane)
                    launch(tile, lane, column, instruction);
                consume_stream_operands(tile, column, instruction);
                executed[tile][column] = true;
            }
        }

        advance_pipelines();
        return executed;
    }

    void finish_cycle()
    {
        fabric_.registers().stage_linear_links();
        fabric_.registers().commit_cycle();
    }

    ExecutionMask tick(const Instructions& issued)
    {
        begin_cycle();
        const auto result = execute(issued);
        finish_cycle();
        return result;
    }

    bool idle() const
    {
        for (const auto& tile : basic_pipelines_)
            for (const auto& lane : tile)
                for (const auto& pipeline : lane)
                    if (!pipeline.empty()) return false;
        for (const auto& tile : special_pipelines_)
            for (const auto& lane : tile)
                for (const auto& pipeline : lane)
                    if (!pipeline.empty()) return false;
        return true;
    }

private:
    struct Metadata {
        std::size_t tile {0};
        std::size_t lane {0};
        std::size_t column {0};
        VxmSrInstruction instruction {};
    };
    using BasicPipeline = VxmAlu::Pipeline<Metadata>;
    using SpecialPipeline = VxmSpecialAlu::Pipeline<Metadata>;

    static bool is_stream(const VxmSrOperand& operand)
    {
        return operand.kind == VxmSrOperandKind::StreamFloat16
            || operand.kind == VxmSrOperandKind::StreamBFloat16
            || operand.kind == VxmSrOperandKind::StreamFloat32
            || operand.kind == VxmSrOperandKind::StreamInt8;
    }

    static bool uses_rhs(const VxmSrInstruction& instruction)
    {
        const auto* opcode = std::get_if<VxmAluOpcode>(&instruction.operation);
        return opcode && *opcode != VxmAluOpcode::Bypass
            && *opcode != VxmAluOpcode::Negate;
    }

    template <typename Fn>
    static void for_each_used_operand(
        const VxmSrInstruction& instruction, Fn&& fn)
    {
        fn(instruction.lhs);
        if (uses_rhs(instruction)) fn(instruction.rhs);
    }

    bool operands_ready(std::size_t tile, std::size_t column,
        const VxmSrInstruction& instruction) const
    {
        bool ready = true;
        for_each_used_operand(instruction, [&](const VxmSrOperand& operand) {
            if (!is_stream(operand)) return;
            const auto base = StreamId::from_packed(operand.sr);
            const auto width = operand.kind == VxmSrOperandKind::StreamFloat32
                ? std::size_t {4}
                : operand.kind == VxmSrOperandKind::StreamInt8
                ? std::size_t {1} : std::size_t {2};
            for (std::size_t byte = 0; byte < width; ++byte) {
                const auto stream = base.direction() == StreamDirection::East
                    ? StreamId::East(base.index() + byte)
                    : StreamId::West(base.index() + byte);
                if (!fabric_.registers().segment_valid(
                        VxmStreamFabric::operand_column(column, stream),
                        tile, stream))
                    ready = false;
            }
        });
        if (!ready) return false;
        for (std::size_t lane = 0; lane < kLanes; ++lane) {
            for_each_used_operand(instruction, [&](const VxmSrOperand& operand) {
                if (operand.kind == VxmSrOperandKind::Accumulator
                    && !instruction.accumulator_reset
                    && !accumulators_[tile][lane][column])
                    ready = false;
            });
            if (!ready) return false;
        }
        return true;
    }

    float read_operand(std::size_t tile, std::size_t lane,
        std::size_t column, const VxmSrOperand& operand) const
    {
        if (operand.kind == VxmSrOperandKind::Immediate)
            return operand.immediate;
        if (operand.kind == VxmSrOperandKind::Accumulator)
            return *accumulators_[tile][lane][column];

        const auto base = StreamId::from_packed(operand.sr);
        const auto sr_column =
            VxmStreamFabric::operand_column(column, base);
        if (operand.kind == VxmSrOperandKind::StreamInt8)
            return static_cast<float>(static_cast<std::int8_t>(
                fabric_.registers().cell(
                    sr_column, tile, lane, base).data));
        if (operand.kind == VxmSrOperandKind::StreamFloat32) {
            std::uint32_t bits = 0;
            for (std::size_t byte = 0; byte < 4; ++byte) {
                const auto stream = base.direction() == StreamDirection::East
                    ? StreamId::East(base.index() + byte)
                    : StreamId::West(base.index() + byte);
                bits |= static_cast<std::uint32_t>(fabric_.registers().cell(
                    sr_column, tile, lane, stream).data) << (8 * byte);
            }
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        }
        const auto high = base.direction() == StreamDirection::East
            ? StreamId::East(base.index() + 1)
            : StreamId::West(base.index() + 1);
        const auto bits = static_cast<std::uint16_t>(fabric_.registers().cell(
            sr_column, tile, lane, base).data)
            | (static_cast<std::uint16_t>(fabric_.registers().cell(
                   sr_column, tile, lane, high).data) << 8);
        return operand.kind == VxmSrOperandKind::StreamFloat16
            ? VxmDataFormat::fp16_bits_to_float(bits)
            : VxmDataFormat::bf16_bits_to_float(bits);
    }

    void launch(std::size_t tile, std::size_t lane, std::size_t column,
        const VxmSrInstruction& instruction)
    {
        if (instruction.accumulator_reset)
            accumulators_[tile][lane][column] = 0.0f;
        const auto lhs = read_operand(tile, lane, column, instruction.lhs);
        const auto rhs = uses_rhs(instruction)
            ? read_operand(tile, lane, column, instruction.rhs) : 0.0f;
        auto metadata = Metadata {tile, lane, column, instruction};
        if (const auto* special =
                std::get_if<VxmSpecialAluOpcode>(&instruction.operation)) {
            if (!basic_pipelines_[tile][lane][column].empty())
                throw std::logic_error(
                    "VXM column changed to LUT while Basic result is in flight");
            pending_special_[tile][lane][column] =
                SpecialPipeline::Request {*special, lhs, std::move(metadata)};
        } else {
            if (!special_pipelines_[tile][lane][column].empty())
                throw std::logic_error(
                    "VXM column changed to Basic while LUT result is in flight");
            auto request = BasicPipeline::Request {
                {std::get<VxmAluOpcode>(instruction.operation),
                 instruction.precision},
                lhs, rhs, std::move(metadata)};
            pending_basic_[tile][lane][column] = std::move(request);
        }
    }

    void consume_stream_operands(std::size_t tile, std::size_t column,
        const VxmSrInstruction& instruction)
    {
        for_each_used_operand(instruction, [&](const VxmSrOperand& operand) {
            if (!is_stream(operand)) return;
            const auto base = StreamId::from_packed(operand.sr);
            const auto width = operand.kind == VxmSrOperandKind::StreamFloat32
                ? std::size_t {4}
                : operand.kind == VxmSrOperandKind::StreamInt8
                ? std::size_t {1} : std::size_t {2};
            for (std::size_t byte = 0; byte < width; ++byte) {
                const auto stream = base.direction() == StreamDirection::East
                    ? StreamId::East(base.index() + byte)
                    : StreamId::West(base.index() + byte);
                fabric_.consume_operand(column, tile, stream);
            }
        });
    }

    void advance_pipelines()
    {
        for (std::size_t tile = 0; tile < kTiles; ++tile) {
            for (std::size_t lane = 0; lane < kLanes; ++lane) {
                for (std::size_t column = 0; column < kColumns; ++column) {
                    auto basic = basic_pipelines_[tile][lane][column].tick(
                        std::move(pending_basic_[tile][lane][column]));
                    auto special = special_pipelines_[tile][lane][column].tick(
                        *special_alu_,
                        std::move(pending_special_[tile][lane][column]));
                    pending_basic_[tile][lane][column].reset();
                    pending_special_[tile][lane][column].reset();
                    if (basic && special)
                        throw std::logic_error(
                            "VXM Basic and LUT results collided");
                    if (basic)
                        complete(basic->value, basic->metadata);
                    else if (special)
                        complete(special->value, special->metadata);
                }
            }
        }
    }

    void complete(float value, const Metadata& metadata)
    {
        const auto& instruction = metadata.instruction;
        if (instruction.accumulator_write)
            accumulators_[metadata.tile][metadata.lane][metadata.column] = value;
        if (instruction.accumulator_write && !instruction.accumulator_emit)
            return;

        const auto result = StreamId::from_packed(instruction.result_sr);
        const auto bytes = encode_result(value, instruction.result_type);
        for (std::size_t byte = 0; byte < bytes.count; ++byte) {
            const auto stream = result.direction() == StreamDirection::East
                ? StreamId::East(result.index() + byte)
                : StreamId::West(result.index() + byte);
            fabric_.registers().stage_write(
                VxmStreamFabric::result_column(metadata.column, stream),
                metadata.tile, metadata.lane, stream,
                StreamCell::Valid(bytes.values[byte],
                    metadata.lane + 1 == kLanes),
                "VXM VALU");
        }
    }

    struct EncodedResult {
        std::array<std::uint8_t, 4> values {};
        std::size_t count {0};
    };

    static EncodedResult encode_result(float value, VxmCastTarget type)
    {
        EncodedResult result {};
        if (type == VxmCastTarget::Int8) {
            result.values[0] = static_cast<std::uint8_t>(
                static_cast<std::int8_t>(std::clamp(
                    std::nearbyint(value), -128.0f, 127.0f)));
            result.count = 1;
        } else if (type == VxmCastTarget::Float16
                   || type == VxmCastTarget::BFloat16) {
            const auto bits = type == VxmCastTarget::Float16
                ? VxmDataFormat::float_to_fp16_bits(value)
                : VxmDataFormat::float_to_bf16_bits(value);
            result.values[0] = static_cast<std::uint8_t>(bits);
            result.values[1] = static_cast<std::uint8_t>(bits >> 8);
            result.count = 2;
        } else {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            for (std::size_t byte = 0; byte < 4; ++byte)
                result.values[byte] =
                    static_cast<std::uint8_t>(bits >> (8 * byte));
            result.count = 4;
        }
        return result;
    }

    VxmStreamFabric fabric_ {};
    std::shared_ptr<VxmSpecialAlu> special_alu_;
    std::array<std::array<std::array<BasicPipeline, kColumns>, kLanes>, kTiles>
        basic_pipelines_ {};
    std::array<std::array<std::array<SpecialPipeline, kColumns>, kLanes>, kTiles>
        special_pipelines_ {};
    std::array<std::array<std::array<
        std::optional<typename BasicPipeline::Request>, kColumns>, kLanes>, kTiles>
        pending_basic_ {};
    std::array<std::array<std::array<
        std::optional<typename SpecialPipeline::Request>, kColumns>, kLanes>, kTiles>
        pending_special_ {};
    std::array<std::array<std::array<std::optional<float>, kColumns>, kLanes>, kTiles>
        accumulators_ {};
};

} // namespace ftlpu
