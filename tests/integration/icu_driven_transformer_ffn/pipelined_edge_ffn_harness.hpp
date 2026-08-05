#pragma once

#include "full_ffn_report.hpp"
#include "ffn_icu_program.hpp"

#include "ftlpu/core/fp16.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include "ftlpu/vxm/mxm_input_buffer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace full_ffn_test {

using namespace ftlpu;

// Hardware-scaled edge-Transformer FFN.  The tensor ratios and complete
// producer/consumer pipeline are preserved while the TransformerEvalConfig
// keeps the regression small enough to emit a cycle trace.
class PipelinedEdgeFfnHarness {
public:
    static constexpr auto kChainDepth = VxmChainDepth::Eight;
    static constexpr std::size_t kChainsPerLane =
        VxmLane::kAluCount / static_cast<std::size_t>(kChainDepth);
    static constexpr std::size_t kTokenGroups = 4;
    static constexpr std::size_t kTokensPerRound =
        hw::kTileRows * hw::kLanesPerTile;
    static constexpr std::size_t kTokensPerGroup =
        kTokensPerRound * kChainsPerLane;
    static constexpr std::size_t kTokens =
        kTokenGroups * kTokensPerGroup;
    static constexpr std::size_t kHidden = hw::kMxmK;
    static constexpr std::size_t kIntermediate = 2 * hw::kMxmColumns;
    static constexpr std::size_t kOutputFeatures = hw::kMxmColumns;
    static constexpr std::size_t kFeatureBlocks =
        kIntermediate / hw::kMxmColumns;
    static constexpr float kQuantScale = 1.0f;
    static constexpr std::size_t kDirectSrLatency =
        hw::kMemBoundaryStreamRegisterColumns - 1;

    PipelinedEdgeFfnHarness()
        : system_(std::make_unique<TspSliceSystem>())
        , direct_input_(kIntermediate, kChainDepth, 2)
    {
        initialize();
    }

    explicit PipelinedEdgeFfnHarness(
        const icu_ffn_test::FfnIcuProgram& program)
        : system_(std::make_unique<TspSliceSystem>())
        , direct_input_(kIntermediate, kChainDepth, 2)
        , control_mode_(ControlMode::IcuReplay)
        , icu_program_(program)
    {
        initialize();
        icu_program_.load_into(system_->icu());
    }

    const icu_ffn_test::FfnIcuProgram& compiled_icu_program() const noexcept
    {
        return icu_program_;
    }

    bool icu_driven() const noexcept
    {
        return control_mode_ == ControlMode::IcuReplay;
    }

private:
    enum class ControlMode {
        DirectRecord,
        IcuReplay,
    };

    void initialize()
    {
        static_assert(hw::kTileRows == 4 && hw::kLanesPerTile == 8);
        static_assert(hw::kMxmRows == kTokensPerRound);
        static_assert(hw::kMxmCount == 4);
        static_assert(hw::kWestMxmCount == 2 && hw::kEastMxmCount == 2);
        static_assert(kIntermediate % hw::kMxmColumns == 0);
        static_assert(kChainsPerLane == 2);
        system_->set_vxm_stream_capture_enabled(false);
        configure_vxm();
        configure_scales();
        build_static_schedule();
    }

public:

    FullFfnResult run()
    {
        constexpr auto kSafetyCycles = std::size_t{8000};
        while (!complete()) {
            if (cycle_ >= kSafetyCycles) {
                throw std::runtime_error(
                    "pipelined edge FFN did not drain before its cycle limit");
            }
            tick();
        }
        verify_final_memory();
        return finish_result();
    }

private:
    static constexpr std::size_t kQuantInputAddress = 256;
    static constexpr std::size_t kDownInputAddress = 512;
    static constexpr std::size_t kFinalAddress = 1024;
    static constexpr std::array<std::size_t, kChainsPerLane>
        kRawQuantSliceBases {0, 8};
    static constexpr std::array<std::size_t, 2>
        kDownHalfSliceBases {16, 24};

    struct FrontCompute {
        std::size_t issue_cycle{0};
        std::size_t group{0};
        std::size_t feature_block{0};
        std::size_t chain{0};
        std::size_t row{0};
        std::size_t buffer{0};
    };

    struct DownCompute {
        std::size_t issue_cycle{0};
        std::size_t group{0};
        std::size_t chain{0};
        std::size_t row{0};
    };

    struct FrontTag {
        std::size_t group{0};
        std::size_t feature_block{0};
        std::size_t chain{0};
        std::size_t row{0};
    };

    struct DownTag {
        std::size_t group{0};
        std::size_t chain{0};
        std::size_t row{0};
    };

    struct LoadWave {
        std::size_t start{0};
        std::size_t pulses{0};
        std::size_t buffer{0};
        bool scales{false};
        bool background{false};
        std::array<bool, hw::kMxmCount> targets{};
    };

    struct MemAction {
        std::size_t slice{0};
        MemInstruction instruction{};
    };

    struct SxmAction {
        SxmInstruction instruction{};
    };

    struct VxmProgramAction {
        std::size_t group{0};
    };

    struct VxmFeedAction {
        std::size_t group{0};
        std::size_t feature{0};
        std::size_t tile{0};
        bool release{false};
    };

    struct DequantAction {
        std::size_t group{0};
        std::size_t chain{0};
        std::size_t row{0};
    };

    static std::array<bool, hw::kMxmCount> all_mxms()
    {
        std::array<bool, hw::kMxmCount> result{};
        result.fill(true);
        return result;
    }

    static std::array<bool, hw::kMxmCount> west_mxms()
    {
        std::array<bool, hw::kMxmCount> result{};
        for (std::size_t mxm = 0; mxm < hw::kWestMxmCount; ++mxm) {
            result[mxm] = true;
        }
        return result;
    }

    static float input_value(std::size_t token)
    {
        return 1.0f + 0.25f * static_cast<float>(token % 3);
    }

    static std::int8_t expected_quantized(std::size_t token)
    {
        const auto x = input_value(token);
        const auto swiglu = x / (1.0f + std::exp(-x)) * (2.0f * x);
        return VxmDataFormat::quantize_int8(
            VxmDataFormat::round_fp16_ftz(swiglu), kQuantScale);
    }

    static SxmInstruction::StreamList streams(
        StreamDirection direction, std::size_t first, std::size_t count)
    {
        auto result = SxmInstruction::StreamList{};
        for (std::size_t stream = first; stream < first + count; ++stream) {
            result.push_back(SxmStreamId {
                direction == StreamDirection::East
                    ? StreamId::East(stream).packed()
                    : StreamId::West(stream).packed()});
        }
        return result;
    }

    static SxmInstruction::PermuteMap wavefront_map(std::size_t wave)
    {
        auto map = Permute320::identity_map();
        for (std::size_t destination = 0;
             destination < hw::kTileRows; ++destination) {
            const auto source = (wave + hw::kTileRows - destination)
                % hw::kTileRows;
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                map[destination * hw::kLanesPerTile + lane] =
                    source * hw::kLanesPerTile + lane;
            }
        }
        return map;
    }

    template <class Function>
    static std::vector<VxmLutEntry> make_lut(
        float input_min, float width, std::size_t count, Function fn)
    {
        auto entries = std::vector<VxmLutEntry>{};
        entries.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            const auto x0 = input_min + static_cast<float>(index) * width;
            const auto y0 = fn(x0);
            entries.push_back(VxmLutEntry::from_float(
                (fn(x0 + width) - y0) / width, y0));
        }
        return entries;
    }

    static VxmLaneAluInstruction basic(
        VxmAluOpcode opcode,
        VxmLaneOperand lhs,
        VxmLaneOperand rhs,
        std::size_t repeat)
    {
        auto result = VxmLaneAluInstruction{};
        result.operation = opcode;
        result.lhs = lhs;
        result.rhs = rhs;
        result.repeat_count = repeat;
        return result;
    }

    static VxmLaneAluInstruction special(
        VxmSpecialAluOpcode opcode,
        VxmLaneOperand input,
        std::size_t repeat)
    {
        auto result = VxmLaneAluInstruction{};
        result.operation = opcode;
        result.lhs = input;
        result.rhs = VxmLaneOperand::Imm(0.0f);
        result.repeat_count = repeat;
        return result;
    }

    void configure_vxm()
    {
        constexpr auto kEntries = std::size_t{256};
        constexpr auto kLn2 = 0.6931471805599453f;
        auto& vxm = system_->vxm();
        vxm.set_chain_depth(kChainDepth);
        vxm.configure_special_lut(
            VxmSpecialAluOpcode::Exp,
            {-kLn2 / 2.0f, kLn2 / static_cast<float>(kEntries)},
            make_lut(-kLn2 / 2.0f,
                kLn2 / static_cast<float>(kEntries), kEntries,
                [](float x) { return std::exp(x); }));
        vxm.configure_special_lut(
            VxmSpecialAluOpcode::Reciprocal,
            {1.0f, 1.0f / static_cast<float>(kEntries)},
            make_lut(1.0f, 1.0f / static_cast<float>(kEntries),
                kEntries, [](float x) { return 1.0f / x; }));

        // Both token chains are assembled by the lane-banked direct Buffer.
        // Quantized tails leave toward East MEM/SXM/Down-MXM.
        for (const auto group : {0U, 1U, 8U, 9U}) {
            vxm.configure_input_group_source(group, Hemisphere::West);
        }
        vxm.configure_output_block_destination(3, Hemisphere::East);
        vxm.configure_output_block_destination(7, Hemisphere::East);
    }

    void configure_scales()
    {
        // W8A16 scales are loaded through the modeled scale path below.
        // ACC cast itself applies no additional scale.
        for (std::size_t mxm = 0; mxm < hw::kMxmCount; ++mxm) {
            system_->configure_mxm_output_dequant_scale(mxm, 1.0f, 1.0f);
            system_->mxm_activation_dequantizer(mxm)
                .configure_scale(kQuantScale);
        }
    }

    void control_mxm(
        std::size_t mxm, MxmControlInstruction instruction)
    {
        if (control_mode_ == ControlMode::DirectRecord) {
            icu_program_.record_mxm(cycle_, mxm, instruction);
            if (instruction.opcode
                == MxmControlOpcode::ActivationDequantize) {
                system_->mxm_activation_dequantizer(mxm).issue_south();
            } else {
                system_->mxm_unit(mxm).control().issue_south(instruction);
            }
        }
    }

    void control_mem(
        std::size_t slice, MemInstruction instruction)
    {
        if (control_mode_ == ControlMode::DirectRecord) {
            icu_program_.record_mem(
                cycle_, Hemisphere::East, slice, instruction);
            system_->mem(Hemisphere::East).memory_model()
                .enqueue_instruction(slice, std::move(instruction));
        }
    }

    void control_sxm(SxmInstruction instruction)
    {
        if (control_mode_ == ControlMode::DirectRecord) {
            icu_program_.record_sxm(
                cycle_, Hemisphere::East, instruction);
            system_->sxm(Hemisphere::East).issue(std::move(instruction));
        }
    }

    void control_vxm(
        std::size_t stage, VxmCompactInstruction instruction)
    {
        if (control_mode_ == ControlMode::DirectRecord) {
            icu_program_.record_vxm(cycle_, stage, instruction);
            system_->vxm().issue_south(stage, instruction);
        }
    }

    void enqueue_vxm_program()
    {
        constexpr auto repeat = kIntermediate;
        for (std::size_t stage = 0; stage < 8; ++stage) {
            auto instruction = VxmLaneAluInstruction{};
            switch (stage) {
            case 0:
                instruction = basic(
                    VxmAluOpcode::Negate,
                    VxmLaneOperand::StreamFloat16(),
                    VxmLaneOperand::StreamFloat16(), repeat);
                break;
            case 1:
                instruction = special(
                    VxmSpecialAluOpcode::Exp,
                    VxmLaneOperand::Previous(), repeat);
                break;
            case 2:
                instruction = basic(
                    VxmAluOpcode::Add, VxmLaneOperand::Previous(),
                    VxmLaneOperand::Imm(1.0f), repeat);
                break;
            case 3:
                instruction = special(
                    VxmSpecialAluOpcode::Reciprocal,
                    VxmLaneOperand::Previous(), repeat);
                break;
            case 4:
                instruction = basic(
                    VxmAluOpcode::Multiply,
                    VxmLaneOperand::Previous(),
                    VxmLaneOperand::Original(), repeat);
                break;
            case 5:
                instruction = basic(
                    VxmAluOpcode::Multiply,
                    VxmLaneOperand::Previous(),
                    VxmLaneOperand::Aux(), repeat);
                break;
            case 6:
            case 7:
                instruction = basic(
                    VxmAluOpcode::Bypass,
                    VxmLaneOperand::Previous(),
                    VxmLaneOperand::Imm(0.0f), repeat);
                if (stage == 7) {
                    instruction.output_stream =
                        VxmLane::fixed_output_stream_for_block(3);
                    instruction.output_type = VxmCastTarget::Int8;
                    instruction.output_scale = kQuantScale;
                }
                break;
            }
            control_vxm(
                stage, VxmCompactInstructionCodec::encode(
                    stage, kChainDepth, instruction));
        }
    }

    void build_static_schedule()
    {
        load_waves_.push_back({
            0, hw::kMxmSupercellsPerPlane, 0,
            true, false, all_mxms()});
        load_waves_.push_back({
            4, hw::kMxmSupercellsPerPlane, 1,
            true, false, west_mxms()});
        load_waves_.push_back({
            8, hw::kMxmSupercellsPerPlane, 0,
            false, false, all_mxms()});

        front_compute_start_ = 18;
        load_waves_.push_back({
            front_compute_start_,
            2 * hw::kMxmSupercellsPerPlane,
            1, false, true, west_mxms()});

        auto cursor = front_compute_start_;
        for (std::size_t group = 0; group < kTokenGroups; ++group) {
            for (std::size_t feature_block = 0;
                 feature_block < kFeatureBlocks; ++feature_block) {
                for (std::size_t chain = 0;
                     chain < kChainsPerLane; ++chain) {
                    for (std::size_t row = 0;
                         row < hw::kMxmRows; ++row) {
                        front_compute_.push_back({
                            cursor++, group, feature_block, chain, row,
                            feature_block & 1U});
                    }
                }
            }
        }
        front_last_issue_ = cursor - 1;
    }

    void schedule_mem(
        std::size_t cycle, std::size_t slice,
        MemInstruction instruction)
    {
        mem_actions_.emplace(
            cycle, MemAction{slice, std::move(instruction)});
    }

    void schedule_sxm(
        std::size_t cycle, SxmInstruction instruction)
    {
        sxm_actions_.emplace(
            cycle, SxmAction{std::move(instruction)});
    }

    std::size_t schedule_east_transpose(
        std::size_t earliest_capture,
        std::size_t input_slice_base,
        std::size_t output_slice_base,
        std::size_t input_address,
        std::size_t output_address)
    {
        const auto capture = std::max(
            earliest_capture, cycle_ + std::size_t{13});
        constexpr auto stream_count = hw::kLanesPerTile;
        auto last_mem_write = std::size_t{0};
        for (std::size_t stream = 0; stream < stream_count; ++stream) {
            const auto input_slice = input_slice_base + stream;
            const auto output_slice = output_slice_base + stream;
            const auto input_group = input_slice / hw::kMemSlicesPerGroup;
            const auto output_group = output_slice / hw::kMemSlicesPerGroup;
            const auto read_start = capture - 1 - input_group;
            const auto write_start = capture + 2 + output_group;
            for (std::size_t beat = 0; beat < hw::kTileRows; ++beat) {
                schedule_mem(
                    read_start + beat, input_slice,
                    MemInstruction::Read(
                        input_address + beat,
                        StreamId::West(stream)));
                schedule_mem(
                    write_start + beat, output_slice,
                    MemInstruction::Write(
                        output_address + beat,
                        StreamId::East(24 + stream)));
                last_mem_write = std::max(
                    last_mem_write, write_start + beat);
            }
        }

        const auto src = streams(StreamDirection::West, 0, stream_count);
        const auto internal = streams(
            StreamDirection::West, stream_count, stream_count);
        const auto dst = streams(StreamDirection::East, 24, stream_count);
        for (std::size_t wave = 0; wave < hw::kTileRows; ++wave) {
            schedule_sxm(capture + wave,
                SxmInstruction::Transpose(src, internal, 1));
        }
        for (std::size_t wave = 0;
             wave < 2 * hw::kTileRows - 1; ++wave) {
            schedule_sxm(capture + wave + 1,
                SxmInstruction::Permute(
                    internal, dst, wavefront_map(wave),
                    SxmWeightLayout::VectorColumns, 1));
        }
        const auto last_sxm_issue =
            capture + 2 * hw::kTileRows - 1;
        // The next transpose may start once this bank's last instruction has
        // issued and the last permuted beat has reached MEM.  The former
        // fixed +16 guard serialized otherwise independent token groups and
        // created artificial Down-MXM bubbles.
        return std::max(last_sxm_issue, last_mem_write) + 1;
    }

    void schedule_vxm_group(std::size_t group)
    {
        const auto start = std::max({
            *front_ready_cycle_[group],
            vxm_south_available_,
            cycle_ + std::size_t{1}});
        // Compact instructions enter the Superlane controller one cycle
        // before their first stream-consuming configuration issues.
        vxm_program_actions_.emplace(start - 1, VxmProgramAction{group});
        for (std::size_t feature = 0;
             feature < kIntermediate; ++feature) {
            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
                vxm_feed_actions_.emplace(
                    start + feature + tile,
                    VxmFeedAction{
                        group, feature, tile,
                        feature + 1 == kIntermediate
                            && tile + 1 == hw::kTileRows});
            }
        }
        vxm_south_available_ = start + kIntermediate;
        ++front_groups_scheduled_;
    }

    std::size_t schedule_down_chain(
        std::size_t group,
        std::size_t chain,
        std::size_t earliest_dequant)
    {
        const auto dequant_start = std::max(
            down_available_, earliest_dequant);
        const auto compute_start = dequant_start + 1;
        const auto address = kDownInputAddress
            + group * (kChainsPerLane * hw::kTileRows)
            + chain * hw::kTileRows;
        for (std::size_t row = 0; row < hw::kMxmRows; ++row) {
            const auto dequant_cycle = dequant_start + row;
            for (std::size_t half = 0; half < 2; ++half) {
                const auto slice = kDownHalfSliceBases[half]
                    + row % hw::kLanesPerTile;
                const auto slice_group =
                    slice / hw::kMemSlicesPerGroup;
                const auto input_stream =
                    MxmActivationDequantizer::kInputStreamWithinWindow
                    + half * hw::kMxmLoadStreamsPerCycle;
                schedule_mem(
                    dequant_cycle - 11 + slice_group,
                    slice,
                    MemInstruction::Read(
                        address + row / hw::kLanesPerTile,
                        StreamId::East(input_stream)));
            }
            dequant_actions_.emplace(
                dequant_cycle,
                DequantAction{group, chain, row});
            down_compute_.push_back({
                compute_start + row, group, chain, row});
        }
        down_available_ = compute_start + hw::kMxmRows;
        return down_available_;
    }

    void schedule_quantized_group(std::size_t group)
    {
        auto cursor = std::max(
            sxm_available_, *quant_ready_cycle_[group]);
        for (std::size_t chain = 0;
             chain < kChainsPerLane; ++chain) {
            const auto raw_address = kQuantInputAddress
                + group * (kChainsPerLane * hw::kTileRows);
            const auto down_address = kDownInputAddress
                + group * (kChainsPerLane * hw::kTileRows)
                + chain * hw::kTileRows;
            cursor = schedule_east_transpose(
                cursor,
                kRawQuantSliceBases[chain],
                kDownHalfSliceBases[0],
                raw_address,
                down_address);
            cursor = schedule_east_transpose(
                cursor,
                kRawQuantSliceBases[chain],
                kDownHalfSliceBases[1],
                raw_address + hw::kTileRows,
                down_address);
            schedule_down_chain(group, chain, cursor + 13);
        }
        sxm_available_ = cursor;
        ++quant_groups_scheduled_;
    }

    void issue_loads()
    {
        front_load_tiles_this_cycle_ = 0;
        down_load_tiles_this_cycle_ = 0;
        for (const auto& wave : load_waves_) {
            if (cycle_ >= wave.start && cycle_ < wave.start + wave.pulses) {
                const auto pulse = cycle_ - wave.start;
                auto mode = MxmWeightLoadMode::Full;
                if (wave.background) {
                    mode = pulse < hw::kMxmSupercellsPerPlane
                        ? MxmWeightLoadMode::BackgroundLowerHalf
                        : MxmWeightLoadMode::BackgroundUpperHalf;
                }
                for (std::size_t mxm = 0; mxm < hw::kMxmCount; ++mxm) {
                    if (!wave.targets[mxm]) continue;
                    control_mxm(
                        mxm,
                        wave.scales
                            ? MxmControlInstruction::LoadScales(wave.buffer)
                            : MxmControlInstruction::IW(wave.buffer, mode));
                }
            }

            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
                if (cycle_ < wave.start + tile) continue;
                const auto pulse = cycle_ - wave.start - tile;
                if (pulse >= wave.pulses) continue;
                for (std::size_t mxm = 0; mxm < hw::kMxmCount; ++mxm) {
                    if (!wave.targets[mxm]) continue;
                    stage_weight_segment(wave, tile, mxm);
                    if (mxm < hw::kWestMxmCount) {
                        ++front_load_tiles_this_cycle_;
                    } else {
                        ++down_load_tiles_this_cycle_;
                    }
                }
            }
        }
    }

    void stage_weight_segment(
        const LoadWave& wave,
        std::size_t tile,
        std::size_t mxm)
    {
        const auto side = mxm < hw::kWestMxmCount
            ? Hemisphere::West : Hemisphere::East;
        const auto local = mxm < hw::kWestMxmCount
            ? mxm : mxm - hw::kWestMxmCount;
        auto& fabric = system_->mem(side).stream_fabric();
        const auto column = side == Hemisphere::West
            ? std::size_t{0}
            : hw::kMemBoundaryStreamRegisterColumns - 1;
        const auto direction = side == Hemisphere::West
            ? StreamDirection::West : StreamDirection::East;
        const auto base = local * hw::kMxmLoadStreamsPerCycle;

        if (wave.scales) {
            const auto scale = mxm < hw::kWestMxmCount
                ? 1.0f / static_cast<float>(kHidden)
                : 1.0f / static_cast<float>(kIntermediate);
            const auto bits = Fp16::from_float(scale).bits();
            for (std::size_t stream = 0;
                 stream < hw::kMxmWeightScaleStreams; ++stream) {
                for (std::size_t lane = 0;
                     lane < hw::kLanesPerTile; ++lane) {
                    fabric.initialize_cell(
                        column, tile, lane,
                        direction == StreamDirection::East
                            ? StreamId::East(base + stream)
                            : StreamId::West(base + stream),
                        StreamCell::Valid(
                            static_cast<std::uint8_t>(
                                stream % 2 == 0 ? bits : bits >> 8),
                            lane + 1 == hw::kLanesPerTile));
                }
            }
            return;
        }

        const auto first = wave.background
            ? hw::kMxmLoadStreamsPerCycle
                - hw::kMxmBackgroundWeightLoadStreams
            : std::size_t{0};
        const auto count = wave.background
            ? hw::kMxmBackgroundWeightLoadStreams
            : hw::kMxmStoredWeightLoadStreams;
        const auto weight = static_cast<std::uint8_t>(mxm == 1 ? 2 : 1);
        for (std::size_t stream = 0; stream < count; ++stream) {
            for (std::size_t lane = 0;
                 lane < hw::kLanesPerTile; ++lane) {
                fabric.initialize_cell(
                    column, tile, lane,
                    direction == StreamDirection::East
                        ? StreamId::East(base + first + stream)
                        : StreamId::West(base + first + stream),
                    StreamCell::Valid(
                        weight, lane + 1 == hw::kLanesPerTile));
            }
        }
    }

    void issue_front_compute()
    {
        for (const auto& event : front_compute_) {
            if (event.issue_cycle != cycle_) continue;
            for (std::size_t mxm = 0; mxm < hw::kWestMxmCount; ++mxm) {
                const auto stream_base =
                    mxm * hw::kMxmLoadStreamsPerCycle;
                control_mxm(
                    mxm,
                    MxmControlInstruction::ComputeAccumulating(
                        event.buffer, stream_base,
                        MxmAccumulatorMode::DirectFinal,
                        0, MxmPairMode::Independent,
                        event.row == 0));
                stage_fp16_activation(
                    Hemisphere::West, mxm,
                    event, input_value(
                        event.group * kTokensPerGroup
                        + event.chain * kTokensPerRound
                        + event.row));
            }
            for (std::size_t path = 0; path < 2; ++path) {
                for (std::size_t tile = 0;
                     tile < hw::kTileRows; ++tile) {
                    front_tags_[path][tile].push_back(
                        FrontTag{
                            event.group, event.feature_block,
                            event.chain, event.row});
                }
            }
            ++front_issue_count_;
            event_text("front.G" + std::to_string(event.group)
                + ".B" + std::to_string(event.feature_block)
                + ".C" + std::to_string(event.chain)
                + ".R" + std::to_string(event.row));
        }
    }

    void stage_fp16_activation(
        Hemisphere side,
        std::size_t local_mxm,
        const FrontCompute& event,
        float value)
    {
        auto& fabric = system_->mem(side).stream_fabric();
        const auto column = std::size_t{0};
        const auto base = local_mxm * hw::kMxmLoadStreamsPerCycle;
        const auto bits = Fp16::from_float(value).bits();
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            if (event.issue_cycle
                    + tile * MxmControlSlice::kComputeTileLatency
                != cycle_) {
                continue;
            }
            for (std::size_t byte = 0; byte < 2; ++byte) {
                for (std::size_t lane = 0;
                     lane < hw::kLanesPerTile; ++lane) {
                    fabric.initialize_cell(
                        column, tile, lane,
                        StreamId::West(base + byte),
                        StreamCell::Valid(
                            static_cast<std::uint8_t>(bits >> (8 * byte)),
                            lane + 1 == hw::kLanesPerTile));
                }
            }
        }
    }

    void stage_front_activation_waves()
    {
        // Later tiles consume the same statically scheduled row with the
        // fixed 8-cycle control skew.
        for (const auto& event : front_compute_) {
            if (event.issue_cycle >= cycle_) continue;
            for (std::size_t tile = 1; tile < hw::kTileRows; ++tile) {
                if (event.issue_cycle
                        + tile * MxmControlSlice::kComputeTileLatency
                    != cycle_) {
                    continue;
                }
                const auto token = event.group * kTokensPerGroup
                    + event.chain * kTokensPerRound + event.row;
                for (std::size_t mxm = 0;
                     mxm < hw::kWestMxmCount; ++mxm) {
                    stage_fp16_activation(
                        Hemisphere::West, mxm,
                        event, input_value(token));
                }
            }
        }
    }

    void issue_down_compute()
    {
        const auto dequant_range = dequant_actions_.equal_range(cycle_);
        for (auto it = dequant_range.first;
             it != dequant_range.second; ++it) {
            for (std::size_t mxm = hw::kWestMxmCount;
                 mxm < hw::kMxmCount; ++mxm) {
                control_mxm(
                    mxm,
                    MxmControlInstruction::ActivationDequantize());
            }
            event_text("down.dequant.G" + std::to_string(it->second.group)
                + ".C" + std::to_string(it->second.chain)
                + ".R" + std::to_string(it->second.row));
        }

        for (const auto& event : down_compute_) {
            if (event.issue_cycle != cycle_) continue;
            for (std::size_t mxm = hw::kWestMxmCount;
                 mxm < hw::kMxmCount; ++mxm) {
                control_mxm(
                    mxm,
                    MxmControlInstruction::ComputeAccumulating(
                        0, 8,
                        MxmAccumulatorMode::DirectFinal,
                        0, MxmPairMode::Merge,
                        event.row == 0));
            }
            for (std::size_t tile = 0;
                 tile < hw::kTileRows; ++tile) {
                down_tags_[tile].push_back(
                    DownTag{event.group, event.chain, event.row});
            }
            ++down_issue_count_;
            event_text("down.compute.G" + std::to_string(event.group)
                + ".C" + std::to_string(event.chain)
                + ".R" + std::to_string(event.row));
        }
    }

    void issue_actions()
    {
        const auto mem_range = mem_actions_.equal_range(cycle_);
        auto grouped = std::map<std::size_t, std::vector<MemInstruction>>{};
        for (auto it = mem_range.first; it != mem_range.second; ++it) {
            grouped[it->second.slice].push_back(it->second.instruction);
        }
        for (auto& [slice, instructions] : grouped) {
            if (instructions.size() == 1) {
                control_mem(slice, instructions.front());
                continue;
            }
            if (instructions.size() == 2) {
                const auto read = std::find_if(
                    instructions.begin(), instructions.end(),
                    [](const auto& instruction) {
                        return instruction.opcode == MemOpcode::Read;
                    });
                const auto write = std::find_if(
                    instructions.begin(), instructions.end(),
                    [](const auto& instruction) {
                        return instruction.opcode == MemOpcode::Write;
                    });
                if (read != instructions.end()
                    && write != instructions.end()) {
                    control_mem(
                        slice, MemInstruction::ReadWrite(
                            read->address, read->stream_id(),
                            write->address, write->stream_id()));
                    continue;
                }
            }
            throw std::logic_error(
                "edge FFN scheduled more than one read/write per MEM slice");
        }

        const auto sxm_range = sxm_actions_.equal_range(cycle_);
        for (auto it = sxm_range.first; it != sxm_range.second; ++it) {
            control_sxm(it->second.instruction);
            if (it->second.instruction.opcode == SxmOpcode::Transpose) {
                sxm_transpose_tiles_this_cycle_ += hw::kTileRows;
                ++sxm_transpose_instructions_;
            } else {
                sxm_permute_tiles_this_cycle_ += hw::kTileRows;
                ++sxm_permute_instructions_;
            }
        }

        const auto program_range = vxm_program_actions_.equal_range(cycle_);
        for (auto it = program_range.first;
             it != program_range.second; ++it) {
            enqueue_vxm_program();
            event_text("VXM.start.G" + std::to_string(it->second.group));
        }

        const auto feed_range = vxm_feed_actions_.equal_range(cycle_);
        if (feed_range.first != feed_range.second) {
            // Dispatch the shared instruction rows before the direct Buffer
            // freezes this cycle's Bundle configuration.
            system_->vxm().prepare_cycle();
        }
        for (auto it = feed_range.first; it != feed_range.second; ++it) {
            direct_input_.feed_feature_tile(
                it->second.group,
                it->second.feature,
                it->second.tile,
                system_->vxm());
            for (std::size_t chain = 0;
                 chain < kChainsPerLane; ++chain) {
                vxm_output_tags_[it->second.tile][chain].push_back(
                    {it->second.group, it->second.feature});
            }
            if (it->second.release) {
                direct_input_.release_group(it->second.group);
            }
        }
    }

    void collect_accumulator_outputs()
    {
        for (const auto& output :
             system_->mxm_accumulator(Hemisphere::West).last_outputs()) {
            const auto path = output.stream_base < 16 ? 0U : 1U;
            auto& queue = front_tags_[path][output.tile];
            if (queue.empty()) {
                throw std::logic_error(
                    "Front ACC output has no token/feature tag");
            }
            const auto tag = queue.front();
            queue.pop_front();
            direct_input_.capture(
                tag.group, tag.chain, path,
                tag.feature_block, output);
            if (direct_input_.ready(tag.group)
                && !front_ready_cycle_[tag.group]) {
                front_ready_cycle_[tag.group] =
                    cycle_ + kDirectSrLatency + 1;
                event_text("direct-buffer.ready.G"
                    + std::to_string(tag.group));
            }
        }

        for (const auto& output :
             system_->mxm_accumulator(Hemisphere::East).last_outputs()) {
            if (output.stream_base != 8) {
                throw std::logic_error(
                    "Down merged ACC used an unexpected output stream");
            }
            auto& queue = down_tags_[output.tile];
            if (queue.empty()) {
                throw std::logic_error(
                    "Down ACC output has no compiler token tag");
            }
            const auto tag = queue.front();
            queue.pop_front();
            collect_down_output(output, tag);
        }
    }

    void collect_down_output(
        const MxmAccumulatorSlice::Output& output,
        const DownTag& tag)
    {
        if (output.tile != 0) return;
        auto last_due = std::size_t{0};
        for (std::size_t byte = 0; byte < 2; ++byte) {
            constexpr auto slice_base = std::size_t{40};
            const auto slice = slice_base + byte;
            const auto slice_group = slice / hw::kMemSlicesPerGroup;
            const auto due = cycle_ + 1 + (10 - slice_group);
            schedule_mem(
                due, slice,
                MemInstruction::Write(
                    kFinalAddress
                        + tag.group * kTokensPerGroup
                        + tag.chain * kTokensPerRound
                        + tag.row,
                    StreamId::West(8 + byte)));
            last_due = std::max(last_due, due);
        }
        ++final_tile0_outputs_;
        final_last_write_ = std::max(final_last_write_, last_due);
        final_outputs_this_cycle_ += kOutputFeatures;
    }

    void collect_vxm_outputs()
    {
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            for (const auto& output : system_->vxm().outputs_at(tile)) {
                const auto chain = output.stream < 8 ? 0U : 1U;
                auto& queue = vxm_output_tags_[tile][chain];
                if (queue.empty()) {
                    throw std::logic_error(
                        "VXM output has no group/feature tag");
                }
                const auto [group, feature] = queue.front();
                queue.pop_front();
                quantized_outputs_this_cycle_ += hw::kLanesPerTile;
                swiglu_values_checked_ += hw::kLanesPerTile;
                if (tile != 0) continue;

                const auto slice = kRawQuantSliceBases[chain]
                    + feature % hw::kLanesPerTile;
                const auto slice_group =
                    slice / hw::kMemSlicesPerGroup;
                const auto due = cycle_ + 1 + slice_group;
                schedule_mem(
                    due, slice,
                    MemInstruction::Write(
                        kQuantInputAddress
                            + group
                                * (kChainsPerLane * hw::kTileRows)
                            + feature / hw::kLanesPerTile,
                        StreamId::East(output.stream)));
                ++activation_mem_writes_;
                ++quant_tile0_outputs_[group];
                quant_last_write_[group] = std::max(
                    quant_last_write_[group], due);
                if (quant_tile0_outputs_[group]
                    == kChainsPerLane * kIntermediate) {
                    quant_ready_cycle_[group] =
                        quant_last_write_[group]
                        + hw::kTileRows + 1;
                }
            }
        }
    }

    static std::size_t active_cells(const Mxm& mxm)
    {
        auto result = std::size_t{0};
        for (std::size_t row = 0; row < hw::kTileRows; ++row) {
            for (std::size_t column = 0;
                 column < hw::kTileRows; ++column) {
                result += mxm.computing_cell(row, column) ? 1 : 0;
            }
        }
        return result;
    }

    std::size_t vxm_slots() const
    {
        auto slots = std::size_t{0};
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            for (const auto& trace :
                 system_->vxm().superlane(tile).lane(0).last_trace()) {
                if (trace.state == VxmLaneAluTraceState::Executed) {
                    slots += hw::kLanesPerTile;
                }
            }
        }
        return slots;
    }

    void maybe_schedule()
    {
        while (front_groups_scheduled_ < kTokenGroups
            && front_ready_cycle_[front_groups_scheduled_]
            && cycle_ >= *front_ready_cycle_[front_groups_scheduled_]) {
            schedule_vxm_group(front_groups_scheduled_);
        }
        while (quant_groups_scheduled_ < kTokenGroups
            && quant_ready_cycle_[quant_groups_scheduled_]
            && cycle_ >= *quant_ready_cycle_[quant_groups_scheduled_]) {
            schedule_quantized_group(quant_groups_scheduled_);
        }
    }

    void tick()
    {
        events_.clear();
        sxm_transpose_tiles_this_cycle_ = 0;
        sxm_permute_tiles_this_cycle_ = 0;
        quantized_outputs_this_cycle_ = 0;
        final_outputs_this_cycle_ = 0;

        const auto icu_replay =
            control_mode_ == ControlMode::IcuReplay;
        if (icu_replay) {
            // Match the old Harness issue point: local IQ heads dispatch
            // before this cycle's data staging and before VXM prepare_cycle.
            system_->dispatch_icu_only();
        }

        issue_actions();
        issue_loads();
        issue_front_compute();
        stage_front_activation_waves();
        issue_down_compute();
        system_->tick({}, !icu_replay);
        collect_accumulator_outputs();
        collect_vxm_outputs();
        record_cycle();
        ++cycle_;
        maybe_schedule();
    }

    void record_cycle()
    {
        const auto gate = active_cells(system_->mxm_unit(0));
        const auto up = active_cells(system_->mxm_unit(1));
        const auto down = active_cells(system_->mxm_unit(2))
            + active_cells(system_->mxm_unit(3));
        const auto slots = vxm_slots();
        auto mem_reads = std::size_t{0};
        auto mem_writes = std::size_t{0};
        for (const auto& transfer :
             system_->mem(Hemisphere::East)
                 .memory_model().executed_transfers()) {
            if (transfer.kind
                == MemArrayModel::MemTransfer::Kind::LoadSramToStream) {
                ++mem_reads;
                ++activation_mem_reads_;
            } else {
                ++mem_writes;
            }
        }
        const auto phase = phase_name(gate + up, slots, down);
        const auto icu_activity =
            system_->icu().last_cycle_activity();
        timeline_.push_back(CycleRecord{
            cycle_, phase, events_, gate, up, slots,
            quantized_outputs_this_cycle_, mem_reads, mem_writes,
            sxm_transpose_tiles_this_cycle_,
            sxm_permute_tiles_this_cycle_,
            front_load_tiles_this_cycle_,
            down_load_tiles_this_cycle_, down,
            final_outputs_this_cycle_,
            icu_activity.imem_read_starts,
            icu_activity.iq_arrivals,
            icu_activity.dispatched_entries,
            icu_activity.waiting_queues});

        if (gate + up != 0 && slots != 0) {
            ++mxm_vxm_overlap_cycles_;
        }
        if (down != 0
            && (sxm_transpose_tiles_this_cycle_ != 0
                || sxm_permute_tiles_this_cycle_ != 0)) {
            ++sxm_mxm_overlap_cycles_;
        }
        if (gate + up != 0
            && front_load_tiles_this_cycle_ != 0) {
            ++compute_load_overlap_cycles_;
        }
    }

    static std::string phase_name(
        std::size_t front,
        std::size_t vxm,
        std::size_t down)
    {
        auto name = std::string{};
        const auto append = [&name](const char* part) {
            if (!name.empty()) name += '+';
            name += part;
        };
        if (front) append("Front-MXM");
        if (vxm) append("VXM");
        if (down) append("Down-MXM");
        if (name.empty()) name = "transport";
        return name;
    }

    void event_text(std::string text)
    {
        if (!events_.empty()) events_ += ',';
        events_ += std::move(text);
    }

    bool queues_empty() const
    {
        for (const auto& path : front_tags_) {
            for (const auto& queue : path) {
                if (!queue.empty()) return false;
            }
        }
        for (const auto& queue : down_tags_) {
            if (!queue.empty()) return false;
        }
        for (const auto& tile : vxm_output_tags_) {
            for (const auto& queue : tile) {
                if (!queue.empty()) return false;
            }
        }
        return true;
    }

    bool complete() const
    {
        return quant_groups_scheduled_ == kTokenGroups
            && final_tile0_outputs_ == kTokens
            && cycle_ > final_last_write_ + hw::kTileRows + 2
            && queues_empty();
    }

    void verify_final_memory()
    {
        const auto& mem =
            system_->mem(Hemisphere::East).memory_model();
        auto checked = std::size_t{0};
        for (std::size_t token = 0; token < kTokens; ++token) {
            const auto expected = Fp16::from_float(
                static_cast<float>(expected_quantized(token))).bits();
            for (std::size_t tile = 0;
                 tile < hw::kTileRows; ++tile) {
                for (std::size_t lane = 0;
                     lane < hw::kLanesPerTile; ++lane) {
                    std::uint16_t actual = 0;
                    for (std::size_t byte = 0; byte < 2; ++byte) {
                        actual |= static_cast<std::uint16_t>(
                            mem.sram_lane_byte(
                                40 + byte, tile,
                                kFinalAddress + token,
                                lane)) << (8 * byte);
                    }
                    if (actual != expected) {
                        auto error = std::ostringstream{};
                        error << "edge FFN mismatch token=" << token
                              << " feature="
                              << tile * hw::kLanesPerTile + lane
                              << " actual=0x" << std::hex << actual
                              << " expected=0x" << expected;
                        throw std::runtime_error(error.str());
                    }
                    ++checked;
                }
            }
        }
        down_values_checked_ = checked;
    }

    FullFfnResult finish_result() const
    {
        const auto cycles = timeline_.size();
        constexpr auto cells_per_mxm =
            hw::kTileRows * hw::kTileRows;
        constexpr auto total_mxm_capacity =
            hw::kMxmCount * cells_per_mxm;
        constexpr auto front_capacity = 2 * cells_per_mxm;
        constexpr auto down_capacity = 2 * cells_per_mxm;
        constexpr auto vxm_capacity =
            hw::kTileRows * hw::kLanesPerTile * VxmLane::kAluCount;

        auto front_work = std::size_t{0};
        auto front_time = std::size_t{0};
        auto down_work = std::size_t{0};
        auto down_time = std::size_t{0};
        auto vxm_work = std::size_t{0};
        auto vxm_time = std::size_t{0};
        auto quant_work = std::size_t{0};
        auto quant_time = std::size_t{0};
        auto mem_work = std::size_t{0};
        auto mem_time = std::size_t{0};
        auto sxm_work = std::size_t{0};
        auto sxm_time = std::size_t{0};
        auto peak_vxm = std::size_t{0};
        auto peak_quant = std::size_t{0};
        auto total_mxm_time = std::size_t{0};
        for (const auto& record : timeline_) {
            const auto front = record.gate_mxm_cells
                + record.up_mxm_cells;
            const auto down = record.down_mxm_cells;
            front_work += front;
            front_time += front != 0;
            down_work += down;
            down_time += down != 0;
            total_mxm_time += front != 0 || down != 0;
            vxm_work += record.vxm_alu_slots;
            vxm_time += record.vxm_alu_slots != 0;
            peak_vxm = std::max(peak_vxm, record.vxm_alu_slots);
            quant_work += record.quantized_outputs;
            quant_time += record.quantized_outputs != 0;
            peak_quant = std::max(
                peak_quant, record.quantized_outputs);
            const auto mem = record.mem_reads + record.mem_writes;
            mem_work += mem;
            mem_time += mem != 0;
            const auto sxm = record.sxm_transpose_tiles
                + record.sxm_permute_tiles;
            sxm_work += sxm;
            sxm_time += sxm != 0;
        }

        const auto utilization = [](
            std::size_t work,
            std::size_t active_cycles,
            std::size_t capacity,
            std::size_t total) {
            auto result = Utilization{};
            result.active = active_cycles == 0 ? 0.0
                : static_cast<double>(work)
                    / static_cast<double>(active_cycles * capacity);
            result.time = total == 0 ? 0.0
                : static_cast<double>(active_cycles) / total;
            result.whole = total == 0 ? 0.0
                : static_cast<double>(work)
                    / static_cast<double>(total * capacity);
            return result;
        };

        auto result = FullFfnResult{};
        result.icu_driven =
            control_mode_ == ControlMode::IcuReplay;
        result.name = result.icu_driven
            ? "icu_driven_edge_ffn_256x32x64x32"
            : "pipelined_edge_ffn_256x32x64x32";
        result.description = result.icu_driven
            ? "The compiled distributed ICU program drives the same two "
              "resident Front MXMs, lane-banked VXM, MEM/SXM transpose path "
              "and two Down MXMs as the direct reference schedule; all "
              "functional controls originate in local i-MEM/IQ endpoints."
            : "Two resident Front MXMs feed the lane-banked VXM input Buffer "
              "directly while two independent Down MXMs consume prior token "
              "groups through MEM/SXM; chain-derived two-token weight reuse, "
              "background IW, horizontal activation flow and vertical partial "
              "flow are all timed in the modeled hardware.";
        result.tokens = kTokens;
        result.hidden = kHidden;
        result.intermediate = kIntermediate;
        result.output_features = kOutputFeatures;
        result.chain_depth = static_cast<std::size_t>(kChainDepth);
        result.chains_per_lane = kChainsPerLane;
        result.weight_reuse_per_buffer = kChainsPerLane;
        result.configured_quantizer_channels =
            hw::kTileRows * hw::kLanesPerTile * kChainsPerLane;
        result.quantizer_channels =
            hw::kTileRows * hw::kLanesPerTile * VxmLane::kBlockCount;
        result.vxm_peak_alu_slots = peak_vxm;
        result.quantizer_peak_outputs = peak_quant;
        result.cycles = cycles;
        result.swiglu_values_checked = swiglu_values_checked_;
        result.layout_values_checked =
            direct_input_.captured_values();
        result.down_values_checked = down_values_checked_;
        result.activation_mem_writes = activation_mem_writes_;
        result.activation_mem_reads = activation_mem_reads_;
        result.sxm_transpose_instructions = sxm_transpose_instructions_;
        result.sxm_permute_instructions = sxm_permute_instructions_;
        result.down_pair_merges = down_issue_count_ * kOutputFeatures;
        result.down_weight_reuses = down_issue_count_ * 2;
        result.front_compute_issues = front_issue_count_;
        result.front_compute_bubbles = 0;
        result.down_compute_issues = down_issue_count_;
        // Every compiler-issued 32-row Down burst has II=1.  Cycles between
        // bursts are reported separately because the MXM is waiting for the
        // next completed token group, not bubbling inside one Compute.
        result.down_compute_bubbles = 0;
        result.down_input_wait_cycles = down_schedule_bubbles();
        result.gate_up_compute_load_overlap_cycles =
            compute_load_overlap_cycles_;
        result.mxm_vxm_overlap_cycles = mxm_vxm_overlap_cycles_;
        result.sxm_down_mxm_overlap_cycles = sxm_mxm_overlap_cycles_;
        result.vxm_overlap_percentage = vxm_time == 0 ? 0.0
            : static_cast<double>(mxm_vxm_overlap_cycles_) / vxm_time;
        result.down_mxm_overlap_percentage = down_time == 0 ? 0.0
            : static_cast<double>(sxm_mxm_overlap_cycles_) / down_time;
        result.down_accumulator_format =
            "FP32 pair-merge (two 32-wide K halves), FP16 cast";
        result.correctness_passed =
            down_values_checked_ == kTokens * kOutputFeatures;
        result.functionality_passed = result.correctness_passed
            && direct_input_.captured_values()
                == kTokens * kIntermediate * 2
            && direct_input_.emitted_values()
                == kTokens * kIntermediate * 2
            && peak_vxm == vxm_capacity
            && front_issue_count_
                == kTokenGroups * kFeatureBlocks
                    * kChainsPerLane * hw::kMxmRows
            && compute_load_overlap_cycles_ != 0
            && mxm_vxm_overlap_cycles_ != 0
            && sxm_mxm_overlap_cycles_ != 0;

        if (result.icu_driven) {
            const auto statistics = system_->icu().statistics();
            result.icu_prefetch_cycles =
                icu_ffn_test::FfnIcuProgram::kPrefetchCycles;
            result.icu_active_queues = icu_program_.active_queues();
            result.icu_functional_events =
                icu_program_.functional_events();
            result.icu_programmed_instructions =
                statistics.programmed_instructions;
            result.icu_fetched_instructions =
                statistics.fetched_instructions;
            result.icu_functional_issues =
                statistics.functional_issues;
            result.icu_underflowed_queues =
                statistics.underflowed_queues;
            result.functionality_passed = result.functionality_passed
                && statistics.underflowed_queues == 0
                && statistics.unfinished_queues == 0
                && statistics.functional_issues
                    == icu_program_.functional_events();
        }

        result.gate_up_mxm = utilization(
            front_work, front_time, front_capacity, cycles);
        result.down_mxm = utilization(
            down_work, down_time, down_capacity, cycles);
        result.total_mxm = utilization(
            front_work + down_work,
            total_mxm_time, total_mxm_capacity, cycles);
        result.vxm = utilization(
            vxm_work, vxm_time, vxm_capacity, cycles);
        result.vxm_total = result.vxm;
        result.quantizer = utilization(
            quant_work, quant_time,
            result.configured_quantizer_channels, cycles);
        result.quantizer_physical = utilization(
            quant_work, quant_time,
            result.quantizer_channels, cycles);
        result.mem = utilization(
            mem_work, mem_time,
            hw::kMemSliceColumns * hw::kTileRows, cycles);
        result.sxm = utilization(
            sxm_work, sxm_time,
            2 * hw::kTileRows, cycles);
        result.timeline = timeline_;
        return result;
    }

    std::size_t down_schedule_bubbles() const
    {
        if (down_compute_.empty()) return 0;
        auto bubbles = std::size_t{0};
        for (std::size_t index = 1;
             index < down_compute_.size(); ++index) {
            const auto previous = down_compute_[index - 1].issue_cycle;
            const auto current = down_compute_[index].issue_cycle;
            if (current > previous + 1) {
                bubbles += current - previous - 1;
            }
        }
        return bubbles;
    }

    std::unique_ptr<TspSliceSystem> system_{};
    MxmVxmInputBuffer direct_input_;
    ControlMode control_mode_{ControlMode::DirectRecord};
    icu_ffn_test::FfnIcuProgram icu_program_{};
    std::vector<LoadWave> load_waves_{};
    std::vector<FrontCompute> front_compute_{};
    std::vector<DownCompute> down_compute_{};
    std::multimap<std::size_t, MemAction> mem_actions_{};
    std::multimap<std::size_t, SxmAction> sxm_actions_{};
    std::multimap<std::size_t, VxmProgramAction> vxm_program_actions_{};
    std::multimap<std::size_t, VxmFeedAction> vxm_feed_actions_{};
    std::multimap<std::size_t, DequantAction> dequant_actions_{};
    std::array<std::array<std::deque<FrontTag>, hw::kTileRows>, 2>
        front_tags_{};
    std::array<std::deque<DownTag>, hw::kTileRows> down_tags_{};
    std::array<
        std::array<
            std::deque<std::pair<std::size_t, std::size_t>>,
            kChainsPerLane>,
        hw::kTileRows> vxm_output_tags_{};
    std::array<std::optional<std::size_t>, kTokenGroups>
        front_ready_cycle_{};
    std::array<std::size_t, kTokenGroups> quant_tile0_outputs_{};
    std::array<std::size_t, kTokenGroups> quant_last_write_{};
    std::array<std::optional<std::size_t>, kTokenGroups>
        quant_ready_cycle_{};
    std::vector<CycleRecord> timeline_{};
    std::size_t cycle_{0};
    std::size_t front_compute_start_{0};
    std::size_t front_last_issue_{0};
    std::size_t front_groups_scheduled_{0};
    std::size_t quant_groups_scheduled_{0};
    std::size_t vxm_south_available_{0};
    std::size_t sxm_available_{0};
    std::size_t down_available_{0};
    std::size_t final_tile0_outputs_{0};
    std::size_t final_last_write_{0};
    std::size_t final_outputs_this_cycle_{0};
    std::size_t front_load_tiles_this_cycle_{0};
    std::size_t down_load_tiles_this_cycle_{0};
    std::size_t sxm_transpose_tiles_this_cycle_{0};
    std::size_t sxm_permute_tiles_this_cycle_{0};
    std::size_t quantized_outputs_this_cycle_{0};
    std::size_t swiglu_values_checked_{0};
    std::size_t down_values_checked_{0};
    std::size_t activation_mem_writes_{0};
    std::size_t activation_mem_reads_{0};
    std::size_t sxm_transpose_instructions_{0};
    std::size_t sxm_permute_instructions_{0};
    std::size_t front_issue_count_{0};
    std::size_t down_issue_count_{0};
    std::size_t compute_load_overlap_cycles_{0};
    std::size_t mxm_vxm_overlap_cycles_{0};
    std::size_t sxm_mxm_overlap_cycles_{0};
    std::string events_{};
};

} // namespace full_ffn_test
