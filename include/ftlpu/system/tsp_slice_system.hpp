#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/hemisphere.hpp"
#include "ftlpu/mem/tile_array.hpp"
#include "ftlpu/mxm/mxm.hpp"
#include "ftlpu/sxm/slice.hpp"
#include "ftlpu/system/icu.hpp"
#include "ftlpu/vxm/slice.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <vector>

namespace ftlpu {

class TspSliceSystem {
public:
    static constexpr std::size_t kMxmCountPerHemisphere =
        hw::kMxmsPerHemisphere;
    static constexpr std::size_t kMxmCount = hw::kMxmCount;

    struct LogSinks {
        std::ostream* icu{nullptr};
        std::ostream* mem{nullptr};
        std::ostream* mxm{nullptr};
        std::ostream* vxm{nullptr};
        std::ostream* system{nullptr};
        std::optional<std::size_t> mem_log_tile{};
        std::optional<std::size_t> mxm_log_tile{};
        std::optional<std::size_t> vxm_log_tile{};
        std::ostream* sxm{nullptr};
    };

    struct VxmSuperlaneTiming {
        VxmChainDepth depth{VxmChainDepth::Two};
        std::array<VxmLaneAluTraceState, VxmLane::kAluCount> alu_states{};
        std::size_t outputs{0};
    };

    struct VxmTimingSnapshot {
        std::size_t cycle{0};
        std::array<VxmSuperlaneTiming, hw::kTileRows> superlanes{};
    };

    struct MemTiming {
        std::size_t reads{0};
        std::size_t writes{0};
    };

    struct MxmTiming {
        std::size_t compute_issues{0};
        std::size_t computing_cells{0};
        std::size_t deskew_writes{0};
        std::size_t deskew_vectors{0};
        std::size_t outputs{0};
    };

    struct SystemTimingSnapshot {
        std::size_t cycle{0};
        InstructionControlUnit::TimingSnapshot icu{};
        std::array<MemTiming, hw::kHemispheres> mems{};
        std::array<MxmTiming, kMxmCount> mxms{};
        std::array<SxmSlice::TimingSnapshot, hw::kHemispheres> sxms{};
        VxmTimingSnapshot vxm{};
    };

    using TimingObserver =
        std::function<void(const SystemTimingSnapshot&)>;

    void set_timing_observer(TimingObserver observer)
    {
        timing_observer_ = std::move(observer);
    }

    void clear_timing_observer() noexcept
    {
        timing_observer_ = {};
    }

    TspSliceSystem()
        : sxms_ {
            SxmSlice(make_sxm_port_map()),
            SxmSlice(make_sxm_port_map()),
        }
    {
    }

    void initialize_mem_sram_lane_byte(
        std::size_t column,
        std::size_t tile,
        std::size_t row,
        std::size_t lane,
        std::uint8_t value)
    {
        initialize_mem_sram_lane_byte(Hemisphere::East, column, tile, row, lane, value);
    }

    void initialize_mem_sram_lane_byte(
        Hemisphere hemisphere,
        std::size_t column,
        std::size_t tile,
        std::size_t row,
        std::size_t lane,
        std::uint8_t value)
    {
        mems_[hemisphere_index(hemisphere)].set_sram_lane_byte(column, tile, row, lane, value);
    }

    std::uint8_t read_mem_sram_lane_byte(
        std::size_t column,
        std::size_t tile,
        std::size_t row,
        std::size_t lane) const
    {
        return read_mem_sram_lane_byte(Hemisphere::East, column, tile, row, lane);
    }

    std::uint8_t read_mem_sram_lane_byte(
        Hemisphere hemisphere,
        std::size_t column,
        std::size_t tile,
        std::size_t row,
        std::size_t lane) const
    {
        return mems_[hemisphere_index(hemisphere)].sram_lane_byte(column, tile, row, lane);
    }

    InstructionControlUnit& icu()
    {
        return icu_;
    }

    const InstructionControlUnit& icu() const
    {
        return icu_;
    }

    Mxm& mxm_unit(std::size_t mxm)
    {
        return mxms_.at(mxm);
    }

    const Mxm& mxm_unit(std::size_t mxm) const
    {
        return mxms_.at(mxm);
    }

    VxmSlice& vxm_unit() noexcept
    {
        return vxm_;
    }

    const VxmSlice& vxm_unit() const noexcept
    {
        return vxm_;
    }

    // Boot-time LUT data loading is exposed at the system boundary just like
    // MEM SRAM initialization. Tests and loaders do not need access to VXM
    // lane/superlane internals.
    void initialize_vxm_lut(
        VxmSpecialAluOpcode opcode, VxmLutConfig config,
        const std::vector<VxmLutEntry>& entries)
    {
        vxm_.configure_special_lut(opcode, config, entries);
    }

    // Boot-time VXM edge routing.  This is architectural configuration at
    // the system boundary; workloads still supply operands through MEM/SR
    // and ICU instructions rather than through VXM implementation objects.
    void configure_vxm_input_group_source(
        std::size_t group, Hemisphere source)
    {
        vxm_.configure_input_group_source(group, source);
    }

    void configure_vxm_output_block_destination(
        std::size_t block, Hemisphere destination)
    {
        vxm_.configure_output_block_destination(block, destination);
    }

    void tick(std::ostream& os)
    {
        LogSinks sinks {&os, &os, &os, &os, &os};
        tick(sinks);
    }

    void tick(LogSinks sinks)
    {
        begin_cycle_phase(sinks);
        dispatch_phase(sinks);
        mxm_phase(sinks);
        vxm_phase(sinks);
        mem_sxm_commit_phase(sinks);
        end_cycle_phase();
        if (timing_observer_) {
            timing_observer_(system_timing_snapshot());
        }
    }

    std::size_t cycle() const
    {
        return cycle_;
    }

    // Read-only system-boundary instrumentation. A Superlane's eight SIMD
    // lanes share control and execute in lockstep, so one 16-stage state
    // vector represents the whole Superlane without exposing lane data.
    VxmTimingSnapshot vxm_timing_snapshot() const
    {
        auto result = VxmTimingSnapshot{};
        result.cycle = cycle_ == 0 ? 0 : cycle_ - 1;
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            const auto& superlane = vxm_.superlane(tile);
            auto& timing = result.superlanes[tile];
            timing.depth = superlane.lane(0).chain_depth();
            timing.outputs = vxm_.outputs_at(tile).size();
            for (std::size_t stage = 0;
                 stage < VxmLane::kAluCount; ++stage) {
                timing.alu_states[stage] =
                    superlane.lane(0).last_trace()[stage].state;
            }
        }
        return result;
    }

    // Unified read-only instrumentation for integration timing reports.  It
    // exposes activity counts only; data values and module internals remain
    // behind the normal system boundary.
    SystemTimingSnapshot system_timing_snapshot() const
    {
        auto result = SystemTimingSnapshot{};
        result.cycle = cycle_ == 0 ? 0 : cycle_ - 1;
        result.icu = icu_.timing_snapshot();
        result.vxm = vxm_timing_snapshot();

        for (std::size_t hemisphere = 0;
             hemisphere < hw::kHemispheres; ++hemisphere) {
            const auto& memory = mems_[hemisphere].memory_model();
            result.mems[hemisphere].reads = memory.executed_read_count();
            result.mems[hemisphere].writes = memory.executed_write_count();
            result.sxms[hemisphere] = sxms_[hemisphere].timing_snapshot();
        }

        for (std::size_t mxm = 0; mxm < kMxmCount; ++mxm) {
            const auto& unit = mxms_[mxm];
            auto& timing = result.mxms[mxm];
            for (std::size_t tile = 0;
                 tile < hw::kMxmSupercellsPerPlane; ++tile) {
                timing.compute_issues += unit.control().compute_active(tile)
                    ? 1U : 0U;
                for (std::size_t column = 0;
                     column < hw::kMxmSupercellsPerPlane; ++column) {
                    timing.computing_cells += unit.computing_cell(tile, column)
                        ? 1U : 0U;
                }
            }
            timing.deskew_writes = unit.last_deskew_writes();
            timing.deskew_vectors = unit.last_deskew_vectors();
            timing.outputs = unit.last_outputs().size();
        }
        return result;
    }

    void reset_execution_state()
    {
        require_phase(CyclePhase::Idle, "resetting execution state");
        for (auto& mem : mems_) mem.reset_execution_state();
        vxm_.reset();
        for (auto& sxm : sxms_) sxm.reset();
        for (auto& mxm : mxms_) mxm.reset();
        icu_.reset();
        cycle_ = 0;
    }

private:
    enum class CyclePhase {
        Idle,
        Begun,
        Dispatched,
        MxmEvaluated,
        VxmEvaluated,
        MemSxmCommitted,
    };

    void require_phase(CyclePhase expected, const char* operation) const
    {
        if (phase_ != expected) {
            throw std::logic_error(
                std::string("TSP cycle phase violation while ") + operation);
        }
    }

    void begin_cycle_phase(LogSinks sinks)
    {
        require_phase(CyclePhase::Idle, "beginning cycle");
        if (sinks.system != nullptr) {
            *sinks.system << "system cycle " << cycle_ << '\n';
        }
        phase_ = CyclePhase::Begun;
    }

    void dispatch_phase(LogSinks sinks)
    {
        require_phase(CyclePhase::Begun, "dispatching ICU instructions");
        icu_.dispatch(mems_, vxm_, sxms_, mxms_, sinks.icu);
        phase_ = CyclePhase::Dispatched;
    }

    void mxm_phase(LogSinks sinks)
    {
        require_phase(CyclePhase::Dispatched, "evaluating MXM");
        tick_mxm_controls(sinks);
        tick_mxm_datapaths(sinks);
        phase_ = CyclePhase::MxmEvaluated;
    }

    void vxm_phase(LogSinks sinks)
    {
        require_phase(CyclePhase::MxmEvaluated, "evaluating VXM");
        vxm_.prepare_cycle();
        transfer_mem_edges_to_vxm(sinks);
        transfer_unconsumed_streams_across_vxm(sinks);
        vxm_.tick(sinks.vxm, sinks.vxm_log_tile);
        transfer_vxm_to_mem_edges(sinks);
        phase_ = CyclePhase::VxmEvaluated;
    }

    void mem_sxm_commit_phase(LogSinks sinks)
    {
        require_phase(CyclePhase::VxmEvaluated, "committing MEM and SXM");
        for (std::size_t hemisphere = 0; hemisphere < hw::kHemispheres; ++hemisphere) {
            sxms_[hemisphere].set_trace_enabled(sinks.sxm != nullptr);
            if (sinks.mem != nullptr) {
                *sinks.mem << "mem." << hemisphere_short_name(static_cast<Hemisphere>(hemisphere))
                           << " cycle " << cycle_ << '\n';
                mems_[hemisphere].tick(sxms_[hemisphere], *sinks.mem, sinks.mem_log_tile);
            } else {
                mems_[hemisphere].tick(sxms_[hemisphere]);
            }
            if (sinks.sxm != nullptr) {
                *sinks.sxm << "sxm."
                            << hemisphere_short_name(static_cast<Hemisphere>(hemisphere))
                            << " system_cycle " << cycle_ << '\n';
                sxms_[hemisphere].log_cycle(*sinks.sxm);
            }
        }
        phase_ = CyclePhase::MemSxmCommitted;
    }

    void end_cycle_phase()
    {
        require_phase(CyclePhase::MemSxmCommitted, "ending cycle");
        icu_.advance_barrier_events();
        ++cycle_;
        phase_ = CyclePhase::Idle;
    }
    static SxmStreamPortMap make_sxm_port_map()
    {
        return SxmStreamPortMap::BetweenColumns(
            hw::kMemEastBoundaryStreamRegisterColumn,
            hw::kMxmBoundaryStreamRegisterColumn,
            hw::kMxmBoundaryStreamRegisterColumn,
            hw::kMemEastBoundaryStreamRegisterColumn);
    }

    static Hemisphere mxm_hemisphere(std::size_t mxm)
    {
        return static_cast<Hemisphere>(mxm / kMxmCountPerHemisphere);
    }

    static std::size_t local_mxm_index(std::size_t mxm)
    {
        return mxm % kMxmCountPerHemisphere;
    }

    static const TileArrayModel::StreamSlot& mem_edge_stream(
        const TileArrayModel& mem,
        std::size_t tile,
        std::size_t lane,
        std::size_t stream)
    {
        if (stream < hw::kEastStreams) {
            return mem.east_register(tile, lane, 0, stream);
        }
        return mem.west_register(tile, lane, 0, stream - hw::kEastStreams);
    }

    void tick_mxm_controls(LogSinks sinks)
    {
        for (std::size_t mxm = 0; mxm < kMxmCount; ++mxm) {
            if (sinks.mxm != nullptr) {
                *sinks.mxm << "mxm" << mxm << " cycle " << cycle_ << '\n';
            }
            auto provider = [this, mxm, sinks](std::size_t tile) {
                if (sinks.mxm != nullptr && (!sinks.mxm_log_tile.has_value() || tile == *sinks.mxm_log_tile)) {
                    *sinks.mxm << "  SXM.sreg"
                               << hw::kMxmBoundaryStreamRegisterColumn
                               << " -> MXM" << mxm << " tile " << tile << '\n';
                }
                try {
                    return collect_mxm_weight_input_from_streams(mxm, tile);
                } catch (const std::exception& ex) {
                    throw std::logic_error(
                        "MXM" + std::to_string(mxm)
                        + " IW tile " + std::to_string(tile) + ": " + ex.what());
                }
            };
            if (sinks.mxm != nullptr) {
                mxms_[mxm].control().tick(*sinks.mxm, provider, false, sinks.mxm_log_tile);
            } else {
                static NullStream null_stream;
                mxms_[mxm].control().tick(null_stream.stream(), provider, false);
            }
        }
    }

    void tick_mxm_datapaths(LogSinks sinks)
    {
        for (std::size_t mxm = 0; mxm < kMxmCount; ++mxm) {
            const auto hemisphere = hemisphere_index(mxm_hemisphere(mxm));
            mxms_[mxm].tick_datapath(
                mems_[hemisphere], local_mxm_index(mxm), sinks.mxm, sinks.mxm_log_tile);
        }
    }

    MxmControlSlice::WeightInput collect_mxm_weight_input_from_streams(std::size_t mxm, std::size_t tile)
    {
        constexpr auto kTargetSreg = hw::kMxmBoundaryStreamRegisterColumn;
        auto input = MxmControlSlice::WeightInput {};
        const auto hemisphere = hemisphere_index(mxm_hemisphere(mxm));
        const auto& instruction = mxms_[mxm].control().instruction_at(tile);
        if (!instruction.has_value()
            || instruction->opcode != MxmControlOpcode::IW) {
            throw std::logic_error(
                "MXM weight input requested without an active IW instruction");
        }
        mxms_[mxm].validate_weight_buffer_load(
            instruction->weight_buffer, tile);
        if (instruction->weight_input_mode
            == MxmWeightInputMode::Int8DequantBf16) {
            const auto stream_base = local_mxm_index(mxm)
                * hw::kMxmInt8LoadStreamStride;
            if (instruction->weight_load_mode
                == MxmWeightLoadMode::Column) {
                const auto column = instruction->weight_inner_column;
                for (std::size_t lane = 0;
                     lane < hw::kLanesPerTile;
                     ++lane) {
                    const auto word =
                        mems_[hemisphere].consume_east_register(
                            tile,
                            lane,
                            kTargetSreg,
                            stream_base);
                    if (!word.has_value()) {
                        throw std::logic_error(
                            "MXM INT8 column IW reached tile before its weight stream arrived at the MXM boundary register");
                    }
                    input[lane][column] =
                        MxmArray::Supercell::InputWord {
                            word->data,
                            lane + 1 == hw::kLanesPerTile,
                        };
                }
                return input;
            }

            for (std::size_t lane = 0;
                 lane < hw::kLanesPerTile;
                 ++lane) {
                for (std::size_t column = 0;
                     column < hw::kMxmSupercellColumns;
                     ++column) {
                    const auto word =
                        mems_[hemisphere].consume_east_register(
                            tile,
                            lane,
                            kTargetSreg,
                            stream_base + column);
                    if (!word.has_value()) {
                        throw std::logic_error(
                            "MXM INT8 IW reached tile before all eight weight streams arrived at the MXM boundary register");
                    }
                    input[lane][column] =
                        MxmArray::Supercell::InputWord {
                            word->data,
                            column + 1
                                == hw::kMxmSupercellColumns,
                        };
                }
            }
            return input;
        }

        const auto stream_base =
            local_mxm_index(mxm) * hw::kMxmLoadStreamStride;
        if (instruction->weight_input_mode
            != MxmWeightInputMode::Direct16) {
            throw std::invalid_argument("MXM weight input mode is invalid");
        }
        if (instruction->weight_load_mode == MxmWeightLoadMode::Column) {
            const auto low = stream_base;
            const auto high =
                stream_base + hw::kMxmColumnLoadStreamsPerCycle - 1;
            const auto column = instruction->weight_inner_column;
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                const auto low_word = mems_[hemisphere].consume_east_register(
                    tile, lane, kTargetSreg, low);
                const auto high_word = mems_[hemisphere].consume_east_register(
                    tile, lane, kTargetSreg, high);
                if (!low_word.has_value() || !high_word.has_value()) {
                    throw std::logic_error(
                        "MXM column IW reached tile before both 16-bit weight streams arrived at the MXM boundary register");
                }
                const auto bits = static_cast<std::uint16_t>(
                    static_cast<std::uint16_t>(low_word->data)
                    | (static_cast<std::uint16_t>(high_word->data) << 8));
                input[lane][column] = MxmArray::Supercell::InputWord {
                    bits,
                    lane + 1 == hw::kLanesPerTile,
                };
            }
            return input;
        }

        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            for (std::size_t column = 0; column < hw::kMxmSupercellColumns; ++column) {
                const auto low_stream = stream_base + column * hw::kMxmWeightBytesPerValue;
                const auto low = mems_[hemisphere].consume_east_register(tile, lane, kTargetSreg, low_stream);
                const auto high = mems_[hemisphere].consume_east_register(tile, lane, kTargetSreg, low_stream + 1);
                if (!low.has_value() || !high.has_value()) {
                    throw std::logic_error(
                        "MXM IW reached tile before both 16-bit weight streams arrived at the MXM boundary register");
                }
                const auto bits = static_cast<std::uint16_t>(
                    static_cast<std::uint16_t>(low->data)
                    | (static_cast<std::uint16_t>(high->data) << 8));
                input[lane][column] = MxmArray::Supercell::InputWord {
                    bits,
                    column + 1 == hw::kMxmSupercellColumns,
                };
            }
        }
        return input;
    }

    bool has_complete_vxm_input(std::size_t tile) const
    {
        const auto& required_streams = vxm_.required_streams_at(tile);
        if (!required_streams.has_value()) {
            return false;
        }

        for (std::size_t group = 0;
             group < VxmLane::kStreamGroupCount;
             ++group) {
            const auto base = group * VxmLane::kStreamGroupBytes;
            auto required_group = false;
            for (std::size_t byte = 0;
                 byte < VxmLane::kStreamGroupBytes;
                 ++byte) {
                required_group = required_group
                    || (*required_streams)[base + byte];
            }
            if (!required_group) {
                continue;
            }

            const auto source = vxm_.input_group_source(group);
            const auto& mem = mems_[hemisphere_index(source)];
            for (std::size_t lane = 0;
                 lane < hw::kLanesPerTile;
                 ++lane) {
                for (std::size_t byte = 0;
                     byte < VxmLane::kStreamGroupBytes;
                     ++byte) {
                    if (!mem.west_register(
                            tile, lane,
                            hw::kMemWestBoundaryStreamRegisterColumn,
                            base + byte)
                             .has_value()) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    bool vxm_requires_stream_from(
        Hemisphere source,
        std::size_t tile,
        std::size_t stream) const
    {
        const auto& required = vxm_.required_streams_at(tile);
        if (!required.has_value() || !(*required)[stream]) {
            return false;
        }
        const auto group = stream / VxmLane::kStreamGroupBytes;
        return vxm_.input_group_source(group) == source;
    }

    void transfer_mem_edges_to_vxm(LogSinks sinks)
    {
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            if (!has_complete_vxm_input(tile)) {
                continue;
            }

            const auto& required = *vxm_.required_streams_at(tile);
            auto required_groups = std::size_t {0};
            for (std::size_t group = 0;
                 group < VxmLane::kStreamGroupCount;
                 ++group) {
                const auto base = group * VxmLane::kStreamGroupBytes;
                if (required[base] || required[base + 1]) {
                    ++required_groups;
                }
            }
            vxm_.configure_input_buffer(tile, required_groups);

            for (std::size_t group = 0;
                 group < VxmLane::kStreamGroupCount;
                 ++group) {
                const auto base = group * VxmLane::kStreamGroupBytes;
                if (!required[base] && !required[base + 1]) {
                    continue;
                }

                const auto source = vxm_.input_group_source(group);
                auto& mem = mems_[hemisphere_index(source)];
                auto values = VxmSlice::InputBuffer::GroupVector {};
                for (std::size_t lane = 0;
                     lane < hw::kLanesPerTile;
                     ++lane) {
                    for (std::size_t byte = 0;
                         byte < VxmLane::kStreamGroupBytes;
                         ++byte) {
                        const auto cell = mem.consume_west_register(
                            tile, lane,
                            hw::kMemWestBoundaryStreamRegisterColumn,
                            base + byte);
                        if (!cell.has_value()) {
                            throw std::logic_error(
                                "VXM input group became incomplete during MEM-edge capture");
                        }
                        values[lane][byte] = cell->data;
                    }
                }
                vxm_.capture_stream_group(tile, group, values);
            }

            if (sinks.vxm != nullptr
                && (!sinks.vxm_log_tile.has_value()
                    || tile == *sinks.vxm_log_tile)) {
                *sinks.vxm << "  MEM.edge -> VXM tile " << tile
                           << " groups=" << required_groups << '\n';
            }
        }
    }

    void transfer_unconsumed_streams_across_vxm(LogSinks sinks)
    {
        for (std::size_t source_index = 0; source_index < hw::kHemispheres; ++source_index) {
            const auto source = static_cast<Hemisphere>(source_index);
            const auto destination_index = source_index ^ 1;
            auto& destination = mems_[destination_index];
            const auto& source_mem = mems_[source_index];
            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
                for (std::size_t stream = 0; stream < hw::kWestStreams; ++stream) {
                    if (vxm_requires_stream_from(source, tile, stream)) {
                        continue;
                    }

                    auto complete = true;
                    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                        complete = complete
                            && source_mem.west_register(
                                tile, lane,
                                hw::kMemWestBoundaryStreamRegisterColumn,
                                stream)
                                   .has_value();
                    }
                    if (!complete) continue;

                    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                        const auto& cell = source_mem.west_register(
                            tile, lane,
                            hw::kMemWestBoundaryStreamRegisterColumn,
                            stream);
                        destination.set_east_stream_input(
                            tile,
                            lane,
                            stream,
                            TileArrayModel::DataWord {cell->data, cell->last});
                    }
                    if (sinks.system != nullptr) {
                        *sinks.system << "  passive VXM bridge "
                                      << hemisphere_short_name(source) << ".W" << stream
                                      << " -> "
                                      << hemisphere_short_name(
                                             static_cast<Hemisphere>(destination_index))
                                      << ".E" << stream << " tile " << tile << '\n';
                    }
                }
            }
        }
    }

    void transfer_vxm_to_mem_edges(LogSinks sinks)
    {
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            for (const auto& output : vxm_.outputs_at(tile)) {
                if (output.stream + output.byte_count
                    > hw::kStreamsPerDirection) {
                    throw std::out_of_range(
                        "VXM output is outside the fixed 32-byte stream set");
                }
                const auto destination =
                    vxm_.output_stream_destination(output.stream);
                auto& mem = mems_[hemisphere_index(destination)];
                for (std::size_t byte = 0; byte < output.byte_count; ++byte) {
                    const auto stream = output.stream + byte;
                    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                        const auto word = TileArrayModel::DataWord {
                            output.byte_values[lane][byte],
                            lane + 1 == hw::kLanesPerTile,
                        };
                        // Both MEM hemispheres use the same local orientation:
                        // VXM is at sreg0 and its results travel eastward.
                        mem.set_east_stream_input(tile, lane, stream, word);
                    }
                }
                if (sinks.mem != nullptr && (!sinks.mem_log_tile.has_value() || tile == *sinks.mem_log_tile)) {
                    *sinks.mem << "  VXM -> MEM." << hemisphere_short_name(destination)
                               << " tile " << tile << " stream " << output.stream
                               << " bytes=" << output.byte_count << '\n';
                }
            }
        }
    }

    class NullStream {
    public:
        std::ostream& stream()
        {
            return stream_;
        }

    private:
        class Buffer : public std::streambuf {
        public:
            int overflow(int c) override
            {
                return c;
            }
        };

        Buffer buffer_{};
        std::ostream stream_{&buffer_};
    };

    std::array<TileArrayModel, hw::kHemispheres> mems_{};
    VxmSlice vxm_{};
    std::array<SxmSlice, hw::kHemispheres> sxms_;
    std::array<Mxm, kMxmCount> mxms_{};
    InstructionControlUnit icu_{};
    std::size_t cycle_{0};
    TimingObserver timing_observer_{};
    CyclePhase phase_{CyclePhase::Idle};
};

} // namespace ftlpu
