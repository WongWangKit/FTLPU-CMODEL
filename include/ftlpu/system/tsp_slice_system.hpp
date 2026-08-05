#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/icu/icu.hpp"
#include "ftlpu/mem/tile_array.hpp"
#include "ftlpu/mxm/accumulator_slice.hpp"
#include "ftlpu/mxm/activation_dequantizer.hpp"
#include "ftlpu/mxm/mxm.hpp"
#include "ftlpu/sxm/slice.hpp"
#include "ftlpu/vxm/slice.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <stdexcept>

namespace ftlpu {

class TspSliceSystem {
public:
    static constexpr std::size_t kMxmCount = hw::kMxmCount;

    explicit TspSliceSystem(
        std::size_t barrier_latency_cycles = hw::kIcuBarrierLatencyCycles)
        : mems_ {
            TileArrayModel(MemStreamPortMap::BetweenBoundaries()),
            TileArrayModel(MemStreamPortMap::BetweenBoundaries()),
        }
        , sxms_ {
            SxmSlice(SxmStreamPortMap::SameDirection(
                0,
                0)),
            SxmSlice(SxmStreamPortMap::SameDirection(
                hw::kMemBoundaryStreamRegisterColumns - 1,
                hw::kMemBoundaryStreamRegisterColumns - 1)),
        }
        , icu_(barrier_latency_cycles)
    {
        static_assert(
            hw::kHemispheres == 2,
            "TspSliceSystem SXM construction currently expects two hemispheres");
        static_assert(
            hw::kWestMxmCount <= 2 && hw::kEastMxmCount <= 2,
            "one shared MXM accumulator supports at most two MXMs per hemisphere");
        constexpr auto kWestMxmBoundary = std::size_t{0};
        constexpr auto kEastMxmBoundary =
            hw::kMemBoundaryStreamRegisterColumns - 1;
        accumulators_[hemisphere_index(Hemisphere::West)].set_endpoint(
            MxmAccumulatorSlice::Endpoint {
                kWestMxmBoundary, StreamDirection::East});
        accumulators_[hemisphere_index(Hemisphere::East)].set_endpoint(
            MxmAccumulatorSlice::Endpoint {
                kEastMxmBoundary, StreamDirection::West});
        for (std::size_t mxm = 0; mxm < kMxmCount; ++mxm) {
            const auto west = mxm < hw::kWestMxmCount;
            const auto local_mxm =
                west ? mxm : mxm - hw::kWestMxmCount;
            const auto weight_direction = west
                ? StreamDirection::West : StreamDirection::East;
            const auto mxm_boundary = west
                ? kWestMxmBoundary : kEastMxmBoundary;
            mxms_[mxm].set_stream_ports(MxmStreamPortMap {
                MxmStreamPortMap::InputEndpoint {
                    mxm_boundary,
                    weight_direction,
                    local_mxm * hw::kMxmLoadStreamsPerCycle},
            });
            activation_dequantizers_[mxm] =
                MxmActivationDequantizer(
                    MxmActivationDequantizer::Endpoint {
                        mxm_boundary,
                        weight_direction,
                        MxmActivationDequantizer::
                            kInputStreamWithinWindow
                                + local_mxm
                                    * hw::kMxmLoadStreamsPerCycle,
                        weight_direction,
                        local_mxm * hw::kMxmLoadStreamsPerCycle,
                        false});
        }
    }

    struct LogSinks {
        std::ostream* icu{nullptr};
        std::ostream* mem{nullptr};
        std::ostream* mxm{nullptr};
        std::ostream* vxm{nullptr};
        std::ostream* system{nullptr};
        std::optional<std::size_t> mem_log_tile{};
        std::optional<std::size_t> mxm_log_tile{};
        std::optional<std::size_t> vxm_log_tile{};
        bool detailed_system_stages{false};
    };

    TileArrayModel& mem()
    {
        return mem(Hemisphere::East);
    }

    const TileArrayModel& mem() const
    {
        return mem(Hemisphere::East);
    }

    TileArrayModel& mem(Hemisphere hemisphere)
    {
        return mems_.at(hemisphere_index(hemisphere));
    }

    const TileArrayModel& mem(Hemisphere hemisphere) const
    {
        return mems_.at(hemisphere_index(hemisphere));
    }

    VxmSlice& vxm()
    {
        return vxm_;
    }

    const VxmSlice& vxm() const
    {
        return vxm_;
    }

    MxmArray& mxm()
    {
        return mxms_[0].array();
    }

    const MxmArray& mxm() const
    {
        return mxms_[0].array();
    }

    MxmControlSlice& mxm_control()
    {
        return mxms_[0].control();
    }

    Mxm& mxm_unit(std::size_t index)
    {
        check_mxm_index(index);
        return mxms_[index];
    }

    const Mxm& mxm_unit(std::size_t index) const
    {
        check_mxm_index(index);
        return mxms_[index];
    }

    MxmActivationDequantizer& mxm_activation_dequantizer(
        std::size_t index)
    {
        check_mxm_index(index);
        return *activation_dequantizers_[index];
    }

    const MxmActivationDequantizer& mxm_activation_dequantizer(
        std::size_t index) const
    {
        check_mxm_index(index);
        return *activation_dequantizers_[index];
    }

    MxmAccumulatorSlice& mxm_accumulator(Hemisphere hemisphere)
    {
        return accumulators_.at(hemisphere_index(hemisphere));
    }

    const MxmAccumulatorSlice& mxm_accumulator(
        Hemisphere hemisphere) const
    {
        return accumulators_.at(hemisphere_index(hemisphere));
    }

    void configure_mxm_output_dequant_scale(
        std::size_t mxm,
        float activation_scale,
        float weight_scale)
    {
        check_mxm_index(mxm);
        const auto hemisphere = mxm_hemisphere(mxm);
        const auto local_path = hemisphere == Hemisphere::West
            ? mxm : mxm - hw::kWestMxmCount;
        accumulators_.at(hemisphere_index(hemisphere))
            .configure_output_dequant_scale(
                local_path, activation_scale, weight_scale);
    }

    InstructionControlUnit& icu()
    {
        return icu_;
    }

    const InstructionControlUnit& icu() const
    {
        return icu_;
    }

    SxmSlice& sxm(Hemisphere hemisphere = Hemisphere::East)
    {
        return sxms_.at(hemisphere_index(hemisphere));
    }

    const SxmSlice& sxm(
        Hemisphere hemisphere = Hemisphere::East) const
    {
        return sxms_.at(hemisphere_index(hemisphere));
    }

    // Selects the legacy direct stream-group capture bridge.  A modeled
    // token-major MXM->VXM input Buffer owns the same VXM entrance and must
    // disable this scanner so that one compiler-selected controller is the
    // sole writer of each Bundle.
    void set_vxm_stream_capture_enabled(bool enabled) noexcept
    {
        vxm_stream_capture_enabled_ = enabled;
    }

    bool vxm_stream_capture_enabled() const noexcept
    {
        return vxm_stream_capture_enabled_;
    }

    void tick(std::ostream& os)
    {
        LogSinks sinks {&os, &os, &os, &os, &os};
        tick(sinks, true);
    }

    void tick(LogSinks sinks, bool dispatch_icu = true)
    {
        if (sinks.system != nullptr) {
            *sinks.system << "system cycle " << cycle_ << '\n';
        }
        for (auto& mem : mems_) {
            mem.begin_cycle();
        }
        log_system_stage(sinks, "begin_cycle");
        if (dispatch_icu) {
            icu_.dispatch(
                mems_, vxm_, sxms_, mxms_, activation_dequantizers_,
                sinks.icu);
            log_system_stage(sinks, "dispatch");
        }
        for (std::size_t mxm = 0; mxm < kMxmCount; ++mxm) {
            activation_dequantizers_[mxm]->evaluate(
                mem(mxm_hemisphere(mxm)).stream_fabric());
        }
        log_system_stage(sinks, "mxm_activation_dequantize");
        tick_mxm_controls(sinks);
        tick_mxm_datapaths(sinks);
        tick_mxm_accumulators();
        log_system_stage(sinks, "mxm");
        for (std::size_t hemisphere = 0;
             hemisphere < hw::kHemispheres;
             ++hemisphere) {
            sxms_[hemisphere].evaluate(
                mems_[hemisphere].stream_fabric());
        }
        log_system_stage(sinks, "sxm");
        vxm_.prepare_cycle();
        auto vxm_buffered_before =
            std::array<bool, hw::kTileRows>{};
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            vxm_buffered_before[tile] =
                vxm_.input_buffer(tile).ready();
        }
        if (vxm_stream_capture_enabled_) {
            capture_vxm_input_groups(sinks);
        }
        vxm_.tick(sinks.vxm, sinks.vxm_log_tile);
        // In steady state, a Bundle that was already buffered is issued in
        // the first half of the cycle and the current SR values refill the
        // same lane-banked registers at the cycle edge.  This is a fixed
        // two-phase register update, not a ready/backpressure handshake.
        if (vxm_stream_capture_enabled_) {
            capture_vxm_input_groups(sinks, &vxm_buffered_before);
        }
        transfer_vxm_to_stream_registers(sinks);
        log_system_stage(sinks, "vxm");
        icu_.advance_barrier_events();
        log_system_stage(sinks, "barrier");
        for (std::size_t hemisphere = 0;
             hemisphere < hw::kHemispheres;
             ++hemisphere) {
            if (sinks.mem != nullptr) {
                *sinks.mem << "mem."
                           << hemisphere_short_name(
                                  static_cast<Hemisphere>(
                                      hemisphere))
                           << " system_cycle " << cycle_ << '\n';
                mems_[hemisphere].tick(
                    *sinks.mem, sinks.mem_log_tile);
            } else {
                mems_[hemisphere].tick();
            }
        }
        log_system_stage(sinks, "mem");
        ++cycle_;
    }

    void dispatch_icu_only(std::ostream* os = nullptr)
    {
        icu_.dispatch(
            mems_, vxm_, sxms_, mxms_, activation_dequantizers_, os);
    }

    void tick_mxm_controls_only(LogSinks sinks)
    {
        mem().begin_cycle();
        tick_mxm_controls(sinks);
    }

    void tick_mxm_datapaths_only(LogSinks sinks)
    {
        mem().begin_cycle();
        tick_mxm_datapaths(sinks);
        tick_mxm_accumulators();
    }

    void tick_vxm_stream_bridge(LogSinks sinks, std::optional<std::size_t> log_tile = std::nullopt)
    {
        if (log_tile.has_value()) {
            sinks.vxm_log_tile = log_tile;
        }
        vxm_.prepare_cycle();
        auto vxm_buffered_before =
            std::array<bool, hw::kTileRows>{};
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            vxm_buffered_before[tile] =
                vxm_.input_buffer(tile).ready();
        }
        capture_vxm_input_groups(sinks);
        vxm_.tick(sinks.vxm, sinks.vxm_log_tile);
        capture_vxm_input_groups(sinks, &vxm_buffered_before);
        transfer_vxm_to_stream_registers(sinks);
    }

    std::size_t cycle() const
    {
        return cycle_;
    }

private:
    static void log_system_stage(
        const LogSinks& sinks,
        const char* stage)
    {
        if (sinks.detailed_system_stages
            && sinks.system != nullptr) {
            *sinks.system << "  completed stage "
                          << stage << '\n';
        }
    }

    static void check_mxm_index(std::size_t index)
    {
        if (index >= kMxmCount) {
            throw std::out_of_range("MXM index is outside the configured MXM units");
        }
    }

    static constexpr Hemisphere mxm_hemisphere(
        std::size_t mxm) noexcept
    {
        return mxm < hw::kWestMxmCount
            ? Hemisphere::West
            : Hemisphere::East;
    }

    void tick_mxm_controls(LogSinks sinks)
    {
        for (std::size_t mxm = 0; mxm < kMxmCount; ++mxm) {
            if (sinks.mxm != nullptr) {
                *sinks.mxm << "mxm" << mxm << " cycle " << cycle_ << '\n';
            }
            mxms_[mxm].evaluate_control(
                mem(mxm_hemisphere(mxm)).stream_fabric(),
                mxm,
                sinks.mxm,
                sinks.mxm_log_tile);
        }
    }

    void tick_mxm_datapaths(LogSinks sinks)
    {
        for (std::size_t mxm = 0; mxm < kMxmCount; ++mxm) {
            mxms_[mxm].evaluate_datapath(
                mem(mxm_hemisphere(mxm)).stream_fabric(),
                mxm,
                sinks.mxm,
                sinks.mxm_log_tile);
        }
    }

    void tick_mxm_accumulators()
    {
        static const auto empty = std::vector<Mxm::ColumnOutput>{};
        for (std::size_t hemisphere = 0;
             hemisphere < hw::kHemispheres;
             ++hemisphere) {
            const auto side = static_cast<Hemisphere>(hemisphere);
            const auto start = side == Hemisphere::West
                ? std::size_t{0} : hw::kWestMxmCount;
            const auto count = side == Hemisphere::West
                ? hw::kWestMxmCount : hw::kEastMxmCount;
            const auto& first = count > 0
                ? mxms_[start].last_outputs() : empty;
            const auto& second = count > 1
                ? mxms_[start + 1].last_outputs() : empty;
            accumulators_[hemisphere].evaluate(
                mem(side).stream_fabric(), first, second);
        }
    }

    void capture_vxm_input_groups(
        LogSinks sinks,
        const std::array<bool, hw::kTileRows>* tile_mask = nullptr)
    {
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            if (tile_mask != nullptr && !(*tile_mask)[tile]) {
                continue;
            }
            const auto& required = vxm_.required_streams_at(tile);
            if (!required.has_value()) {
                continue;
            }

            auto& buffer = vxm_.input_buffer(tile);
            if (buffer.ready()) {
                continue;
            }

            auto required_group_count = std::size_t{0};
            for (std::size_t group = 0;
                 group < VxmInputBuffer::kGroupCount;
                 ++group) {
                const auto local_base =
                    group * VxmInputBuffer::kGroupBytes;
                const auto east_low =
                    (*required)[local_base];
                const auto east_high =
                    (*required)[local_base + 1];
                const auto east_required =
                    east_low || east_high;
                const auto west_base =
                    hw::kStreamsPerDirection + local_base;
                const auto west_low =
                    (*required)[west_base];
                const auto west_high =
                    (*required)[west_base + 1];
                const auto west_required =
                    west_low || west_high;
                if (east_low != east_high
                    || west_low != west_high) {
                    throw std::logic_error(
                        "VXM FP16 input requires both bytes of a fixed stream group");
                }
                if (east_required && west_required) {
                    throw std::logic_error(
                        "VXM input group cannot collect both hemispheres into one fixed position");
                }
                if (east_required || west_required) {
                    ++required_group_count;
                }
            }
            if (required_group_count == 0) {
                continue;
            }

            if (buffer.empty()) {
                vxm_.configure_input_buffer(
                    tile, required_group_count);
            } else if (
                buffer.expected_count()
                != required_group_count) {
                throw std::logic_error(
                    "VXM stream requirements changed while its input Buffer was collecting");
            }

            for (std::size_t group = 0;
                 group < VxmInputBuffer::kGroupCount;
                 ++group) {
                if (buffer.has_group(group)) {
                    continue;
                }
                const auto local_base =
                    group * VxmInputBuffer::kGroupBytes;
                const auto east_low =
                    (*required)[local_base];
                const auto east_high =
                    (*required)[local_base + 1];
                const auto east_required =
                    east_low || east_high;
                const auto west_base =
                    hw::kStreamsPerDirection + local_base;
                const auto west_low =
                    (*required)[west_base];
                const auto west_high =
                    (*required)[west_base + 1];
                const auto west_required =
                    west_low || west_high;
                if (!east_required && !west_required) {
                    continue;
                }

                const auto hemisphere = east_required
                    ? Hemisphere::East
                    : Hemisphere::West;
                const auto direction =
                    hemisphere == Hemisphere::East
                    ? StreamDirection::West
                    : StreamDirection::East;
                const auto column =
                    hemisphere == Hemisphere::East
                    ? std::size_t{0}
                    : hw::kMemBoundaryStreamRegisterColumns
                        - 1;
                auto& fabric =
                    mem(hemisphere).stream_fabric();
                const auto make_stream =
                    [direction](std::size_t stream) {
                        return direction
                                == StreamDirection::East
                            ? StreamId::East(stream)
                            : StreamId::West(stream);
                    };
                const auto low = make_stream(local_base);
                const auto high =
                    make_stream(local_base + 1);
                if (!fabric.segment_valid(
                        column, tile, low)
                    || !fabric.segment_valid(
                        column, tile, high)) {
                    continue;
                }

                auto values =
                    VxmInputBuffer::GroupVector{};
                for (std::size_t lane = 0;
                     lane < hw::kLanesPerTile;
                     ++lane) {
                    values[lane][0] =
                        fabric.cell(
                            column, tile, lane, low)
                            .data;
                    values[lane][1] =
                        fabric.cell(
                            column, tile, lane, high)
                            .data;
                }
                vxm_.capture_stream_group(
                    tile, group, values);
                fabric.consume_segment(
                    column, tile, low,
                    "VXM input Buffer");
                fabric.consume_segment(
                    column, tile, high,
                    "VXM input Buffer");

                if (sinks.vxm != nullptr
                    && (!sinks.vxm_log_tile.has_value()
                        || tile
                            == *sinks.vxm_log_tile)) {
                    *sinks.vxm
                        << "  SR -> VXM input Buffer tile "
                        << tile
                        << " group " << group
                        << " fill "
                        << buffer.fill_count()
                        << '/'
                        << buffer.expected_count()
                        << '\n';
                }
            }
        }
    }

    void transfer_vxm_to_stream_registers(LogSinks sinks)
    {
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            for (const auto& output : vxm_.outputs_at(tile)) {
                if (output.stream + output.byte_count > hw::kStreams) {
                    throw std::out_of_range(
                        "VXM output exceeds the 64-stream interface");
                }
                const auto hemisphere =
                    vxm_.output_stream_destination(output.stream);
                const auto local_stream = output.stream;
                if (local_stream + output.byte_count
                    > hw::kStreamsPerDirection) {
                    throw std::out_of_range(
                        "VXM output crosses a hemisphere stream boundary");
                }
                auto& destination_mem =
                    mem(hemisphere);
                for (std::size_t byte = 0; byte < output.byte_count; ++byte) {
                    const auto stream = local_stream + byte;
                    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                        const auto word = TileArrayModel::DataWord {
                            output.byte_values[lane][byte],
                            lane + 1 == hw::kLanesPerTile,
                        };
                        if (hemisphere == Hemisphere::East) {
                            destination_mem
                                .set_east_stream_input(
                                    tile,
                                    lane,
                                    stream,
                                    word);
                        } else {
                            destination_mem
                                .set_west_stream_input(
                                    tile,
                                    lane,
                                    stream,
                                    word);
                        }
                    }
                }
                if (sinks.mem != nullptr && (!sinks.mem_log_tile.has_value() || tile == *sinks.mem_log_tile)) {
                    *sinks.mem << "  VXM -> SR."
                               << hemisphere_short_name(
                                      hemisphere)
                               << " tile " << tile
                               << " stream " << output.stream
                               << " bytes=" << output.byte_count << '\n';
                }
            }
        }
    }

    std::array<TileArrayModel, hw::kHemispheres> mems_{};
    VxmSlice vxm_{};
    std::array<SxmSlice, hw::kHemispheres> sxms_;
    std::array<Mxm, kMxmCount> mxms_{};
    std::array<std::optional<MxmActivationDequantizer>, kMxmCount>
        activation_dequantizers_{};
    std::array<MxmAccumulatorSlice, hw::kHemispheres> accumulators_{};
    InstructionControlUnit icu_{};
    std::size_t cycle_{0};
    bool vxm_stream_capture_enabled_{true};
};

} // namespace ftlpu
