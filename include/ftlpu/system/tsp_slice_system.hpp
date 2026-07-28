#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/icu/icu.hpp"
#include "ftlpu/mem/tile_array.hpp"
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
        : sxms_ {
            SxmSlice(SxmStreamPortMap::SameDirection(
                hw::kMemBoundaryStreamRegisterColumns - 1,
                hw::kMemBoundaryStreamRegisterColumns - 1)),
            SxmSlice(SxmStreamPortMap::SameDirection(
                hw::kMemBoundaryStreamRegisterColumns - 1,
                hw::kMemBoundaryStreamRegisterColumns - 1)),
        }
        , icu_(barrier_latency_cycles)
    {
        static_assert(
            hw::kHemispheres == 2,
            "TspSliceSystem SXM construction currently expects two hemispheres");
        constexpr auto kMxmBoundary = hw::kMemBoundaryStreamRegisterColumns - 1;
        for (std::size_t mxm = 0; mxm < kMxmCount; ++mxm) {
            const auto west = mxm < hw::kWestMxmCount;
            const auto local_mxm =
                west ? mxm : mxm - hw::kWestMxmCount;
            const auto weight_direction = west
                ? StreamDirection::West : StreamDirection::East;
            const auto result_direction = west
                ? StreamDirection::East : StreamDirection::West;
            mxms_[mxm].set_stream_ports(MxmStreamPortMap {
                MxmStreamPortMap::WeightEndpoint {
                    {kMxmBoundary, weight_direction, false},
                    local_mxm * hw::kMxmLoadStreamsPerCycle},
                MxmStreamPortMap::InputEndpoint {
                    kMxmBoundary, weight_direction, true},
                MxmStreamPortMap::OutputEndpoint {
                    kMxmBoundary, result_direction},
            });
            fetch_ports_.map(IcuLocation::MxmLoad(mxm), kMxmBoundary);
            fetch_ports_.map(IcuLocation::MxmCompute(mxm), kMxmBoundary);
        }
        fetch_ports_.map(
            IcuLocation::Sxm(Hemisphere::East), kMxmBoundary);
        fetch_ports_.map(
            IcuLocation::Sxm(Hemisphere::West), kMxmBoundary);
        for (std::size_t mem_slice = 0;
             mem_slice < hw::kMemSliceColumns;
             ++mem_slice) {
            for (std::size_t hemisphere = 0;
                 hemisphere < hw::kHemispheres;
                 ++hemisphere) {
                const auto side =
                    static_cast<Hemisphere>(hemisphere);
                fetch_ports_.map(
                    IcuLocation::Mem(side, mem_slice),
                    mems_[hemisphere]
                        .memory_model()
                        .ports()
                        .output_column(
                            mem_slice,
                            StreamDirection::East));
            }
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

    IcuFetchPortMap& icu_fetch_ports() noexcept
    {
        return fetch_ports_;
    }

    const IcuFetchPortMap& icu_fetch_ports() const noexcept
    {
        return fetch_ports_;
    }

    void tick(std::ostream& os)
    {
        LogSinks sinks {&os, &os, &os, &os, &os};
        tick(sinks);
    }

    void tick(LogSinks sinks)
    {
        if (sinks.system != nullptr) {
            *sinks.system << "system cycle " << cycle_ << '\n';
        }
        for (auto& mem : mems_) {
            mem.begin_cycle();
        }
        log_system_stage(sinks, "begin_cycle");
        for (std::size_t hemisphere = 0;
             hemisphere < hw::kHemispheres;
             ++hemisphere) {
            const auto side =
                static_cast<Hemisphere>(hemisphere);
            icu_.evaluate_fetches(
                mems_[hemisphere].stream_fabric(),
                fetch_ports_,
                side);
            if (sinks.detailed_system_stages
                && sinks.system != nullptr) {
                *sinks.system
                    << "  completed stream ifetch "
                    << hemisphere_short_name(side)
                    << '\n';
            }
            icu_.evaluate_mem_local_fetches(
                mems_[hemisphere].memory_model(),
                side);
            if (sinks.detailed_system_stages
                && sinks.system != nullptr) {
                *sinks.system
                    << "  completed local ifetch "
                    << hemisphere_short_name(side)
                    << '\n';
            }
        }
        log_system_stage(sinks, "ifetch");
        icu_.dispatch(mems_, vxm_, sxms_, mxms_, sinks.icu);
        log_system_stage(sinks, "dispatch");
        tick_mxm_controls(sinks);
        tick_mxm_datapaths(sinks);
        log_system_stage(sinks, "mxm");
        for (std::size_t hemisphere = 0;
             hemisphere < hw::kHemispheres;
             ++hemisphere) {
            sxms_[hemisphere].evaluate(
                mems_[hemisphere].stream_fabric());
        }
        log_system_stage(sinks, "sxm");
        vxm_.prepare_cycle();
        transfer_mem_west_to_vxm(sinks);
        transfer_unconsumed_streams_across_vxm(sinks);
        vxm_.tick(sinks.vxm, sinks.vxm_log_tile);
        transfer_vxm_to_mem_east(sinks);
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
        icu_.commit_fetches();
        log_system_stage(sinks, "commit_fetch");
        ++cycle_;
    }

    void dispatch_icu_only(std::ostream* os = nullptr)
    {
        icu_.dispatch(mems_, vxm_, sxms_, mxms_, os);
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
    }

    void tick_vxm_stream_bridge(LogSinks sinks, std::optional<std::size_t> log_tile = std::nullopt)
    {
        if (log_tile.has_value()) {
            sinks.vxm_log_tile = log_tile;
        }
        vxm_.prepare_cycle();
        transfer_mem_west_to_vxm(sinks);
        transfer_unconsumed_streams_across_vxm(sinks);
        vxm_.tick(sinks.vxm, sinks.vxm_log_tile);
        transfer_vxm_to_mem_east(sinks);
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

    bool has_complete_vxm_input(
        Hemisphere hemisphere,
        std::size_t tile) const
    {
        const auto& required_streams =
            vxm_.required_streams_at(hemisphere, tile);
        if (!required_streams.has_value()) {
            return false;
        }

        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            for (std::size_t stream = 0; stream < hw::kStreams; ++stream) {
                if (!(*required_streams)[stream]) {
                    continue;
                }
                if (!mem_edge_stream(
                        mem(hemisphere),
                        tile,
                        lane,
                        stream)
                         .has_value()) {
                    return false;
                }
            }
        }
        return true;
    }

    void transfer_mem_west_to_vxm(LogSinks sinks)
    {
        for (std::size_t hemisphere_index_value = 0;
             hemisphere_index_value < hw::kHemispheres;
             ++hemisphere_index_value) {
            const auto hemisphere =
                static_cast<Hemisphere>(
                    hemisphere_index_value);
            for (std::size_t tile = 0;
                 tile < hw::kTileRows;
                 ++tile) {
                if (!has_complete_vxm_input(hemisphere, tile)) {
                    continue;
                }

                auto streams = VxmSlice::StreamMatrix {};
                for (std::size_t lane = 0;
                     lane < hw::kLanesPerTile;
                     ++lane) {
                    for (std::size_t stream = 0;
                         stream < hw::kStreams;
                         ++stream) {
                        const auto& slot = mem_edge_stream(
                            mem(hemisphere),
                            tile,
                            lane,
                            stream);
                        streams[lane][stream] =
                            slot.has_value() ? slot->data : 0;
                    }
                }
                vxm_.set_stream_inputs(
                    hemisphere, tile, streams);
                if (sinks.vxm != nullptr
                    && (!sinks.vxm_log_tile.has_value()
                        || tile == *sinks.vxm_log_tile)) {
                    *sinks.vxm << "  MEM."
                               << hemisphere_short_name(hemisphere)
                               << ".edge -> VXM tile "
                               << tile << '\n';
                }
            }
        }
    }

    void transfer_unconsumed_streams_across_vxm(LogSinks sinks)
    {
        for (std::size_t source_index = 0;
             source_index < hw::kHemispheres;
             ++source_index) {
            const auto source =
                static_cast<Hemisphere>(source_index);
            const auto destination =
                static_cast<Hemisphere>(source_index ^ 1);
            for (std::size_t tile = 0;
                 tile < hw::kTileRows;
                 ++tile) {
                const auto& required =
                    vxm_.required_streams_at(source, tile);
                for (std::size_t west_stream = 0;
                     west_stream < hw::kWestStreams;
                     ++west_stream) {
                    const auto packed =
                        hw::kEastStreams + west_stream;
                    if (required.has_value()
                        && (*required)[packed]) {
                        continue;
                    }
                    auto complete = true;
                    for (std::size_t lane = 0;
                         lane < hw::kLanesPerTile;
                         ++lane) {
                        complete = complete
                            && mem_edge_stream(
                                   mem(source),
                                   tile,
                                   lane,
                                   packed)
                                   .has_value();
                    }
                    if (!complete) continue;
                    for (std::size_t lane = 0;
                         lane < hw::kLanesPerTile;
                         ++lane) {
                        const auto& cell =
                            mem_edge_stream(
                                mem(source),
                                tile,
                                lane,
                                packed);
                        mem(destination)
                            .set_east_stream_input(
                                tile,
                                lane,
                                west_stream,
                                TileArrayModel::DataWord {
                                    cell->data,
                                    cell->last});
                    }
                    if (sinks.system != nullptr) {
                        *sinks.system
                            << "  passive VXM bridge "
                            << hemisphere_short_name(source)
                            << ".W" << west_stream
                            << " -> "
                            << hemisphere_short_name(destination)
                            << ".E" << west_stream
                            << " tile " << tile << '\n';
                    }
                }
            }
        }
    }

    void transfer_vxm_to_mem_east(LogSinks sinks)
    {
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
            for (const auto& output : vxm_.outputs_at(tile)) {
                if (output.stream + output.byte_count > hw::kStreams) {
                    throw std::out_of_range("VXM output stream is outside the 64-stream lane");
                }
                auto& destination_mem =
                    mem(output.hemisphere);
                for (std::size_t byte = 0; byte < output.byte_count; ++byte) {
                    const auto stream = output.stream + byte;
                    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                        const auto word = TileArrayModel::DataWord {
                            output.byte_values[lane][byte],
                            lane + 1 == hw::kLanesPerTile,
                        };
                        if (stream < hw::kEastStreams) {
                            destination_mem.set_east_stream_input(
                                    tile, lane, stream, word);
                        } else {
                            destination_mem.set_west_stream_input(
                                    tile,
                                    lane,
                                    stream - hw::kEastStreams,
                                    word);
                        }
                    }
                }
                if (sinks.mem != nullptr && (!sinks.mem_log_tile.has_value() || tile == *sinks.mem_log_tile)) {
                    *sinks.mem << "  VXM -> MEM."
                               << hemisphere_short_name(
                                      output.hemisphere)
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
    InstructionControlUnit icu_{};
    IcuFetchPortMap fetch_ports_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
