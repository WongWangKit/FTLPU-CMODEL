#pragma once

#include "ftlpu/c2c/dma.hpp"
#include "ftlpu/c2c/link.hpp"
#include "ftlpu/c2c/slice.hpp"
#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/hemisphere.hpp"
#include "ftlpu/mem/tile_array.hpp"
#include "ftlpu/mxm/mxm.hpp"
#include "ftlpu/sxm/slice.hpp"
#include "ftlpu/system/hardware_configuration.hpp"
#include "ftlpu/system/icu.hpp"
#include "ftlpu/vxm/slice.hpp"
#include <algorithm>

#include <array>
#include <cstddef>
#include <cstdint>
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

    explicit TspSliceSystem(
        SystemHardwareConfiguration hardware = {})
        : hardware_configuration_(hardware)
        , mems_ {
            TileArrayModel(MemStreamPortMap::BetweenBoundaries()),
            TileArrayModel(MemStreamPortMap::BetweenBoundaries()),
        }
        , sxms_ {
            SxmSlice(make_sxm_port_map()),
            SxmSlice(make_sxm_port_map()),
        }
    {
        configure_hardware(hardware_configuration_);
    }

    void configure_hardware(SystemHardwareConfiguration hardware)
    {
        require_phase(CyclePhase::Idle, "configuring hardware");
        hardware.validate();
        const bool rebuild_c2c =
            hardware.c2c_dedicated_streams
                != hardware_configuration_.c2c_dedicated_streams;
        hardware_configuration_ = hardware;
        for (auto& mem : mems_)
            mem.set_sram_depth_rows(hardware.sram_depth_rows);
        if (rebuild_c2c) {
            for (std::size_t index = 0; index < hw::kHemispheres; ++index) {
                if (c2c_dmas_[index] == nullptr) continue;
                c2cs_[index].emplace(C2cStreamPortMap::EastEdge(
                    hw::kMemEastBoundaryStreamRegisterColumn), "C2C DMA",
                    hardware_configuration_.c2c_dedicated_streams, true);
            }
        }
    }

    const SystemHardwareConfiguration& hardware_configuration() const noexcept
    {
        return hardware_configuration_;
    }

    void attach_c2c(
        Hemisphere hemisphere,
        C2cLink& outbound_link,
        C2cLink& inbound_link)
    {
        require_phase(CyclePhase::Idle, "attaching C2C");
        const auto index = hemisphere_index(hemisphere);
        c2cs_[index].emplace(C2cStreamPortMap::EastEdge(
            hw::kMemEastBoundaryStreamRegisterColumn));
        c2c_outbound_links_[index] = &outbound_link;
        c2c_inbound_links_[index] = &inbound_link;
        c2c_dmas_[index] = nullptr;
    }

    void attach_c2c_dma(
        Hemisphere hemisphere,
        C2cDmaEngine& dma)
    {
        require_phase(CyclePhase::Idle, "attaching C2C DMA");
        const auto index = hemisphere_index(hemisphere);
        c2cs_[index].emplace(C2cStreamPortMap::EastEdge(
            hw::kMemEastBoundaryStreamRegisterColumn), "C2C DMA",
            hardware_configuration_.c2c_dedicated_streams, true);
        c2c_outbound_links_[index] = nullptr;
        c2c_inbound_links_[index] = nullptr;
        c2c_dmas_[index] = &dma;
    }

    C2cDmaEngine& c2c_dma(Hemisphere hemisphere)
    {
        auto* dma = c2c_dmas_[hemisphere_index(hemisphere)];
        if (dma == nullptr) {
            throw std::logic_error("TSP hemisphere has no C2C DMA");
        }
        return *dma;
    }

    const C2cDmaEngine& c2c_dma(Hemisphere hemisphere) const
    {
        return const_cast<TspSliceSystem*>(this)->c2c_dma(hemisphere);
    }

    bool has_c2c(Hemisphere hemisphere) const noexcept
    {
        return c2cs_[hemisphere_index(hemisphere)].has_value();
    }

    bool has_c2c() const noexcept
    {
        return std::any_of(
            c2cs_.begin(), c2cs_.end(),
            [](const auto& endpoint) { return endpoint.has_value(); });
    }

    void detach_c2c(Hemisphere hemisphere)
    {
        require_phase(CyclePhase::Idle, "detaching C2C");
        const auto index = hemisphere_index(hemisphere);
        c2cs_[index].reset();
        c2c_outbound_links_[index] = nullptr;
        c2c_inbound_links_[index] = nullptr;
        c2c_dmas_[index] = nullptr;
    }

    void detach_c2c()
    {
        require_phase(CyclePhase::Idle, "detaching all C2C endpoints");
        for (std::size_t index = 0; index < hw::kHemispheres; ++index) {
            c2cs_[index].reset();
            c2c_outbound_links_[index] = nullptr;
            c2c_inbound_links_[index] = nullptr;
            c2c_dmas_[index] = nullptr;
        }
    }

    C2cEndpoint& c2c_endpoint(Hemisphere hemisphere)
    {
        auto& endpoint = c2cs_[hemisphere_index(hemisphere)];
        if (!endpoint.has_value()) {
            throw std::logic_error(
                "TSP hemisphere has no attached C2C endpoint");
        }
        return *endpoint;
    }

    const C2cEndpoint& c2c_endpoint(Hemisphere hemisphere) const
    {
        const auto& endpoint = c2cs_[hemisphere_index(hemisphere)];
        if (!endpoint.has_value()) {
            throw std::logic_error(
                "TSP hemisphere has no attached C2C endpoint");
        }
        return *endpoint;
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

    void initialize_mem_sram_lane_byte(
        Hemisphere hemisphere,
        std::size_t column,
        std::size_t bank,
        std::size_t tile,
        std::size_t row,
        std::size_t lane,
        std::uint8_t value)
    {
        mems_[hemisphere_index(hemisphere)].set_sram_lane_byte(
            column, bank, tile, row, lane, value);
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

    std::uint8_t read_mem_sram_lane_byte(
        Hemisphere hemisphere,
        std::size_t column,
        std::size_t bank,
        std::size_t tile,
        std::size_t row,
        std::size_t lane) const
    {
        return mems_[hemisphere_index(hemisphere)].sram_lane_byte(
            column, bank, tile, row, lane);
    }

    InstructionControlUnit& icu()
    {
        return icu_;
    }

    const InstructionControlUnit& icu() const
    {
        return icu_;
    }

    const VxmSlice& vxm_unit() const noexcept
    {
        return vxm_;
    }

    VxmSlice& vxm_unit() noexcept
    {
        return vxm_;
    }

    void initialize_vxm_lut(
        VxmSpecialAluOpcode opcode,
        VxmLutConfig config,
        const std::vector<VxmLutEntry>& entries)
    {
        vxm_.configure_special_lut(opcode, config, entries);
    }

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

    const StreamRegisterFabric& stream_fabric(
        Hemisphere hemisphere) const noexcept
    {
        return mems_[hemisphere_index(hemisphere)].stream_fabric();
    }

    Mxm& mxm_unit(std::size_t mxm)
    {
        require_active_mxm(mxm);
        return mxms_.at(mxm);
    }

    const Mxm& mxm_unit(std::size_t mxm) const
    {
        require_active_mxm(mxm);
        return mxms_.at(mxm);
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
    }

    std::size_t cycle() const
    {
        return cycle_;
    }

    std::size_t passive_bridge_transfer_count(
        Hemisphere source, std::size_t stream) const
    {
        if (stream >= hw::kWestStreams)
            throw std::out_of_range("passive bridge stream is outside the westbound stream set");
        return passive_bridge_transfer_counts_[hemisphere_index(source)][stream];
    }

    std::optional<std::size_t> last_passive_bridge_cycle(
        Hemisphere source, std::size_t stream) const
    {
        if (stream >= hw::kWestStreams)
            throw std::out_of_range("passive bridge stream is outside the westbound stream set");
        const auto encoded =
            last_passive_bridge_cycles_[hemisphere_index(source)][stream];
        return encoded == 0
            ? std::nullopt
            : std::optional<std::size_t> {encoded - 1};
    }

    void reset_execution_state()
    {
        require_phase(CyclePhase::Idle, "resetting execution state");
        for (auto& mem : mems_) mem.reset_execution_state();
        vxm_.reset();
        for (auto& sxm : sxms_) sxm.reset();
        for (auto& mxm : mxms_) mxm.reset();
        icu_.reset();
        for (auto& counts : passive_bridge_transfer_counts_) counts.fill(0);
        for (auto& cycles : last_passive_bridge_cycles_) cycles.fill(0);
        for (auto& endpoint : c2cs_) {
            if (endpoint.has_value()) endpoint->reset();
        }
        cycle_ = 0;
        for (auto* dma : c2c_dmas_) {
            if (dma != nullptr) dma->reset();
        }
    }

private:
    void require_active_mxm(std::size_t mxm) const
    {
        if (mxm >= kMxmCount
            || mxm % hw::kMxmsPerHemisphere
                >= hardware_configuration_.mxms_per_hemisphere)
            throw std::out_of_range(
                "MXM unit is disabled by the system hardware configuration");
    }

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
        auto c2c_endpoints =
            std::array<C2cEndpoint*, hw::kHemispheres> {};
        auto c2c_dmas =
            std::array<C2cDmaEngine*, hw::kHemispheres> {};
        for (std::size_t hemisphere = 0;
             hemisphere < hw::kHemispheres;
             ++hemisphere) {
            c2c_endpoints[hemisphere] = c2cs_[hemisphere].has_value()
                ? &*c2cs_[hemisphere] : nullptr;
            c2c_dmas[hemisphere] = c2c_dmas_[hemisphere];
        }
        icu_.dispatch(
            mems_, vxm_, sxms_, mxms_, sinks.icu, c2c_endpoints, c2c_dmas);
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
            try {
                sxms_[hemisphere].set_trace_enabled(sinks.sxm != nullptr);
                if (sinks.mem != nullptr) {
                    *sinks.mem << "mem." << hemisphere_short_name(static_cast<Hemisphere>(hemisphere))
                               << " cycle " << cycle_ << '\n';
                    if (c2cs_[hemisphere].has_value()) {
                        const auto evaluate = [this, hemisphere](StreamRegisterFabric& fabric) {
                            evaluate_c2c(hemisphere, fabric);
                        };
                        mems_[hemisphere].tick(
                            sxms_[hemisphere], evaluate, *sinks.mem,
                            sinks.mem_log_tile);
                    } else {
                        mems_[hemisphere].tick(
                            sxms_[hemisphere], *sinks.mem,
                            sinks.mem_log_tile);
                    }
                } else {
                    if (c2cs_[hemisphere].has_value()) {
                        const auto evaluate = [this, hemisphere](StreamRegisterFabric& fabric) {
                            evaluate_c2c(hemisphere, fabric);
                        };
                        mems_[hemisphere].tick(sxms_[hemisphere], evaluate);
                    } else {
                        mems_[hemisphere].tick(sxms_[hemisphere]);
                    }
                }
            } catch (const std::exception& error) {
                throw std::logic_error(
                    "MEM/SXM commit failed at system cycle "
                    + std::to_string(cycle_) + " in hemisphere "
                    + hemisphere_short_name(static_cast<Hemisphere>(hemisphere))
                    + ": " + error.what());
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
        for (std::size_t hemisphere = 0;
             hemisphere < hw::kHemispheres;
             ++hemisphere) {
            auto* dma = c2c_dmas_[hemisphere];
            if (dma == nullptr) continue;
            dma->tick();
            if (dma->take_completion_notification()) {
                icu_.notify(IcuLocation::C2cDma(
                    static_cast<Hemisphere>(hemisphere)));
            }
        }
        icu_.advance_barrier_events();
        ++cycle_;
        phase_ = CyclePhase::Idle;
    }

    void evaluate_c2c(
        std::size_t hemisphere,
        StreamRegisterFabric& fabric)
    {
        auto& endpoint = c2cs_[hemisphere];
        if (!endpoint.has_value()) {
            throw std::logic_error("incomplete TSP C2C attachment");
        }

        std::optional<C2cReceiveNotification> notification;
        if (c2c_dmas_[hemisphere] != nullptr) {
            endpoint->tx().evaluate(fabric, *c2c_dmas_[hemisphere]);
            const auto notify = [this](C2cReceiveNotification received) {
                const auto& consumer = received.consumer;
                if (consumer.notify_mem) {
                    icu_.notify(IcuLocation::Mem(
                        consumer.hemisphere,
                        consumer.mem_slice,
                        consumer.mem_bank));
                }
            };
            if (hardware_configuration_.c2c_dedicated_streams) {
                endpoint->rx().evaluate_dedicated(
                    *c2c_dmas_[hemisphere],
                    hardware_configuration_.c2c_streams_per_direction,
                    [this, &notify](C2cReceiveNotification received) {
                        const auto& consumer = received.consumer;
                        for (std::size_t tile = 0; tile < hw::kTileRows;
                             ++tile) {
                            for (std::size_t lane = 0;
                                 lane < hw::kLanesPerTile; ++lane) {
                                mems_[hemisphere_index(consumer.hemisphere)]
                                    .set_sram_lane_byte(
                                        consumer.mem_slice,
                                        consumer.mem_bank,
                                        tile,
                                        consumer.base_row,
                                        lane,
                                        received.vector.payload[tile][lane]);
                            }
                        }
                        notify(std::move(received));
                    });
            } else {
                endpoint->rx().evaluate_shared(
                    fabric, *c2c_dmas_[hemisphere],
                    hardware_configuration_.c2c_streams_per_direction,
                    notify);
            }
        } else {
            if (c2c_outbound_links_[hemisphere] == nullptr
                || c2c_inbound_links_[hemisphere] == nullptr) {
                throw std::logic_error("incomplete TSP C2C link attachment");
            }
            notification = endpoint->evaluate(
                fabric,
                *c2c_outbound_links_[hemisphere],
                *c2c_inbound_links_[hemisphere]);
        }
        if (notification.has_value()
            && notification->consumer.notify_mem) {
            icu_.notify(IcuLocation::Mem(
                notification->consumer.hemisphere,
                notification->consumer.mem_slice,
                notification->consumer.mem_bank));
        }
    }

    static SxmStreamPortMap make_sxm_port_map()
    {
        return SxmStreamPortMap::BetweenColumns(
            hw::kC2cSxmBoundaryStreamRegisterColumn,
            hw::kMxmBoundaryStreamRegisterColumn,
            hw::kMxmBoundaryStreamRegisterColumn,
            hw::kC2cSxmBoundaryStreamRegisterColumn);
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
        if (instruction->weight_input_mode
            == MxmWeightInputMode::Int8DequantBf16) {
            const auto stream_base = instruction->weight_stream_base;
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
                            "MXM INT8 IW reached tile before all eight weight streams arrived at the MXM boundary register"
                            " (tile=" + std::to_string(tile)
                            + ", lane=" + std::to_string(lane)
                            + ", stream="
                            + std::to_string(stream_base + column) + ")");
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

        const auto stream_base = instruction->weight_stream_base;
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
            if (!(*required_streams)[base]
                && !(*required_streams)[base + 1]) {
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
                    ++passive_bridge_transfer_counts_[source_index][stream];
                    last_passive_bridge_cycles_[source_index][stream] = cycle_ + 1;
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

    SystemHardwareConfiguration hardware_configuration_{};
    std::array<TileArrayModel, hw::kHemispheres> mems_{};
    VxmSlice vxm_{};
    std::array<SxmSlice, hw::kHemispheres> sxms_;
    std::array<Mxm, kMxmCount> mxms_{};
    InstructionControlUnit icu_{};
    std::array<std::optional<C2cEndpoint>, hw::kHemispheres> c2cs_{};
    std::array<C2cLink*, hw::kHemispheres> c2c_outbound_links_{};
    std::array<C2cLink*, hw::kHemispheres> c2c_inbound_links_{};
    std::array<C2cDmaEngine*, hw::kHemispheres> c2c_dmas_{};
    std::size_t cycle_{0};
    std::array<std::array<std::size_t, hw::kWestStreams>, hw::kHemispheres>
        passive_bridge_transfer_counts_{};
    std::array<std::array<std::size_t, hw::kWestStreams>, hw::kHemispheres>
        last_passive_bridge_cycles_{};
    CyclePhase phase_{CyclePhase::Idle};
};

} // namespace ftlpu
