#pragma once

#include "ftlpu/c2c/dma.hpp"
#include "ftlpu/c2c/link.hpp"
#include "ftlpu/c2c/slice.hpp"
#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/hemisphere.hpp"
#include "ftlpu/core/stream_port.hpp"
#include "ftlpu/mem/mem_array.hpp"
#include "ftlpu/mxm/mxm.hpp"
#include "ftlpu/sxm/slice.hpp"
#include "ftlpu/system/hardware_configuration.hpp"
#include "ftlpu/system/icu.hpp"
#include "ftlpu/system/stream_topology_builder.hpp"
#include "ftlpu/vxm/sr_slice.hpp"
#include <algorithm>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
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
        // Preserve the per-cycle MEM pipeline/transfer records without
        // requiring the verbose human-readable MEM log stream.
        bool capture_mem_trace{false};
    };

    explicit TspSliceSystem(
        SystemHardwareConfiguration hardware = {})
        : hardware_configuration_(hardware)
        , stream_layout_(make_configured_stream_layout())
        , active_stream_routes_ {
            stream_layout_.topology.default_route_selection(),
            stream_layout_.topology.default_route_selection(),
        }
        , mems_ {
            MemArrayModel(stream_layout_.mem_ports),
            MemArrayModel(stream_layout_.mem_ports),
        }
        , streams_ {
            StreamRegisterFabric(stream_layout_.topology.column_count()),
            StreamRegisterFabric(stream_layout_.topology.column_count()),
        }
        , sxms_ {
            SxmSlice(stream_layout_.sxm_ports),
            SxmSlice(stream_layout_.sxm_ports),
        }
    {
        if (stream_layout_.fabric_names.size() != hw::kHemispheres) {
            throw std::invalid_argument(
                "stream layout fabric count does not match the CModel");
        }
        for (std::size_t index = 0; index < hw::kHemispheres; ++index) {
            if (stream_layout_.fabric_names[index]
                != hemisphere_name(static_cast<Hemisphere>(index))) {
                throw std::invalid_argument(
                    "stream layout fabric order does not match Hemisphere");
            }
        }
        for (const auto& transfer : stream_layout_.system_transfers) {
            if (transfer.latency_cycles != 1) {
                throw std::invalid_argument(
                    "TspSliceSystem supports one-cycle system stream transfers only");
            }
        }
        configure_hardware(hardware_configuration_);
    }

    void configure_hardware(SystemHardwareConfiguration hardware)
    {
        require_phase(CyclePhase::Idle, "configuring hardware");
        hardware.validate();
        hardware_configuration_ = hardware;
        for (auto& mem : mems_)
            mem.set_sram_depth_rows(hardware.sram_depth_rows);
    }

    const SystemHardwareConfiguration& hardware_configuration() const noexcept
    {
        return hardware_configuration_;
    }

    const StreamLayout& stream_layout() const noexcept
    {
        return stream_layout_;
    }

    StreamTopology::RouteSelection& stream_route_selection(
        Hemisphere hemisphere) noexcept
    {
        return active_stream_routes_[hemisphere_index(hemisphere)];
    }

    const StreamTopology::RouteSelection& stream_route_selection(
        Hemisphere hemisphere) const noexcept
    {
        return active_stream_routes_[hemisphere_index(hemisphere)];
    }

    void attach_c2c(
        Hemisphere hemisphere,
        C2cLink& outbound_link,
        C2cLink& inbound_link)
    {
        require_phase(CyclePhase::Idle, "attaching C2C");
        const auto index = hemisphere_index(hemisphere);
        c2cs_[index].emplace(stream_layout_.c2c_ports);
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
        c2cs_[index].emplace(stream_layout_.c2c_ports, "C2C DMA",
            true);
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

    const VxmSrSlice& vxm_unit() const noexcept
    {
        return vxm_;
    }

    VxmSrSlice& vxm_unit() noexcept
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
        if (group >= hw::kStreamsPerDirection / 2)
            throw std::out_of_range("VXM input group is outside 0..15");
        legacy_vxm_input_sources_[group] = source;
    }

    void configure_vxm_output_block_destination(
        std::size_t block, Hemisphere destination)
    {
        if (block >= hw::kVxmAluCount / 2)
            throw std::out_of_range(
                "VXM output block exceeds the configured physical block count");
        legacy_vxm_output_destinations_[block] = destination;
    }

    const StreamRegisterFabric& stream_fabric(
        Hemisphere hemisphere) const noexcept
    {
        return streams_[hemisphere_index(hemisphere)];
    }

    StreamRegisterFabric& stream_fabric(
        Hemisphere hemisphere) noexcept
    {
        return streams_[hemisphere_index(hemisphere)];
    }

    const MemArrayModel& mem_array(Hemisphere hemisphere) const noexcept
    {
        return mems_[hemisphere_index(hemisphere)];
    }

    MemArrayModel& mem_array(Hemisphere hemisphere) noexcept
    {
        return mems_[hemisphere_index(hemisphere)];
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
        for (auto& streams : streams_) streams.reset();
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
        for (auto& streams : streams_) {
            streams.begin_cycle();
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
        vxm_.begin_cycle();
        transfer_stream_edges_into_vxm(sinks);
        transfer_vxm_egress_to_stream_edges(sinks);
        vxm_.tick(sinks.vxm, sinks.vxm_log_tile);
        phase_ = CyclePhase::VxmEvaluated;
    }

    void mem_sxm_commit_phase(LogSinks sinks)
    {
        require_phase(CyclePhase::VxmEvaluated, "committing MEM and SXM");
        for (std::size_t hemisphere = 0; hemisphere < hw::kHemispheres; ++hemisphere) {
            try {
                auto& fabric = streams_[hemisphere];
                sxms_[hemisphere].set_trace_enabled(sinks.sxm != nullptr);
                if (sinks.mem != nullptr) {
                    *sinks.mem << "mem." << hemisphere_short_name(static_cast<Hemisphere>(hemisphere))
                               << " cycle " << cycle_ << '\n';
                }

                mems_[hemisphere].evaluate(
                    fabric,
                    sinks.mem != nullptr || sinks.capture_mem_trace);
                sxms_[hemisphere].evaluate(fabric);
                if (c2cs_[hemisphere].has_value()) {
                    evaluate_c2c(hemisphere, fabric);
                }
                stream_layout_.topology.stage_active_routes(
                    fabric, active_stream_routes_[hemisphere]);
                fabric.commit_cycle();

                if (sinks.mem != nullptr) {
                    mems_[hemisphere].log_cycle(
                        *sinks.mem, sinks.mem_log_tile);
                    log_streams(
                        *sinks.mem, fabric, sinks.mem_log_tile);
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
            if (const auto tag = dma->take_completion_sync_tag()) {
                icu_.notify_tagged(IcuLocation::C2cDma(
                    static_cast<Hemisphere>(hemisphere)), *tag);
            }
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
            while (const auto request =
                       endpoint->tx().take_mem_read_request()) {
                const auto& consumer = request->consumer;
                const auto location = IcuLocation::Mem(
                    consumer.hemisphere, consumer.mem_slice,
                    consumer.mem_bank);
                icu_.notify_mem_synchronized(location,
                    request->sync_tag == 0
                        ? request->stream_index : request->sync_tag);
            }
            const auto notify = [this](C2cReceiveNotification received) {
                const auto& consumer = received.consumer;
                if (consumer.notify_mem) {
                    const auto location = IcuLocation::Mem(
                        consumer.hemisphere,
                        consumer.mem_slice,
                        consumer.mem_bank);
                    if (received.sync_tag == 0) {
                        icu_.notify(location);
                        icu_.notify_mem_synchronized(
                            location, received.stream_index);
                    } else
                        icu_.notify_mem_synchronized(
                            location, received.sync_tag);
                }
            };
            const auto ready = [this](const C2cInstruction& received) {
                const auto& consumer = received.consumer;
                if (!consumer.notify_mem) return true;
                const auto location = IcuLocation::Mem(
                    consumer.hemisphere,
                    consumer.mem_slice,
                    consumer.mem_bank);
                const auto tag = received.sync_tag == 0
                    ? received.fabric_stream_index : received.sync_tag;
                return icu_.mem_accepts_synchronized_notification(
                    location, tag);
            };
            endpoint->rx().evaluate_shared(
                fabric, *c2c_dmas_[hemisphere],
                hardware_configuration_.c2c_streams_per_direction,
                ready, notify);
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
            const auto location = IcuLocation::Mem(
                notification->consumer.hemisphere,
                notification->consumer.mem_slice,
                notification->consumer.mem_bank);
            if (notification->sync_tag == 0) {
                icu_.notify(location);
                icu_.notify_mem_synchronized(
                    location, notification->stream_index);
            } else
                icu_.notify_mem_synchronized(
                    location, notification->sync_tag);
        }
    }

    static Hemisphere mxm_hemisphere(std::size_t mxm)
    {
        return static_cast<Hemisphere>(mxm / kMxmCountPerHemisphere);
    }

    static std::size_t local_mxm_index(std::size_t mxm)
    {
        return mxm % kMxmCountPerHemisphere;
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
                               << stream_layout_.mxm_weight_input.column
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
            auto input = StreamInputPort(
                streams_[hemisphere],
                stream_layout_.mxm_activation_input.column,
                stream_layout_.mxm_activation_input.direction,
                "MXM" + std::to_string(mxm) + " datapath");
            auto output = StreamOutputPort(
                streams_[hemisphere],
                stream_layout_.mxm_result_output.column,
                stream_layout_.mxm_result_output.direction,
                "MXM" + std::to_string(mxm) + " datapath");
            mxms_[mxm].tick_datapath(
                input, output, local_mxm_index(mxm), sinks.mxm,
                sinks.mxm_log_tile);
        }
    }

    MxmControlSlice::WeightInput collect_mxm_weight_input_from_streams(std::size_t mxm, std::size_t tile)
    {
        auto input = MxmControlSlice::WeightInput {};
        const auto hemisphere = hemisphere_index(mxm_hemisphere(mxm));
        auto streams = StreamInputPort(
            streams_[hemisphere],
            stream_layout_.mxm_weight_input.column,
            stream_layout_.mxm_weight_input.direction,
            "MXM" + std::to_string(mxm) + " weight input");
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
                    const auto word = streams.consume_cell(
                        tile, lane, stream_base);
                    if (!word.valid) {
                        throw std::logic_error(
                            "MXM INT8 column IW reached tile before its weight stream arrived at the MXM boundary register");
                    }
                    input[lane][column] =
                        MxmArray::Supercell::InputWord {
                            word.data,
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
                    const auto word = streams.consume_cell(
                        tile, lane, stream_base + column);
                    if (!word.valid) {
                        throw std::logic_error(
                            "MXM INT8 IW reached tile before all eight weight streams arrived at the MXM boundary register"
                            " (tile=" + std::to_string(tile)
                            + ", lane=" + std::to_string(lane)
                            + ", stream="
                            + std::to_string(stream_base + column) + ")");
                    }
                    input[lane][column] =
                        MxmArray::Supercell::InputWord {
                            word.data,
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
                const auto low_word = streams.consume_cell(tile, lane, low);
                const auto high_word = streams.consume_cell(tile, lane, high);
                if (!low_word.valid || !high_word.valid) {
                    throw std::logic_error(
                        "MXM column IW reached tile before both 16-bit weight streams arrived at the MXM boundary register");
                }
                const auto bits = static_cast<std::uint16_t>(
                    static_cast<std::uint16_t>(low_word.data)
                    | (static_cast<std::uint16_t>(high_word.data) << 8));
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
                const auto low = streams.consume_cell(
                    tile, lane, low_stream);
                const auto high = streams.consume_cell(
                    tile, lane, low_stream + 1);
                if (!low.valid || !high.valid) {
                    throw std::logic_error(
                        "MXM IW reached tile before both 16-bit weight streams arrived at the MXM boundary register");
                }
                const auto bits = static_cast<std::uint16_t>(
                    static_cast<std::uint16_t>(low.data)
                    | (static_cast<std::uint16_t>(high.data) << 8));
                input[lane][column] = MxmArray::Supercell::InputWord {
                    bits,
                    column + 1 == hw::kMxmSupercellColumns,
                };
            }
        }
        return input;
    }


    // Move complete byte streams from the two ordinary hemisphere fabrics
    // into the corresponding end register of the VXM SR chain.  The system
    // transfer descriptor supplies the physical direction; no VXM-private
    // bundle or group mux exists here.
    void transfer_stream_edges_into_vxm(LogSinks sinks)
    {
        for (const auto& transfer : stream_layout_.system_transfers) {
            if (transfer.kind
                != target::StreamSystemTransferKind::PassiveBridge)
                continue;
            auto& source = streams_[transfer.source_fabric];
            auto input = StreamInputPort(source, transfer.source.column,
                transfer.source.direction, "VXM SR ingress");
            const auto internalDirection = transfer.source_fabric == 0
                ? StreamDirection::East : StreamDirection::West;
            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
                for (std::size_t stream = 0;
                     stream < hw::kStreamsPerDirection; ++stream) {
                    auto complete = true;
                    for (std::size_t lane = 0;
                         lane < hw::kLanesPerTile; ++lane) {
                        const auto id = transfer.source.direction
                                == StreamDirection::East
                            ? StreamId::East(stream)
                            : StreamId::West(stream);
                        complete = complete
                            && source.cell(transfer.source.column, tile,
                                lane, id).valid;
                    }
                    if (!complete) continue;
                    auto values = StreamTileSegment {};
                    for (std::size_t lane = 0;
                         lane < hw::kLanesPerTile; ++lane)
                        values[lane] = input.consume_cell(tile, lane, stream);
                    const auto internal = internalDirection
                            == StreamDirection::East
                        ? StreamId::East(stream) : StreamId::West(stream);
                    vxm_.stage_ingress(tile, internal, values);
                    if (sinks.vxm
                        && (!sinks.vxm_log_tile
                            || *sinks.vxm_log_tile == tile))
                        *sinks.vxm << "  "
                            << stream_layout_.fabric_names[
                                   transfer.source_fabric]
                            << " -> VXM "
                            << (internalDirection == StreamDirection::East
                                    ? 'E' : 'W')
                            << stream << " tile " << tile << '\n';
                }
            }
        }
    }

    // Drain values that have traversed all sixteen ordinary VXM SR columns.
    // East-moving values arrive in fabric 1 and west-moving values in fabric
    // 0.  They are written to the same ordinary boundary SR used by MEM.
    void transfer_vxm_egress_to_stream_edges(LogSinks sinks)
    {
        for (std::size_t directionIndex = 0; directionIndex < 2;
             ++directionIndex) {
            const auto direction = directionIndex == 0
                ? StreamDirection::East : StreamDirection::West;
            const auto destinationFabric = direction == StreamDirection::East
                ? std::size_t {1} : std::size_t {0};
            const auto* transfer = [&]() -> const StreamSystemTransfer* {
                for (const auto& candidate : stream_layout_.system_transfers)
                    if (candidate.destination_fabric == destinationFabric)
                        return &candidate;
                return nullptr;
            }();
            if (!transfer)
                throw std::logic_error(
                    "VXM SR egress has no system transfer destination");
            auto output = StreamOutputPort(streams_[destinationFabric],
                transfer->destination.column,
                transfer->destination.direction, "VXM SR egress");
            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
                for (std::size_t stream = 0;
                     stream < hw::kStreamsPerDirection; ++stream) {
                    const auto id = direction == StreamDirection::East
                        ? StreamId::East(stream) : StreamId::West(stream);
                    if (!vxm_.egress_valid(tile, id)) continue;
                    const auto values = vxm_.egress_segment(tile, id);
                    for (std::size_t lane = 0;
                         lane < hw::kLanesPerTile; ++lane)
                        output.write_cell(tile, lane, stream, values[lane]);
                    vxm_.consume_egress(tile, id);
                    ++passive_bridge_transfer_counts_[
                        direction == StreamDirection::East ? 0 : 1][stream];
                    last_passive_bridge_cycles_[
                        direction == StreamDirection::East ? 0 : 1][stream]
                        = cycle_ + 1;
                    if (sinks.system)
                        *sinks.system << "  VXM "
                            << (direction == StreamDirection::East ? 'E' : 'W')
                            << stream << " -> "
                            << stream_layout_.fabric_names[destinationFabric]
                            << " tile " << tile << '\n';
                }
            }
        }
    }

    static void print_hex_bytes(
        std::ostream& os,
        const StreamPayloadTileSegment& bytes)
    {
        const auto old_flags = os.flags();
        const auto old_fill = os.fill();
        os << std::hex << std::setfill('0');
        for (const auto byte : bytes) {
            os << std::setw(2) << static_cast<unsigned>(byte);
        }
        os.flags(old_flags);
        os.fill(old_fill);
    }

    static bool collect_stream_bytes(
        const StreamRegisterFabric& fabric,
        std::size_t tile,
        std::size_t reg_column,
        StreamId stream,
        StreamPayloadTileSegment& bytes)
    {
        bool any = false;
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const auto& slot = fabric.cell(reg_column, tile, lane, stream);
            bytes[lane] = slot.valid ? slot.data : 0;
            any = any || slot.valid;
        }
        return any;
    }

    static void log_streams(
        std::ostream& os,
        const StreamRegisterFabric& fabric,
        std::optional<std::size_t> log_tile)
    {
        os << "  stream_registers (E/W combined):\n";
        const auto first_tile = log_tile.value_or(0);
        const auto end_tile = log_tile.has_value()
            ? first_tile + 1
            : hw::kTileRows;
        for (std::size_t tile = first_tile; tile < end_tile; ++tile) {
            os << "    tile " << tile << ":\n";
            for (std::size_t reg = 0; reg < fabric.column_count(); ++reg) {
                os << "      sreg " << reg << ":";
                bool any = false;

                for (std::size_t stream = 0;
                     stream < hw::kEastStreams;
                     ++stream) {
                    StreamPayloadTileSegment bytes{};
                    if (collect_stream_bytes(
                            fabric, tile, reg, StreamId::East(stream),
                            bytes)) {
                        any = true;
                        os << " E" << stream << "=0x";
                        print_hex_bytes(os, bytes);
                    }
                }

                for (std::size_t stream = 0;
                     stream < hw::kWestStreams;
                     ++stream) {
                    StreamPayloadTileSegment bytes{};
                    if (collect_stream_bytes(
                            fabric, tile, reg, StreamId::West(stream),
                            bytes)) {
                        any = true;
                        os << " W" << stream << "=0x";
                        print_hex_bytes(os, bytes);
                    }
                }

                if (!any) {
                    os << " empty";
                }
                os << '\n';
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
    StreamLayout stream_layout_;
    std::array<StreamTopology::RouteSelection, hw::kHemispheres>
        active_stream_routes_;
    std::array<MemArrayModel, hw::kHemispheres> mems_;
    std::array<StreamRegisterFabric, hw::kHemispheres> streams_;
    VxmSrSlice vxm_{};
    // Retained only so older host configuration APIs remain source compatible.
    // Physical VXM packets now carry explicit E/W operand and result SRs.
    std::array<Hemisphere, hw::kStreamsPerDirection / 2>
        legacy_vxm_input_sources_{};
    std::array<Hemisphere, hw::kVxmAluCount / 2>
        legacy_vxm_output_destinations_{};
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
