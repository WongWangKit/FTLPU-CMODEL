#pragma once

#include "ftlpu/c2c/dma.hpp"
#include "ftlpu/c2c/link.hpp"
#include "ftlpu/c2c/slice.hpp"
#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/hemisphere.hpp"
#include "ftlpu/core/stream_port.hpp"
#include "ftlpu/core/timing.hpp"
#include "ftlpu/mem/mem_array.hpp"
#include "ftlpu/mxm/mxm.hpp"
#include "ftlpu/sxm/slice.hpp"
#include "ftlpu/system/hardware_configuration.hpp"
#include "ftlpu/system/icu.hpp"
#include "ftlpu/system/stream_topology_builder.hpp"
#include "ftlpu/vxm/slice.hpp"
#include <algorithm>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <vector>

namespace ftlpu {

class C2cEndpointClockSlot final : public timing::ClockedModule {
public:
    void bind(std::optional<C2cEndpoint>& endpoint) noexcept
    {
        endpoint_ = &endpoint;
    }

    void low() override {}

    void validate_posedge() const override
    {
        if (endpoint_ != nullptr && endpoint_->has_value())
            endpoint_->value().validate_posedge();
    }

    void posedge() override
    {
        if (endpoint_ != nullptr && endpoint_->has_value())
            endpoint_->value().posedge();
    }

    void high() override
    {
        if (endpoint_ != nullptr && endpoint_->has_value())
            endpoint_->value().high();
    }

private:
    std::optional<C2cEndpoint>* endpoint_{};
};

// Optional DMA/DDR objects are attached after TspSliceSystem construction.
// Register a permanent forwarding slot with Clock so the global commit set is
// stable and independent of attachment order. LOW remains owned exclusively by
// SystemPhase and therefore is intentionally not forwarded here.
class OptionalClockedModuleSlot final : public timing::ClockedModule {
public:
    void bind(timing::ClockedModule* module) noexcept { module_ = module; }

    void low() override {}
    void validate_posedge() const override
    {
        if (module_ != nullptr) module_->validate_posedge();
    }
    void posedge() override
    {
        if (module_ != nullptr) module_->posedge();
    }
    void high() override
    {
        if (module_ != nullptr) module_->high();
    }
    void negedge() override
    {
        if (module_ != nullptr) module_->negedge();
    }

private:
    timing::ClockedModule* module_{};
};

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
        SystemHardwareConfiguration hardware = {},
        SxmTransposeArchitecture sxm_architecture = SxmTransposeArchitecture::LocalBlocks,
        MemArrayModel::ReadTiming mem_read_timing =
            MemArrayModel::ReadTiming::RegisteredR1)
        : hardware_configuration_(hardware)
        , stream_layout_(make_configured_stream_layout())
        , active_stream_routes_ {
            stream_layout_.topology.default_route_selection(),
            stream_layout_.topology.default_route_selection(),
        }
        , mems_ {
            MemArrayModel(stream_layout_.mem_ports,
                MemArrayModel::MissingStreamPolicy::Error, mem_read_timing),
            MemArrayModel(stream_layout_.mem_ports,
                MemArrayModel::MissingStreamPolicy::Error, mem_read_timing),
        }
        , streams_ {
            StreamRegisterFabric(stream_layout_.topology.column_count()),
            StreamRegisterFabric(stream_layout_.topology.column_count()),
        }
        , sxms_ {
            SxmSlice(stream_layout_.sxm_ports,sxm_architecture),
            SxmSlice(stream_layout_.sxm_ports,sxm_architecture),
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

        // Clock registration defines only the shared edge/commit set. The
        // SystemPhase node order below independently defines combinational and
        // legacy evaluate topology. The same StreamRegisterFabric objects are
        // referenced by both mechanisms, but each phase callback has one owner.
        icu_.enable_clocked_mode();
        clock_.register_module(icu_);
        clock_.register_module(program_loader_clock_slot_);
        clock_.register_module(program_loader_dma_clock_slot_);
        for (auto& stream : streams_) clock_.register_module(stream);
        for (auto& mem : mems_) clock_.register_module(mem);
        for (auto& mxm : mxms_) {
            clock_.register_module(mxm.control());
            clock_.register_module(mxm);
        }
        clock_.register_module(vxm_);
        for (auto& sxm : sxms_) clock_.register_module(sxm);
        for (std::size_t index = 0; index < hw::kHemispheres; ++index) {
            c2c_clock_slots_[index].bind(c2cs_[index]);
            clock_.register_module(c2c_clock_slots_[index]);
            clock_.register_module(c2c_dma_clock_slots_[index]);
        }
        clock_.register_module(c2c_ddr_clock_slot_);
        system_phase_.add_node("ProgramLoader control LOW", [this] {
            if (program_loader_begin_low_)
                program_loader_begin_low_(cycle());
        });
        system_phase_.add_node("stream-register LOW", [this] {
            begin_cycle_phase(active_sinks_);
        });
        system_phase_.add_node("ICU dispatch", [this] {
            dispatch_phase(active_sinks_);
        });
        system_phase_.add_node("MXM control then datapath", [this] {
            mxm_phase(active_sinks_);
        });
        system_phase_.add_node("VXM capture then execute then route", [this] {
            vxm_phase(active_sinks_);
        });
        system_phase_.add_node("MEM SXM C2C and passive-route staging", [this] {
            mem_sxm_stage_phase(active_sinks_);
        });
        system_phase_.add_node("C2C DMA DDR and barrier progression", [this] {
            advance_dma_ddr_and_barrier_phase();
        });
        system_phase_.add_node("postedge observations", {}, [this] {
            postedge_observation_phase(active_sinks_);
        });
    }

    // SystemPhase callbacks capture this object's address, and Clock keeps
    // non-owning pointers to its StreamRegisterFabric members.
    TspSliceSystem(const TspSliceSystem&) = delete;
    TspSliceSystem& operator=(const TspSliceSystem&) = delete;
    TspSliceSystem(TspSliceSystem&&) = delete;
    TspSliceSystem& operator=(TspSliceSystem&&) = delete;

    void configure_hardware(SystemHardwareConfiguration hardware)
    {
        require_idle("configuring hardware");
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
        require_idle("attaching C2C");
        const auto index = hemisphere_index(hemisphere);
        c2cs_[index].emplace(stream_layout_.c2c_ports);
        c2c_outbound_links_[index] = &outbound_link;
        c2c_inbound_links_[index] = &inbound_link;
        c2c_dmas_[index] = nullptr;
        c2c_dma_clock_slots_[index].bind(nullptr);
    }

    void attach_c2c_dma(
        Hemisphere hemisphere,
        C2cDmaEngine& dma)
    {
        require_idle("attaching C2C DMA");
        const auto index = hemisphere_index(hemisphere);
        c2cs_[index].emplace(stream_layout_.c2c_ports, "C2C DMA",
            true);
        c2c_outbound_links_[index] = nullptr;
        c2c_inbound_links_[index] = nullptr;
        if (!dma.clocked_mode()) dma.enable_clocked_mode();
        c2c_dmas_[index] = &dma;
        c2c_dma_clock_slots_[index].bind(&dma);
    }

    void attach_c2c_ddr(Ddr4Model& ddr4)
    {
        require_idle("attaching C2C DDR4");
        if (c2c_ddr_ != nullptr && c2c_ddr_ != &ddr4)
            throw std::logic_error("TSP already has a different C2C DDR4");
        if (c2c_ddr_ == nullptr) ddr4.enable_clocked_mode();
        c2c_ddr_ = &ddr4;
        c2c_ddr_clock_slot_.bind(&ddr4);
    }

    template<class Loader>
    void attach_program_loader(Loader& loader)
    {
        require_idle("attaching ProgramLoader");
        if (program_loader_begin_low_)
            throw std::logic_error(
                "TSP already has a ProgramLoader");
        if (!loader.clocked_mode()) loader.enable_clocked_mode();
        program_loader_clock_slot_.bind(&loader);
        program_loader_dma_clock_slot_.bind(&loader.dma());
        program_loader_begin_low_=[&loader](std::size_t cycle) {
            loader.begin_clocked_low(cycle);
        };
        program_loader_advance_low_=[&loader](std::size_t cycle) {
            loader.advance_clocked_dma_low(cycle);
        };
        program_loader_high_=[&loader] {
            loader.high();
            loader.dma().high();
        };
    }

    void stage_program_reset()
    {
        require_idle("staging a program reset");
        icu_.stage_program_reset();
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
        require_idle("detaching C2C");
        const auto index = hemisphere_index(hemisphere);
        c2cs_[index].reset();
        c2c_outbound_links_[index] = nullptr;
        c2c_inbound_links_[index] = nullptr;
        c2c_dmas_[index] = nullptr;
        c2c_dma_clock_slots_[index].bind(nullptr);
    }

    void detach_c2c()
    {
        require_idle("detaching all C2C endpoints");
        for (std::size_t index = 0; index < hw::kHemispheres; ++index) {
            c2cs_[index].reset();
            c2c_outbound_links_[index] = nullptr;
            c2c_inbound_links_[index] = nullptr;
            c2c_dmas_[index] = nullptr;
            c2c_dma_clock_slots_[index].bind(nullptr);
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

    const SxmSlice& sxm_unit(Hemisphere hemisphere) const noexcept
    {
        return sxms_[hemisphere_index(hemisphere)];
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
        if (group >= hw::kStreamsPerDirection / 2)
            throw std::out_of_range("VXM input group is outside 0..15");
        vxm_.configure_input_group_source(group, source);
    }

    void configure_vxm_output_block_destination(
        std::size_t block, Hemisphere destination)
    {
        if (block >= hw::kVxmAluCount / 2)
            throw std::out_of_range(
                "VXM output block exceeds the configured physical block count");
        vxm_.configure_output_block_destination(block, destination);
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
        require_idle("starting a physical clock cycle");
        active_sinks_ = sinks;
        try {
            // LOW topology and POSEDGE edge control are deliberately separate.
            system_phase_.low();
            clock_.complete_low();
            clock_.validate_all();
            clock_.commit_all();
            system_phase_.high();
            clock_.complete_high();
            clock_.negedge_all();
        } catch (...) {
            system_phase_.mark_faulted();
            clock_.mark_faulted();
            active_sinks_ = {};
            throw;
        }
        active_sinks_ = {};
    }

    std::size_t cycle() const
    {
        return static_cast<std::size_t>(clock_.cycle());
    }

    std::uint64_t phase_count() const noexcept { return clock_.phase_count(); }
    timing::ClockPhase clock_phase() const noexcept { return clock_.phase(); }

    // The runtime may query this after the last visible MEM commit before
    // notifying an event. Transport must leave shared issue/SR resources.
    bool common_barrier_resources_drained() const noexcept
    {
        if (!vxm_.idle()) return false;
        for (const auto& mxm : mxms_) if (!mxm.idle()) return false;
        for (std::size_t h = 0; h < hw::kHemispheres; ++h)
            if (!mems_[h].idle() || !sxms_[h].idle() || !streams_[h].empty()) return false;
        return true;
    }

    bool program_resources_drained() const noexcept {
        if (!common_barrier_resources_drained()) return false;
        for (std::size_t h=0;h<hw::kHemispheres;++h) {
            if (c2cs_[h] && (!c2cs_[h]->tx().idle() || !c2cs_[h]->rx().idle())) return false;
            if (c2c_dmas_[h] && !c2c_dmas_[h]->drained()) return false;
        }
        return true;
    }
    bool can_load_program() const
    {
        return timing_idle() && icu_.program_queues_done()
            && program_resources_drained();
    }
    // Only task execution state changes; SRAM, DDR/DMA and device time survive.
    void reset_program() {
        require_idle("switching program");
        if (!can_load_program()) throw std::logic_error("previous kernel has not drained");
        for(auto* dma:c2c_dmas_) if(dma) {
            while(dma->take_completion_sync_tag()) {}
            while(dma->take_completion_notification()) {}
        }
        icu_.reset_program();
    }

    bool common_compute_resources_drained() const noexcept
    {
        if (!vxm_.idle()) return false;
        for (const auto& mxm : mxms_) if (!mxm.idle()) return false;
        for (std::size_t h = 0; h < hw::kHemispheres; ++h)
            if (!mems_[h].compute_idle() || !sxms_[h].idle()
                || !streams_[h].compute_empty()) return false;
        return true;
    }

    bool common_transport_resources_drained() const noexcept
    {
        for (std::size_t h = 0; h < hw::kHemispheres; ++h)
            if (!mems_[h].transport_idle() || !streams_[h].transport_empty()
                || (c2cs_[h] && !c2cs_[h]->rx().idle())) return false;
        return true;
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
        require_idle("resetting execution state");
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
        clock_.reset_time();
        system_phase_.reset();
        for (auto* dma : c2c_dmas_) {
            if (dma != nullptr) dma->reset();
        }
        if (c2c_ddr_ != nullptr) c2c_ddr_->reset_execution_state();
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

    [[nodiscard]] bool timing_idle() const noexcept
    {
        return clock_.phase() == timing::ClockPhase::LOW
            && !clock_.faulted()
            && !system_phase_.faulted()
            && system_phase_.ready_for_low();
    }

    void require_idle(const char* operation) const
    {
        if (!timing_idle())
            throw std::logic_error(
                std::string("TSP timing boundary violation while ") + operation);
    }

    void begin_cycle_phase(LogSinks sinks)
    {
        if (sinks.system != nullptr) {
            *sinks.system << "system cycle " << cycle() << '\n';
        }
        for (auto& streams : streams_) {
            streams.low();
        }
    }

    void dispatch_phase(LogSinks sinks)
    {
        // The first implementation only admits an already-drained boundary.
        // Active Macro Context RAM and SRAM/accumulator/transpose storage may
        // remain populated; transient FU pipelines and SR cannot cross HOLD.
        // Validate before consuming even an early-ready barrier at cycle C.
        if (icu_.common_barrier_due()) {
            if (!vxm_.idle())
                throw StaticScheduleError("unsafe common barrier: VXM has in-flight computation");
            for (const auto& mxm : mxms_)
                if (!mxm.idle())
                    throw StaticScheduleError("unsafe common barrier: MXM has in-flight computation");
            for (std::size_t hemisphere = 0; hemisphere < hw::kHemispheres; ++hemisphere) {
                if (!(icu_.transport_overlap_enabled() ? mems_[hemisphere].compute_idle() : mems_[hemisphere].idle())
                    || !sxms_[hemisphere].idle()
                    || !(icu_.transport_overlap_enabled() ? streams_[hemisphere].compute_empty() : streams_[hemisphere].empty()))
                    throw StaticScheduleError("unsafe common barrier: MEM/SXM/SR has in-flight data; schedule a drain window before C");
            }
        }
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
    }

    void mxm_phase(LogSinks sinks)
    {
        tick_mxm_controls(sinks);
        tick_mxm_datapaths(sinks);
    }

    void vxm_phase(LogSinks sinks)
    {
        vxm_.prepare_cycle();
        transfer_mem_edges_to_vxm(sinks);
        vxm_.evaluate(sinks.vxm, sinks.vxm_log_tile);
        transfer_vxm_to_mem_edges(sinks);
        transfer_unconsumed_streams_across_vxm(sinks);
    }

    void mem_sxm_stage_phase(LogSinks sinks)
    {
        for (std::size_t hemisphere = 0; hemisphere < hw::kHemispheres; ++hemisphere) {
            try {
                auto& fabric = streams_[hemisphere];
                sxms_[hemisphere].set_trace_enabled(sinks.sxm != nullptr);
                if (sinks.mem != nullptr) {
                    *sinks.mem << "mem." << hemisphere_short_name(static_cast<Hemisphere>(hemisphere))
                               << " cycle " << cycle() << '\n';
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
            } catch (const std::exception& error) {
                throw std::logic_error(
                    "MEM/SXM staging failed at system cycle "
                    + std::to_string(cycle()) + " in hemisphere "
                    + hemisphere_short_name(static_cast<Hemisphere>(hemisphere))
                    + ": " + error.what());
            }
        }
    }

    // Endpoint interaction has already staged producer/consumer handshakes.
    // DMA LOW consumes those inputs, DDR LOW consumes all ProgramLoader/DMA
    // requests, and ICU barrier notifications target the already-prepared ICU
    // Next snapshot. Every resulting state change commits at one POSEDGE.
    void advance_dma_ddr_and_barrier_phase()
    {
        // Preserve the existing arbitration priority: instruction loading
        // presents its DDR request before data/weight DMA clients.
        if (program_loader_advance_low_)
            program_loader_advance_low_(cycle());
        for (std::size_t hemisphere = 0;
             hemisphere < hw::kHemispheres;
             ++hemisphere) {
            auto* dma = c2c_dmas_[hemisphere];
            if (dma == nullptr) continue;
            dma->low();
            if (const auto tag = dma->take_completion_sync_tag()) {
                if(!icu_.program_loading()) icu_.notify_tagged(IcuLocation::C2cDma(
                    static_cast<Hemisphere>(hemisphere)), *tag);
            }
            if (dma->take_completion_notification()) {
                if(!icu_.program_loading()) icu_.notify(IcuLocation::C2cDma(
                    static_cast<Hemisphere>(hemisphere)));
            }
        }
        if (c2c_ddr_ != nullptr) c2c_ddr_->low();
        icu_.advance_barrier_events();
    }

    // HIGH is observation-only. All StreamRegisterFabric objects have already
    // committed through the global Clock barrier when this callback runs.
    void postedge_observation_phase(LogSinks sinks)
    {
        if (program_loader_high_) program_loader_high_();
        for (auto* dma : c2c_dmas_)
            if (dma != nullptr) dma->high();
        if (c2c_ddr_ != nullptr) c2c_ddr_->high();
        for (std::size_t hemisphere = 0; hemisphere < hw::kHemispheres; ++hemisphere) {
            auto& fabric = streams_[hemisphere];
            fabric.high();
            if (sinks.mem != nullptr) {
                mems_[hemisphere].log_cycle(*sinks.mem, sinks.mem_log_tile);
                log_streams(*sinks.mem, fabric, sinks.mem_log_tile);
            }
            if (sinks.sxm != nullptr) {
                *sinks.sxm << "sxm."
                            << hemisphere_short_name(static_cast<Hemisphere>(hemisphere))
                            << " system_cycle " << cycle() << '\n';
                sxms_[hemisphere].log_cycle(*sinks.sxm);
            }
        }
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
                *sinks.mxm << "mxm" << mxm << " cycle " << cycle() << '\n';
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
            try {
                if (sinks.mxm != nullptr) {
                    mxms_[mxm].control().low(*sinks.mxm, provider, false, sinks.mxm_log_tile);
                } else {
                    static NullStream null_stream;
                    mxms_[mxm].control().low(null_stream.stream(), provider, false);
                }
            } catch (const std::exception& error) {
                throw std::logic_error(
                    "MXM" + std::to_string(mxm)
                    + " control at system cycle " + std::to_string(cycle())
                    + ": " + error.what());
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
            try {
                mxms_[mxm].low_datapath(
                    input, output, local_mxm_index(mxm), sinks.mxm,
                    sinks.mxm_log_tile);
            } catch (const std::exception& error) {
                throw std::logic_error(
                    "MXM" + std::to_string(mxm)
                    + " datapath at system cycle " + std::to_string(cycle())
                    + ": " + error.what());
            }
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
            != MxmWeightInputMode::Direct16) {
            const auto stream_base = instruction->weight_stream_base;
            if (instruction->weight_load_mode
                == MxmWeightLoadMode::Vector) {
                const auto column = instruction->weight_inner_column;
                for (std::size_t lane = 0;
                     lane < hw::kLanesPerTile;
                     ++lane) {
                    const auto word = streams.consume_cell(
                        tile, lane, stream_base);
                    if (!word.valid) {
                        throw std::logic_error(
                            "MXM INT8 column IW reached tile before its weight stream arrived at the MXM boundary register"
                            " buffer=" + std::to_string(instruction->weight_buffer)
                            + " column=" + std::to_string(instruction->weight_column)
                            + " inner=" + std::to_string(column)
                            + " stream=" + std::to_string(stream_base));
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
        if (instruction->weight_load_mode == MxmWeightLoadMode::Vector) {
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
                        "MXM IW reached tile before both 16-bit weight streams arrived at the MXM boundary register"
                        " lane=" + std::to_string(lane)
                        + " column=" + std::to_string(column)
                        + " low_stream=" + std::to_string(low_stream)
                        + " low_valid=" + std::to_string(low.valid)
                        + " high_valid=" + std::to_string(high.valid));
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
            const auto& fabric = streams_[hemisphere_index(source)];
            for (std::size_t lane = 0;
                 lane < hw::kLanesPerTile;
                 ++lane) {
                for (std::size_t byte = 0;
                     byte < VxmLane::kStreamGroupBytes;
                     ++byte) {
                    if (!fabric.cell(
                            stream_layout_.vxm_input.column,
                            tile, lane,
                            stream_layout_.vxm_input.direction
                                == StreamDirection::East
                                ? StreamId::East(base + byte)
                                : StreamId::West(base + byte))
                             .valid) {
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
                const auto& required = vxm_.required_streams_at(tile);
                if (sinks.vxm != nullptr && required.has_value()
                    && (!sinks.vxm_log_tile.has_value()
                        || tile == *sinks.vxm_log_tile)) {
                    *sinks.vxm << "  MEM.edge -> VXM tile " << tile
                               << " incomplete:";
                    for (std::size_t group = 0;
                         group < VxmLane::kStreamGroupCount; ++group) {
                        const auto base = group * VxmLane::kStreamGroupBytes;
                        if (!(*required)[base] && !(*required)[base + 1])
                            continue;
                        const auto source = vxm_.input_group_source(group);
                        const auto& fabric =
                            streams_[hemisphere_index(source)];
                        bool missing = false;
                        for (std::size_t lane = 0;
                             lane < hw::kLanesPerTile && !missing; ++lane) {
                            for (std::size_t byte = 0;
                                 byte < VxmLane::kStreamGroupBytes; ++byte) {
                                missing = !fabric.cell(
                                    stream_layout_.vxm_input.column,
                                    tile, lane,
                                    stream_layout_.vxm_input.direction
                                        == StreamDirection::East
                                        ? StreamId::East(base + byte)
                                        : StreamId::West(base + byte)).valid;
                                if (missing) break;
                            }
                        }
                        if (missing)
                            *sinks.vxm << " group=" << group
                                       << " source="
                                       << hemisphere_short_name(source);
                    }
                    *sinks.vxm << '\n';
                }
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
                auto input = StreamInputPort(
                    streams_[hemisphere_index(source)],
                    stream_layout_.vxm_input.column,
                    stream_layout_.vxm_input.direction,
                    "VXM input");
                auto values = VxmSlice::InputBuffer::GroupVector {};
                for (std::size_t lane = 0;
                     lane < hw::kLanesPerTile;
                     ++lane) {
                    for (std::size_t byte = 0;
                         byte < VxmLane::kStreamGroupBytes;
                         ++byte) {
                        const auto cell = input.consume_cell(
                            tile, lane, base + byte);
                        if (!cell.valid) {
                            throw std::logic_error(
                                "VXM input group became incomplete during MEM-edge capture");
                        }
                        values[lane][byte] = cell.data;
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
        for (const auto& transfer : stream_layout_.system_transfers) {
            if (transfer.kind
                != target::StreamSystemTransferKind::PassiveBridge) {
                throw std::logic_error(
                    "unsupported stream system transfer kind");
            }
            const auto source_index = transfer.source_fabric;
            const auto destination_index = transfer.destination_fabric;
            const auto source = static_cast<Hemisphere>(source_index);
            auto destination = StreamOutputPort(
                streams_[destination_index],
                transfer.destination.column,
                transfer.destination.direction,
                transfer.name);
            const auto& source_fabric = streams_[source_index];
            for (std::size_t tile = 0; tile < hw::kTileRows; ++tile) {
                for (std::size_t stream = 0;
                     stream < hw::kStreamsPerDirection; ++stream) {
                    // The VXM output and passive bypass share one boundary
                    // mux.  A result produced in this cycle owns its output
                    // byte; an unrelated value on the same numbered input
                    // stream is discarded instead of being forwarded into
                    // the same stream register.
                    auto vxm_output_owns_destination = false;
                    for (const auto& output : vxm_.outputs_at(tile)) {
                        if (vxm_.output_stream_destination(output.stream)
                                != static_cast<Hemisphere>(destination_index)
                            || stream < output.stream
                            || stream >= output.stream + output.byte_count) {
                            continue;
                        }
                        vxm_output_owns_destination = true;
                        break;
                    }
                    if (vxm_output_owns_destination) {
                        continue;
                    }
                    const auto stream_id = transfer.source.direction
                        == StreamDirection::East
                        ? StreamId::East(stream)
                        : StreamId::West(stream);
                    if (transfer.source.column == stream_layout_.vxm_input.column
                        && transfer.source.direction
                            == stream_layout_.vxm_input.direction
                        && vxm_requires_stream_from(source, tile, stream)) {
                        continue;
                    }

                    auto complete = true;
                    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                        complete = complete
                            && source_fabric.cell(
                                transfer.source.column,
                                tile, lane, stream_id)
                              .valid;
                    }
                    if (!complete) continue;

                    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                        const auto& cell = source_fabric.cell(
                            transfer.source.column,
                            tile, lane, stream_id);
                        destination.write_cell(tile, lane, stream, cell);
                    }
                    ++passive_bridge_transfer_counts_[source_index][stream];
                    last_passive_bridge_cycles_[source_index][stream] = cycle() + 1;
                    if (sinks.system != nullptr) {
                        *sinks.system << "  passive VXM bridge "
                                      << stream_layout_.fabric_names[source_index]
                                      << '.'
                                      << (transfer.source.direction
                                              == StreamDirection::East ? 'E' : 'W')
                                      << stream
                                      << " -> "
                                      << stream_layout_.fabric_names[destination_index]
                                      << '.'
                                      << (transfer.destination.direction
                                              == StreamDirection::East ? 'E' : 'W')
                                      << stream << " tile " << tile << '\n';
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
                auto stream_output = StreamOutputPort(
                    streams_[hemisphere_index(destination)],
                    stream_layout_.vxm_output.column,
                    stream_layout_.vxm_output.direction,
                    "VXM output");
                for (std::size_t byte = 0; byte < output.byte_count; ++byte) {
                    const auto stream = output.stream + byte;
                    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                        stream_output.write_cell(
                            tile, lane, stream,
                            StreamCell::Valid(
                                output.byte_values[lane][byte],
                                lane + 1 == hw::kLanesPerTile));
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
    VxmSlice vxm_{};
    std::array<SxmSlice, hw::kHemispheres> sxms_;
    std::array<Mxm, kMxmCount> mxms_{};
    InstructionControlUnit icu_{};
    OptionalClockedModuleSlot program_loader_clock_slot_{};
    OptionalClockedModuleSlot program_loader_dma_clock_slot_{};
    std::function<void(std::size_t)> program_loader_begin_low_{};
    std::function<void(std::size_t)> program_loader_advance_low_{};
    std::function<void()> program_loader_high_{};
    std::array<std::optional<C2cEndpoint>, hw::kHemispheres> c2cs_{};
    std::array<C2cEndpointClockSlot, hw::kHemispheres> c2c_clock_slots_{};
    std::array<OptionalClockedModuleSlot, hw::kHemispheres>
        c2c_dma_clock_slots_{};
    OptionalClockedModuleSlot c2c_ddr_clock_slot_{};
    std::array<C2cLink*, hw::kHemispheres> c2c_outbound_links_{};
    std::array<C2cLink*, hw::kHemispheres> c2c_inbound_links_{};
    std::array<C2cDmaEngine*, hw::kHemispheres> c2c_dmas_{};
    Ddr4Model* c2c_ddr_{};
    timing::Clock clock_{};
    timing::SystemPhase system_phase_{};
    LogSinks active_sinks_{};
    std::array<std::array<std::size_t, hw::kWestStreams>, hw::kHemispheres>
        passive_bridge_transfer_counts_{};
    std::array<std::array<std::size_t, hw::kWestStreams>, hw::kHemispheres>
        last_passive_bridge_cycles_{};
};

} // namespace ftlpu
