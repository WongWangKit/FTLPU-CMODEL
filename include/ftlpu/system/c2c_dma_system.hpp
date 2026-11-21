#pragma once

#include "ftlpu/c2c/c2c.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"
#include "ftlpu/system/program_loader.hpp"

#include <array>
#include <cstddef>

namespace ftlpu {

// One chip with a DMA-controlled external C2C port on each hemisphere.
// Both DMA engines share the modeled DDR4 command/data interface.
class C2cDmaSystem {
public:
    explicit C2cDmaSystem(
        Ddr4Config ddr4_config = {},
        std::size_t dma_fifo_depth_vectors = hw::config::kC2cDmaSizing.rt,
        SystemHardwareConfiguration hardware = {},
        ProgramLoadConfig program_load = {},
        SxmTransposeArchitecture sxm_architecture = SxmTransposeArchitecture::LocalBlocks)
        : ddr4_(ddr4_config)
        , dmas_ {
            C2cDmaEngine(ddr4_, dma_fifo_depth_vectors,
                hardware.c2c_streams_per_direction),
            C2cDmaEngine(ddr4_, dma_fifo_depth_vectors,
                hardware.c2c_streams_per_direction,hw::kC2cStreamsPerDirection)}
        , chip_(hardware, sxm_architecture)
        , program_loader_(ddr4_, chip_.icu(), program_load)
    {
        chip_.set_mem_read_timing(MemArrayModel::ReadTiming::RegisteredR1);
        program_loader_.set_drain_checker([this]{return !protocol_isolated() && chip_.icu().program_queues_done() && chip_.program_resources_drained();});
        program_loader_.set_program_reset([this]{chip_.stage_program_reset();});
        chip_.attach_c2c_ddr(ddr4_);chip_.attach_program_loader(program_loader_);
        for (std::size_t index = 0; index < hw::kHemispheres; ++index) {
            const auto hemisphere = static_cast<Hemisphere>(index);
            chip_.attach_c2c_dma(hemisphere, dmas_[index]);
        }
    }

    C2cDmaSystem(Ddr4Config ddr4_config, std::size_t dma_fifo_depth_vectors,
        SystemHardwareConfiguration hardware, SxmTransposeArchitecture sxm_architecture)
        : C2cDmaSystem(ddr4_config, dma_fifo_depth_vectors, hardware,
              ProgramLoadConfig{}, sxm_architecture) {}

    TspSliceSystem& chip() noexcept { return chip_; }
    const TspSliceSystem& chip() const noexcept { return chip_; }

    Ddr4Model& ddr4() noexcept { return ddr4_; }
    const Ddr4Model& ddr4() const noexcept { return ddr4_; }

    C2cDmaEngine& dma(Hemisphere hemisphere) noexcept
    {
        return dmas_[hemisphere_index(hemisphere)];
    }
    const C2cDmaEngine& dma(Hemisphere hemisphere) const noexcept
    {
        return dmas_[hemisphere_index(hemisphere)];
    }

    std::size_t cycle() const noexcept { return cycle_; }
    ProgramLoader& program_loader() noexcept { return program_loader_; }
    const ProgramLoader& program_loader() const noexcept { return program_loader_; }

    void tick(TspSliceSystem::LogSinks sinks = {})
    {
        if(protocol_isolated())throw AxiResponseProtocolError("chip DDR link isolated; platform reset required");
        chip_.tick(sinks);
        ++cycle_;
        program_loader_.observe_completion(cycle_);
    }

    bool data_error() const {
        return std::any_of(dmas_.begin(),dmas_.end(),[](const auto& d){return d.error().has_value();});
    }
    bool task_error_recovering() const {return task_error_recovering_;}
    bool protocol_isolated() const noexcept {return ddr4_.axi().isolated();}
    bool last_program_error() const {return last_program_error_;}
    const auto& last_data_errors() const {return last_data_errors_;}
    void begin_task_error_recovery() {
        if(protocol_isolated())throw AxiResponseProtocolError("cannot retire unidentified AXI response");
        if(task_error_recovering_)return;
        if(!data_error() && !program_loader_.failed())throw std::logic_error("no DMA task error to recover");
        last_program_error_=program_loader_.failed();
        for(std::size_t h=0;h<dmas_.size();++h) {
            last_data_errors_[h]=dmas_[h].error();dmas_[h].begin_task_abort();
        }
        program_loader_.begin_task_abort();
        chip_.abort_task_execution();task_error_recovering_=true;
    }
    bool finish_task_error_recovery() {
        if(!task_error_recovering_)throw std::logic_error("data task abort has not started");
        if(!ddr4_.drained() || !program_loader_.dma().drained())return false;
        for(const auto& dma:dmas_)if(!dma.error_drained())return false;
        for(auto& dma:dmas_)dma.recover_error();
        recover_program_error();task_error_recovering_=false;return true;
    }
    bool kernel_done() const {return !protocol_isolated() && !data_error() && !task_error_recovering_ && program_loader_.kernel_done();}
    bool can_load_program() const {return !protocol_isolated() && !data_error() && !task_error_recovering_ && program_loader_.can_load_program();}
    void recover_program_error() {
        if(!program_loader_.failed() || !ddr4_.drained() || !chip_.program_resources_drained())
            throw std::logic_error("drain external and compute resources before program error recovery");
        program_loader_.recover_failed_program();
    }

    // Prepare a standalone direct transport program after the previous task
    // drained. Device and shared DMA/DDR clocks are deliberately preserved.
    void clear_completed_program() {
        if(!can_load_program()) throw std::logic_error("previous program has not drained");
        chip_.stage_program_reset();
        // Host standalone transport packets are appended immediately after
        // this call. Commit the reset first, with issue/fetch held, so that
        // the next LOW cannot erase those newly appended packets. This is
        // one real physical reset edge; SRAM/DDR and monotonic time survive.
        chip_.icu().stage_program_loading(true);
        chip_.icu().stage_program_issue_enabled(false);
        tick();
        chip_.icu().stage_program_loading(false);
        chip_.icu().stage_program_issue_enabled(true);
    }

    void reset_execution_state()
    {
        if(!ddr4_.drained() || !chip_.program_resources_drained())
            throw std::logic_error("drain accepted traffic before execution reset");
        program_loader_.reset();
        chip_.reset_execution_state();
        ddr4_.reset_execution_state();
        cycle_ = 0;
    }

private:
    Ddr4Model ddr4_;
    std::array<C2cDmaEngine, hw::kHemispheres> dmas_;
    TspSliceSystem chip_{};
    ProgramLoader program_loader_;
    std::size_t cycle_{0};
    bool task_error_recovering_{};
    bool last_program_error_{};
    std::array<std::optional<C2cDmaEngine::Error>,hw::kHemispheres> last_data_errors_{};
};

} // namespace ftlpu
