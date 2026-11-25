#pragma once

#include "ftlpu/icu/instruction_fetch_dma.hpp"
#include "ftlpu/system/program_distribution_transport.hpp"
#include "ftlpu/system/program_load_command_controller.hpp"
#include "ftlpu/system/icu.hpp"

#include <limits>
#include <functional>
#include <ostream>

namespace ftlpu {

struct ProgramLoadConfig {
    std::size_t bus_bytes_per_cycle{hw::config::kProgramLoadBusBytesPerCycle};
    std::size_t route_latency_cycles{hw::config::kProgramLoadRouteLatencyCycles};
    std::size_t imem_write_ii_cycles{hw::config::kProgramLoadImemWriteIiCycles};
    std::size_t imem_write_commit_latency_cycles{
        hw::config::kProgramLoadImemWriteCommitLatencyCycles};

    void validate() const
    {
        for (auto value : {bus_bytes_per_cycle, route_latency_cycles,
                 imem_write_ii_cycles, imem_write_commit_latency_cycles})
            if (value == 0 || value > 2147483647)
                throw std::invalid_argument(
                    "program_load parameters must be positive integers <= 2147483647");
    }
};

struct ProgramImageStatistics {
    ProgramImage image{};
    std::optional<std::size_t> first_ddr_request{}, first_data_return{};
    std::optional<std::size_t> source_done{}, image_committed{};
    std::optional<std::size_t> first_write_accepted{}, first_write_committed{}, last_write_accepted{};
    std::size_t effective_bytes{0}, dma_bytes{0}, written_words{0};
};

struct ProgramStorageStatistics {
    std::size_t effective_bytes{0},dma_bytes{0},written_words{0};
    std::optional<std::size_t> last_commit{};
};
struct ProgramLoadStatistics {
    std::optional<std::size_t> request_accepted{}, first_ddr_request{}, first_data_return{};
    std::optional<std::size_t> all_images_committed{}, frontend_ready{}, program_start{}, kernel_done_cycle{};
    std::size_t effective_bytes{0}, dma_bytes{0}, written_words{0};
    std::size_t dma_credit_wait_cycles{0}, bus_wait_cycles{0}, target_wait_cycles{0};
    std::size_t peak_reassembly_bytes{0}, peak_dma_vectors{0};
    std::size_t peak_instruction_dma_outstanding{0};
    std::size_t peak_instruction_dma_response_fifo{0};
    std::size_t peak_word_fifo{0}, peak_ingress_fifo{0}, peak_relay_level{0};
    std::size_t word_fifo_full_cycles{0}, ingress_fifo_full_cycles{0};
    std::size_t relay_full_cycles{0};
    std::size_t word_fifo_writes{0}, word_fifo_reads{0};
    std::size_t ingress_fifo_writes{0}, ingress_fifo_reads{0};
    std::size_t relay_writes{0}, relay_reads{0};
    std::vector<ProgramImageStatistics> images{};
    std::array<ProgramStorageStatistics,6> stores{};

    void print(std::ostream& os) const
    {
        const auto stamp = [&](const char* name, auto value) {
            os << name << '=';
            if (value) os << *value; else os << "pending";
            os << '\n';
        };
        stamp("request_accepted", request_accepted);
        stamp("first_ddr_request", first_ddr_request);
        stamp("first_data_return", first_data_return);
        stamp("all_images_committed", all_images_committed);
        stamp("frontend_ready", frontend_ready);
        stamp("program_start", program_start);
        stamp("kernel_done", kernel_done_cycle);
        if (request_accepted && all_images_committed)
            os << "load_cycles=" << *all_images_committed - *request_accepted << '\n';
        if (all_images_committed && frontend_ready)
            os << "prefetch_cycles=" << *frontend_ready - *all_images_committed << '\n';
        for (std::size_t i = 0; i < images.size(); ++i) {
            os << "image[" << i << "] ";
            stamp("source_done", images[i].source_done);
            os << "image[" << i << "] ";
            stamp("image_committed", images[i].image_committed);
        }
        for(std::size_t i=0;i<stores.size();++i) {
            os << "storage[" << i << "] effective_bytes=" << stores[i].effective_bytes
               << " dma_bytes=" << stores[i].dma_bytes << " written_words=" << stores[i].written_words << ' ';
            stamp("last_commit",stores[i].last_commit);
        }
        os << "effective_bytes=" << effective_bytes << " dma_bytes=" << dma_bytes
           << " written_words=" << written_words
           << " dma_credit_wait_cycles=" << dma_credit_wait_cycles
           << " bus_wait_cycles=" << bus_wait_cycles
           << " target_wait_cycles=" << target_wait_cycles
           << " peak_instruction_dma_outstanding="
           << peak_instruction_dma_outstanding
           << " peak_instruction_dma_response_fifo="
           << peak_instruction_dma_response_fifo
           << " peak_word_fifo=" << peak_word_fifo
           << " peak_ingress_fifo=" << peak_ingress_fifo
           << " peak_relay_level=" << peak_relay_level
           << " word_fifo_full_cycles=" << word_fifo_full_cycles
           << " ingress_fifo_full_cycles=" << ingress_fifo_full_cycles
           << " relay_full_cycles=" << relay_full_cycles << '\n';
    }
};

// Pin-named view of one active Program Distribution -> MEM Relay endpoint.
// ProgramLoader serializes images, so this is a SEMANTIC adapter for the
// selected physical port rather than an array of all RTL ports.
struct MemProgramPortSignals {
    bool mem_program_valid_o{false};
    bool mem_program_ready_i{false};
    std::size_t mem_program_imem_address_o{0};
    std::array<std::uint8_t, hw::kProgramWordBytes> mem_program_data_o{};
    std::uint8_t mem_program_strb_o{0};
    bool mem_program_last_o{false};
    bool mem_program_error_i{false};
    bool mem_program_fire{false};
};

struct ProgramLoadWordInterfaceRecord {
    std::array<std::uint8_t, hw::kProgramWordBytes> data{};
    std::uint8_t strb{0};
    ProgramLoadScope scope{ProgramLoadScope::East};
    std::uint16_t destination{0};
    std::size_t imem_address{0};
    bool image_last{false};
};

// Formal PRE_POSEDGE interfaces for the R2 transport boundary.  The
// reassembler_image_complete field is a diagnostic derived from a final word
// fire; all valid/ready/fire and payload fields correspond to real RTL ports.
struct ProgramLoadR2HardwareInterfaceView {
    InstructionFetchDmaHardwareInterfaceView dma{};
    bool reassembler_word_valid_o{false};
    bool reassembler_word_ready_i{false};
    bool reassembler_word_fire{false};
    bool reassembler_image_complete{false};
    ProgramLoadWordInterfaceRecord reassembler_word{};
    bool word_fifo_enqueue_valid_i{false};
    bool word_fifo_enqueue_ready_o{false};
    bool word_fifo_enqueue_fire{false};
    ProgramLoadWordInterfaceRecord word_fifo_enqueue{};
    bool word_fifo_dequeue_valid_o{false};
    bool word_fifo_dequeue_ready_i{false};
    bool word_fifo_dequeue_fire{false};
    ProgramLoadWordInterfaceRecord word_fifo_dequeue{};
    std::size_t word_fifo_level_o{0};
};

struct ProgramLoadR3HardwareInterfaceView {
    ProgramDistributionHardwareInterfaceView distribution{};
};

struct ProgramLoadHardwareInterfaceView {
    ProgramLoadCommandHardwareInterfaceView command{};
    ProgramLoadConfigFifoHardwareInterfaceView config_fifo{};
    ProgramLoadR2HardwareInterfaceView r2{};
    ProgramLoadR3HardwareInterfaceView r3{};
    bool canonical_path{false};
    bool load_request_active{false};
    bool load_pipeline_idle{true};
    bool endpoint_write_commit_valid{false};
    bool distribution_idle{true};
    bool program_load_complete_o{false};
    bool program_ready_o{false};
    bool program_running_o{false};
    bool program_done_o{false};
    bool program_fault_o{false};
    std::uint8_t program_fault_code_o{0};
};

// One active program, one active destination. DDR is ticked by the enclosing
// system, exactly once per physical cycle. This engine shares that DDR with
// weight/data DMA. It has its own command, outstanding-ID and response state;
// the two DMA classes only contend through the shared Ddr4Model.
class ProgramLoader : public timing::ClockedModule {
public:
    static constexpr std::size_t kPrimeFifoDepth=32;
    enum class State { Idle, CommandIngress, LoadQueue, WaitCommit, AllCommitted,
        Prime, Ready, Started, Draining, Completed };
    static constexpr std::size_t kReassemblyBytes = 64;

    ProgramLoader(Ddr4Model& ddr, InstructionControlUnit& icu,
        std::size_t fifo_depth = hw::config::kDmaFifoDepthVectors,
        ProgramLoadConfig config = {})
        : dma_(ddr, InstructionFetchDmaConfig {
              256, 8, hw::config::kProgramLoadOutstandingDepth,
              fifo_depth})
        , ddr_(ddr), icu_(icu), config_(config)
    { config_.validate(); }

    void enable_clocked_mode()
    {
        if (clock_managed_ || next_snapshot_.has_value()
            || low_started_ || low_complete_)
            throw std::logic_error(
                "ProgramLoader clocked mode must be enabled before execution");
        if (!dma_.clocked_mode()) dma_.enable_clocked_mode();
        clock_managed_=true;
    }
    bool clocked_mode() const noexcept {return clock_managed_;}

    const MemProgramPortSignals& mem_program_port_signals() const noexcept
    {
        return mem_program_ports_;
    }
    const ProgramLoadHardwareInterfaceView&
    program_load_hardware_interface_view() const noexcept
    {
        return program_load_ports_;
    }
    ProgramLoadCommandController& command_controller() noexcept
    {
        return command_controller_;
    }
    ProgramLoadConfigFifo& config_fifo() noexcept { return config_fifo_; }
    ProgramDistributionTransport& distribution() noexcept
    {return distribution_;}
    const ProgramLoadCommandController& command_controller() const noexcept
    {
        return command_controller_;
    }
    const ProgramLoadConfigFifo& config_fifo() const noexcept
    {
        return config_fifo_;
    }
    const ProgramDistributionTransport& distribution() const noexcept
    {return distribution_;}

    // The ordered SystemPhase supplies physical time explicitly.  Clock only
    // validates and commits this transaction; it never calls LOW itself.
    void begin_clocked_low(std::size_t cycle)
    {
        if (!clock_managed_)
            throw std::logic_error("ProgramLoader is not clock managed");
        if (low_started_ || next_snapshot_.has_value())
            throw std::logic_error("ProgramLoader LOW executed more than once");
        low_config_enqueue_.reset();
        low_config_reset_=false;
        auto current=capture_snapshot();
        tick_before_dma(cycle);
        // Program Distribution is a physical clock participant even when the
        // upstream Word FIFO is idle.  Run one LOW every cycle so its formal
        // ready/level/idle interface remains observable instead of reverting
        // to a default-zero trace view.
        if(!distribution_.prepared())
            advance_rtl_distribution_idle(cycle,low_instruction_load_done_);
        low_observe_dma_=observe_dma_low_;
        next_snapshot_=capture_snapshot();
        restore_snapshot(std::move(current));
        observe_dma_low_=false;
        low_started_=true;
    }
    void advance_clocked_dma_low(std::size_t cycle)
    {
        if (!clock_managed_ || !low_started_ || low_complete_
            || !next_snapshot_.has_value())
            throw std::logic_error(
                "ProgramLoader DMA LOW is outside its SystemPhase slot");
        dma_.low();
        auto current=capture_snapshot();
        restore_snapshot(*next_snapshot_);
        observe_dma_low_=low_observe_dma_;
        observe_dma_after_low(cycle);
        finish_config_fifo_low();
        update_program_load_view();
        next_snapshot_=capture_snapshot();
        restore_snapshot(std::move(current));
        observe_dma_low_=false;
        low_complete_=true;
    }
    void low() override
    {
        throw std::logic_error(
            "ProgramLoader LOW requires SystemPhase physical-cycle input");
    }
    void validate_posedge() const override
    {
        if (clock_managed_ && (!low_started_ || !low_complete_
            || !next_snapshot_.has_value()))
            throw std::logic_error(
                "ProgramLoader posedge requires one complete LOW");
    }
    void posedge() override
    {
        if (!clock_managed_) return;
        validate_posedge();
        if(distribution_.prepared()) distribution_.posedge();
        restore_snapshot(std::move(*next_snapshot_));
        next_snapshot_.reset();
        low_started_=low_complete_=low_observe_dma_=false;
    }
    void high() override {distribution_.high();}

    void configure(ProgramLoadConfig config)
    {
        if (busy()) throw std::logic_error("program loader is busy");
        config.validate();
        config_ = config;
    }
    const ProgramLoadConfig& config() const noexcept { return config_; }
    void set_program_reset(std::function<void()> reset) {program_reset_=std::move(reset);}
    State state() const noexcept { return state_; }
    bool busy() const noexcept
    { return canonical_submission_active_
        || state_ == State::CommandIngress
        || state_ == State::LoadQueue || state_ == State::WaitCommit
        || state_ == State::AllCommitted || state_ == State::Prime || state_ == State::Ready; }
    bool all_images_committed() const noexcept
    { return stats_.all_images_committed.has_value(); }
    bool ready_to_start() const noexcept { return state_ == State::Ready; }
    bool started() const noexcept { return stats_.program_start.has_value(); }
    bool kernel_done() const noexcept {return stats_.kernel_done_cycle.has_value();}
    void set_drain_checker(std::function<bool()> checker) {drain_checker_=std::move(checker);}
    bool can_load_program() const {
        return !busy() && (!started() || kernel_done()) && dma_.drained()
            && config_fifo_.empty() && !command_controller_.busy()
            && commits_.empty() && word_fifo_.empty() && prime_fifo_.empty()
            && distribution_.idle() && reassembly_.empty()
            && std::none_of(route_.begin(),route_.end(),[](const auto& p){return p.has_value();})
            && icu_.program_queues_done() && ddr_.drained() && (!drain_checker_ || drain_checker_());
    }
    void observe_completion(std::size_t edge) {
        if(!started() || kernel_done()) return;
        if(icu_.program_queues_done()) state_=State::Draining;
        if(state_==State::Draining && dma_.drained() && ddr_.drained()
            && (!drain_checker_ || drain_checker_())) {
            stats_.kernel_done_cycle=edge;state_=State::Completed;
        }
    }
    std::uint64_t last_generation() const noexcept {return std::max(last_generation_,icu_.program_generation());}
    void begin_direct_program(std::size_t cycle,std::uint64_t generation=0) {
        last_generation_=std::max(last_generation_,generation);
        stats_={};stats_.program_start=cycle;state_=State::Started;
        images_.clear();route_.clear();reassembly_.clear();commits_.clear();
        word_fifo_.clear();prime_fifo_.clear();distribution_.reset();
        instruction_load_done_=false;
        icu_.stage_program_loading(false);
        icu_.stage_program_issue_enabled(true);
    }
    const ProgramLoadStatistics& statistics() const noexcept { return stats_; }
    InstructionFetchDma& dma() noexcept { return dma_; }
    const InstructionFetchDma& dma() const noexcept { return dma_; }

    static std::pair<std::size_t,std::size_t> relay_destination_for_test(
        const ProgramImage& image)
    {
        return relay_destination(image);
    }

    void reset()
    {
        dma_.reset();
        command_controller_.reset();
        config_fifo_.reset();
        distribution_.reset();
        host_adapter_.reset();
        state_ = State::Idle;
        stats_ = {};
        last_generation_=0;
        images_.clear(); route_.clear(); reassembly_.clear(); commits_.clear();
        word_fifo_.clear();prime_fifo_.clear();
        instruction_load_done_=false;
        distribution_.reset();
        start_requested_ = dependencies_ready_ = false;
        next_snapshot_.reset();
        low_started_=low_complete_=low_observe_dma_=false;
        observe_dma_low_=false;
        pending_images_.clear();
        pending_generation_.reset();
        active_config_.reset();
        canonical_submission_active_=false;
        load_request_active_=descriptor_fault_=false;
        program_load_ports_={};
    }

    // Validation precedes any mutation, including clearing the old program.
    void submit(std::vector<ProgramImage> images, std::size_t physical_cycle)
    {
        if (!can_load_program()) throw std::logic_error("previous program has not completed and drained");
        const auto group=[](ProgramStorage store) {
            return store==ProgramStorage::Imem?0:store==ProgramStorage::TransportImem?1:
                (store==ProgramStorage::Qctl || store==ProgramStorage::Barr)?2:3;
        };
        bool qctl=false,barr=false;
        for (std::size_t i = 0; i < images.size(); ++i) {
            qctl|=images[i].storage==ProgramStorage::Qctl || images[i].storage==ProgramStorage::QueueEvents;
            barr|=images[i].storage==ProgramStorage::Barr || images[i].storage==ProgramStorage::BarrierEvents;
            if((i && group(images[i].storage)<group(images[i-1].storage)) || (qctl && barr))
                throw std::invalid_argument("invalid program storage order or mixed control protocols");
            validate_image(images[i]);
            for (std::size_t j = 0; j < i; ++j)
                if (images[i].key() == images[j].key())
                    throw std::invalid_argument("duplicate program queue destination");
        }
        bool queueEvents=false,barrierEvents=false,barrierRam=false;
        std::optional<std::uint64_t> generation;
        for(const auto& image:images) {
            if(!image.generation) continue;
            if(generation && *generation!=image.generation) throw std::invalid_argument("mixed task generations");
            generation=image.generation;
            queueEvents|=image.storage==ProgramStorage::QueueEvents;
            barrierEvents|=image.storage==ProgramStorage::BarrierEvents;
            barrierRam|=image.storage==ProgramStorage::Barr;
            if(image.storage==ProgramStorage::Qctl && !std::any_of(images.begin(),images.end(),[&](const auto& imem) {
                return imem.storage==ProgramStorage::Imem && imem.mode==IcuQueueMode::Macro
                    && imem.kind==image.kind && imem.queue_index==image.queue_index;
            })) throw std::invalid_argument("QCTL partition has no matching Macro i-MEM");
        }
        // Native-only callers may omit the task generation; allocate a fresh
        // watermark so retired control notifications also stay harmless here.
        if(!generation) {
            if(last_generation()==UINT64_MAX) throw std::overflow_error("program generation exhausted");
            generation=last_generation()+1;
        }
        if(generation && *generation<=last_generation())
            throw std::invalid_argument("program generation must increase across task switches");
        if((qctl && !queueEvents) || (barr && (!barrierEvents || !barrierRam)))
            throw std::invalid_argument("missing committed event or barrier configuration");
        if (!dma_.drained())
            throw std::logic_error("program DMA still has unconsumed traffic");
        std::size_t total_bytes = 0;
        for (const auto& image : images) {
            if (total_bytes > std::numeric_limits<std::size_t>::max() - image.byte_count)
                throw std::overflow_error("aggregate program image size overflow");
            total_bytes += image.byte_count;
        }
        const bool canonical = !images.empty()
            && std::all_of(images.begin(), images.end(),
                [](const auto& image) { return uses_rtl_distribution(image); });
        if (canonical) {
            std::vector<std::pair<ProgramImage,std::uint16_t>> descriptors;
            descriptors.reserve(images.size());
            for (const auto& image : images)
                descriptors.emplace_back(image, program_destination(image));
            const auto commands = make_program_load_command_list(descriptors);
            pending_images_ = std::move(images);
            pending_generation_ = generation;
            pending_submission_cycle_ = physical_cycle;
            canonical_submission_active_ = true;
            instruction_load_done_ = false;
            prime_fifo_.clear();
            initialize_canonical_statistics();
            start_requested_ = false;
            dependencies_ready_ = false;
            host_adapter_.submit(commands);
            return;
        }

        // Explicit compatibility path for Native/control-store ProgramImages.
        // CModelRuntime::load_timed() uses the canonical Macro path above;
        // these historical test/legacy formats have no RTL Program descriptor.
        if(program_reset_) program_reset_(); else icu_.stage_program_reset();
        if(generation) {last_generation_=*generation;icu_.stage_program_generation(*generation);}
        images_ = std::move(images);
        stats_ = {};
        stats_.request_accepted = physical_cycle;
        for (auto image : images_) {
            stats_.images.push_back(ProgramImageStatistics{image});
            stats_.images.back().effective_bytes = image.byte_count;
            stats_.effective_bytes += image.byte_count;
            stats_.stores[static_cast<std::size_t>(image.storage)].effective_bytes+=image.byte_count;
        }
        route_.assign(config_.route_latency_cycles, std::nullopt);
        reassembly_.clear();
        commits_.clear();
        word_fifo_.clear();
        distribution_.reset();
        current_ = 0;
        start_requested_ = false;
        dependencies_ready_ = false;
        if (images_.empty()) {
            stats_.all_images_committed = physical_cycle;
            state_ = State::AllCommitted;
        } else begin_queue();
    }

    // An early request remains pending. The external dependency gate must also
    // be true on the physical cycle in which execution actually starts.
    void start() noexcept { start_requested_ = true; }
    void set_dependencies_ready(bool ready) noexcept { dependencies_ready_ = ready; }

    void tick(std::size_t cycle)
    {
        if (clock_managed_)
            throw std::logic_error(
                "clock-managed ProgramLoader must run through TspSliceSystem");
        low_config_enqueue_.reset();
        low_config_reset_=false;
        tick_before_dma(cycle);
        dma_.tick();
        observe_dma_after_low(cycle);
        finish_config_fifo_low();
        command_controller_.validate_posedge();
        config_fifo_.validate_posedge();
        command_controller_.posedge();
        config_fifo_.posedge();
        if(distribution_.prepared()) distribution_.posedge();
        update_program_load_view();
    }

private:
    void tick_before_dma(std::size_t cycle)
    {
        observe_dma_low_=false;
        mem_program_ports_ = {};
        program_load_ports_.r2={};
        program_load_ports_.r2.reassembler_word_ready_i=
            word_fifo_.size()<hw::config::kProgramLoadWordFifoDepth;
        program_load_ports_.r2.word_fifo_enqueue_ready_o=
            program_load_ports_.r2.reassembler_word_ready_i;
        program_load_ports_.r3={};
        program_load_ports_.endpoint_write_commit_valid=!commits_.empty()
            &&commits_.front().due==cycle;
        low_state_for_view_=state_;
        low_instruction_load_done_=instruction_load_done_;
        low_distribution_idle_=distribution_.idle()
            &&!program_load_ports_.endpoint_write_commit_valid;
        low_load_request_active_=load_request_active_;
        low_descriptor_fault_=descriptor_fault_;
        low_program_reset_=false;
        // LOW at cycle C prepares state committed by edge C+1.
        observe_completion(cycle+1);
        if (canonical_submission_active_ || load_request_active_) {
            tick_canonical_before_dma(cycle);
            return;
        }

        // The command controller remains a real Clock participant while a
        // compatibility ProgramImage is active (and while the loader is idle).
        ProgramLoadCommandInputs idle{};
        idle.program_cmd_write_enable_i=false;
        idle.program_load_start_enable_i=false;
        idle.load_pipeline_idle_i=load_pipeline_idle();
        command_controller_.drive(idle);
        command_controller_.low();
        program_load_ports_.command=
            command_controller_.hardware_interface_view();
        if (!busy()) {
            update_program_load_view();
            return;
        }
        icu_.stage_program_issue_enabled(false);
        if (state_ == State::Ready) {
            if (start_requested_ && dependencies_ready_) {
                stats_.program_start = cycle;
                state_ = State::Started;
                icu_.stage_program_loading(false);
                icu_.stage_program_issue_enabled(true);
                // An empty/control-only task may both start and retire at this
                // edge; this is still one atomic Next-state transition.
                observe_completion(cycle+1);
            }
            return;
        }
        if (state_ == State::AllCommitted) { state_ = State::Prime; }
        if (state_ == State::Prime) {
            icu_.stage_program_frontend_tick();
            if (icu_.program_frontends_ready()) {
                stats_.frontend_ready = cycle;
                state_ = State::Ready;
            }
            return;
        }

        auto& record = stats_.images[current_];
        const auto& image = images_[current_];
        // Commit is independent of accepting the next word: latency is a
        // pipeline, never a per-word round trip.
        while (!commits_.empty() && commits_.front().due <= cycle) {
            if (!record.first_write_committed) record.first_write_committed = cycle;
            commit_word(image, commits_.front());
            commits_.pop_front();
            ++record.written_words;
            ++stats_.written_words;
            ++stats_.stores[static_cast<std::size_t>(image.storage)].written_words;
            stats_.stores[static_cast<std::size_t>(image.storage)].last_commit=cycle;
        }
        if (record.written_words == image.word_count()) {
            configure_target(image);
            record.image_committed = cycle;
            if(!record.source_done) record.source_done=cycle;
            if (++current_ == images_.size()) {
                stats_.all_images_committed = cycle;
                state_ = State::AllCommitted;
            } else {
                begin_queue();
                // begin_queue() stages the next InstructionFetchDma command.
                // Its first DDR request can fire in the DMA LOW later in this
                // same SystemPhase, so this transition cycle must participate
                // in the post-DMA observation just like an ordinary load cycle.
                observe_dma_low_ = images_[current_].word_count() != 0;
            }
            return; // queue n+1 cannot issue before queue n commits
        }

        const bool rtl_distribution=uses_rtl_distribution(image);
        if (rtl_distribution)
            advance_rtl_distribution(image,record,cycle);

        if (reassembly_.size() >= image.word_bytes()) {
            const bool destination_ready = rtl_distribution
                ? word_fifo_.size() < hw::config::kProgramLoadWordFifoDepth
                : cycle >= next_write_cycle_
                    && commits_.size()
                        <= config_.imem_write_commit_latency_cycles;
            if (destination_ready) {
                PendingWrite write{};
                write.program_imem_address = accepted_words_++;
                for (std::size_t b = 0; b < image.word_bytes(); ++b) {
                    write.bytes[b] = reassembly_.front();
                    reassembly_.pop_front();
                }
                if (rtl_distribution) {
                    word_fifo_.push_back(write);
                    ++stats_.word_fifo_writes;
                } else {
                    write.due = cycle
                        + config_.imem_write_commit_latency_cycles;
                    commits_.push_back(write);
                    if (!record.first_write_accepted)
                        record.first_write_accepted = cycle;
                    record.last_write_accepted = cycle;
                    next_write_cycle_ = cycle
                        + config_.imem_write_ii_cycles;
                }
            } else ++stats_.target_wait_cycles;
        }

        // Elastic route stages. A blocked endpoint holds every upstream stage.
        if (route_.back() && reassembly_.size() + route_.back()->size <= kReassemblyBytes) {
            const auto& packet = *route_.back();
            for (std::size_t b = 0; b < packet.size; ++b)
                reassembly_.push_back(packet.bytes[b]);
            route_.back().reset();
        }
        for (std::size_t s = route_.size() - 1; s > 0; --s)
            if (!route_[s] && route_[s - 1]) {
                route_[s] = std::move(route_[s - 1]);
                route_[s - 1].reset();
            }
        if (dma_.word_ready() && routed_bytes_ < image.byte_count) {
            if (!route_.front()) {
                Packet packet{};
                packet.size = std::min({config_.bus_bytes_per_cycle,
                    image.word_bytes() - vector_offset_,
                    image.byte_count - routed_bytes_});
                const auto word = dma_.front_word();
                for (std::size_t b = 0; b < packet.size; ++b)
                    packet.bytes[b] = word.bytes[vector_offset_ + b];
                vector_offset_ += packet.size;
                routed_bytes_ += packet.size;
                if (vector_offset_ == image.word_bytes()) {
                    dma_.pop_word();
                    vector_offset_ = 0;
                }
                route_.front() = packet;
                if (routed_bytes_ < image.byte_count && vector_offset_ != 0)
                    ++stats_.bus_wait_cycles;
            }
        }

        observe_dma_low_=true;
        update_program_load_view();
    }

    void tick_canonical_before_dma(std::size_t cycle)
    {
        const bool execution_active=state_==State::Started
            || state_==State::Draining || state_==State::Completed;
        icu_.stage_program_issue_enabled(execution_active);

        ProgramLoadCommandInputs inputs{};
        host_adapter_.drive(inputs);
        const bool system_ready = !command_controller_.busy()
            && !load_request_active_ && !descriptor_fault_;
        inputs.program_cmd_write_enable_i=system_ready;
        inputs.program_load_start_enable_i=system_ready;
        // Ready is a property of the DMA command/config enqueue sinks, not of
        // the controller's current opcode.  Keep it combinationally observable
        // while valid is low, matching lpu_program_load_path.sv; fire remains
        // valid && ready inside ProgramLoadCommandController.
        inputs.ctrl_load_txn_ready_i=dma_.can_accept_program()
            && !config_fifo_.full();
        inputs.ctrl_prime_ready_i=
            command_controller_.ctrl_prime_valid_current()
            &&load_pipeline_idle()&&prime_fifo_.size()<kPrimeFifoDepth;
        inputs.load_pipeline_idle_i=load_pipeline_idle();
        command_controller_.drive(inputs);
        command_controller_.low();
        const auto command=command_controller_.hardware_interface_view();
        program_load_ports_.command=command;
        host_adapter_.observe(command);

        if (inputs.reset_i) {
            low_program_reset_=true;
            // Canonical reset is a ProgramLoader LOW output.  Stage it directly
            // into the peer ICU transaction; the external callback is only for
            // host-side compatibility submits at an idle TSP boundary.
            icu_.stage_program_reset();
            state_=State::Idle;
            initialize_canonical_statistics();
            images_.clear();
            route_.clear();
            reassembly_.clear();
            commits_.clear();
            word_fifo_.clear();
            prime_fifo_.clear();
            instruction_load_done_=false;
            prepare_distribution_reset();
            // The RTL Word FIFO dequeue Ready is a combinational path through
            // the empty Distributor and is not gated by program_reset_i.
            program_load_ports_.r2.word_fifo_dequeue_ready_i=true;
            current_=routed_bytes_=vector_offset_=accepted_words_=
                next_write_cycle_=0;
            active_config_.reset();
            load_request_active_=false;
            descriptor_fault_=false;
            low_config_reset_=true;
            update_program_load_view();
            return;
        }

        if (command.fault_o) descriptor_fault_=true;
        if (command.program_load_start_fire) {
            if (!pending_generation_ || pending_images_.empty())
                throw std::logic_error(
                    "Program Load start has no pending canonical image set");
            images_=pending_images_;
            last_generation_=*pending_generation_;
            icu_.stage_program_generation(*pending_generation_);
            initialize_canonical_statistics();
            route_.assign(config_.route_latency_cycles,std::nullopt);
            reassembly_.clear();
            commits_.clear();
            word_fifo_.clear();
            prime_fifo_.clear();
            instruction_load_done_=false;
            if(!distribution_.idle())
                throw std::logic_error(
                    "Program Distribution was not empty at Program Load start");
            current_=routed_bytes_=vector_offset_=accepted_words_=
                next_write_cycle_=0;
            active_config_.reset();
            load_request_active_=true;
            state_=State::CommandIngress;
        }

        // Program Load owns the machine only through PRIME/START.  Once START
        // has been accepted, the FU-local Frontends and Scheduler advance the
        // resident program continuously; the still-high load lifecycle state
        // must not feed command DONE back into AllCommitted and re-run PRIME.
        if (state_==State::Started || state_==State::Draining
            || state_==State::Completed) {
            update_program_load_view();
            return;
        }

        if (state_==State::Ready) {
            if (start_requested_ && dependencies_ready_) {
                stats_.program_start=cycle;
                state_=State::Started;
                icu_.stage_program_loading(false);
                icu_.stage_program_issue_enabled(true);
                observe_completion(cycle+1);
            }
            update_program_load_view();
            return;
        }
        if (state_==State::AllCommitted) state_=State::Prime;
        if (state_==State::Prime) {
            if (!prime_fifo_.empty() || !distribution_.idle()) return;
            icu_.stage_program_frontend_tick();
            if (icu_.program_frontends_ready()) {
                stats_.frontend_ready=cycle;
                state_=State::Ready;
            }
            update_program_load_view();
            return;
        }

        if (command.ctrl_load_txn_fire) {
            if (current_>=images_.size())
                throw std::logic_error(
                    "Program Load emitted LOAD after the final image");
            const auto& image=images_.at(current_);
            validate_load_transaction(command,image);
            routed_bytes_=vector_offset_=accepted_words_=next_write_cycle_=0;
            dma_.start_with_id(InstructionFetchProgram{
                command.ctrl_load_txn_source_address_o,
                command.ctrl_load_txn_expected_word_count_o,
                command.ctrl_load_txn_format_o==ProgramLoadFormat::Word96
                    ? PhysicalInstructionWordWidth::Bits96
                    : PhysicalInstructionWordWidth::Bits128},
                command.ctrl_load_txn_dma_id_o);
            low_config_enqueue_=ProgramLoadConfigDescriptor{
                command.ctrl_load_txn_source_address_o,
                command.ctrl_load_txn_byte_count_o,
                command.ctrl_load_txn_scope_o,
                command.ctrl_load_txn_destination_o,
                command.ctrl_load_txn_format_o,
                command.ctrl_load_txn_imem_base_address_o,
                command.ctrl_load_txn_image_first_o,
                command.ctrl_load_txn_image_last_o,
                command.ctrl_load_txn_expected_word_count_o,
                command.ctrl_load_txn_dma_id_o};
            state_=State::LoadQueue;
            observe_dma_low_=true;
        }

        // The Reassembler consumes the real Config FIFO head and latches the
        // descriptor while the corresponding response transaction is active.
        // ProgramImage remains only a host-side diagnostic/reference record.
        if (!active_config_ && !config_fifo_.empty()
            && current_<images_.size()) {
            validate_config_descriptor(config_fifo_.front(),
                images_.at(current_));
            active_config_=config_fifo_.front();
        }

        if (command.ctrl_prime_fire) {
            if (current_>=images_.size())
                throw std::logic_error(
                    "Program Load emitted COMMIT after the final image");
            if(!active_config_)
                throw std::logic_error(
                    "COMMIT_QUEUE has no active Reassembler descriptor");
            ProgramDistributionRecord prime{};
            prime.control_prime=true;
            prime.scope=command.ctrl_prime_scope_o;
            prime.destination=command.ctrl_prime_destination_o;
            prime.image_last=true;
            prime_fifo_.push_back(prime);
            active_config_.reset();
            ++current_;
            state_=State::CommandIngress;
            update_program_load_view();
            return;
        }

        if ((state_==State::LoadQueue || state_==State::WaitCommit)
            && current_<images_.size() && active_config_) {
            auto& record=stats_.images.at(current_);
            const auto image=program_image_from_config(*active_config_);
            while(!commits_.empty() && commits_.front().due<=cycle) {
                if(!record.first_write_committed)
                    record.first_write_committed=cycle;
                commits_.pop_front();
                ++record.written_words;
                ++stats_.written_words;
                ++stats_.stores[static_cast<std::size_t>(image.storage)]
                    .written_words;
                stats_.stores[static_cast<std::size_t>(image.storage)]
                    .last_commit=cycle;
            }
            if(record.written_words==image.word_count()
                && !record.image_committed) {
                record.image_committed=cycle;
                if(!record.source_done) record.source_done=cycle;
                state_=State::WaitCommit;
            }
            if(!record.image_committed) {
                advance_rtl_distribution(image,record,cycle,true);
                advance_canonical_reassembler_word(image);
            }
            observe_dma_low_=true;
        }

        if (command.done_o && current_==images_.size()
            && load_pipeline_idle()) {
            instruction_load_done_=true;
            if(!stats_.all_images_committed)
                stats_.all_images_committed=cycle;
            canonical_submission_active_=false;
            pending_images_.clear();
            pending_generation_.reset();
            state_=State::AllCommitted;
        }
        update_program_load_view();
    }

    void observe_dma_after_low(std::size_t cycle)
    {
        program_load_ports_.r2.dma=dma_.hardware_interface_view();
        if (!observe_dma_low_) return;
        auto& record = stats_.images.at(current_);
        const auto& image = images_.at(current_);
        if (dma_.low_last_read_request()) {
            if (!stats_.first_ddr_request) stats_.first_ddr_request = cycle;
            if (!record.first_ddr_request) record.first_ddr_request = cycle;
            record.dma_bytes += hw::kPhysicalVectorBytes;
            stats_.dma_bytes += hw::kPhysicalVectorBytes;
            stats_.stores[static_cast<std::size_t>(image.storage)].dma_bytes+=hw::kPhysicalVectorBytes;
        }
        if (dma_.low_last_beat()) {
            if (!stats_.first_data_return) stats_.first_data_return = cycle;
            if (!record.first_data_return) record.first_data_return = cycle;
        }
        if (dma_.low_load_credit_wait()) ++stats_.dma_credit_wait_cycles;
        if (dma_.fault())
            throw std::runtime_error("instruction DMA fault code "
                + std::to_string(dma_.fault_code()));
        if (dma_.low_idle() && !record.source_done) {
            record.source_done = cycle;
            state_ = State::WaitCommit;
        }
        stats_.peak_reassembly_bytes = std::max(stats_.peak_reassembly_bytes, reassembly_.size());
        stats_.peak_dma_vectors = std::max(stats_.peak_dma_vectors,
            dma_.low_receive_queue_size());
        stats_.peak_instruction_dma_outstanding = std::max(
            stats_.peak_instruction_dma_outstanding,
            dma_.low_outstanding_reads());
        stats_.peak_instruction_dma_response_fifo = std::max(
            stats_.peak_instruction_dma_response_fifo,
            dma_.low_receive_queue_size());
    }

    void finish_config_fifo_low()
    {
        ProgramLoadConfigFifoInputs inputs{};
        inputs.reset_i=low_config_reset_;
        inputs.enqueue_valid_i=low_config_enqueue_.has_value();
        inputs.enqueue_data_i=low_config_enqueue_;
        inputs.dequeue_ready_i=load_request_active_
            && dma_.low_reassembler_response_fire()
            && dma_.low_reassembler_response_last();
        if(inputs.dequeue_ready_i) {
            if(config_fifo_.empty())
                throw std::logic_error(
                    "Instruction DMA final response has no Config FIFO head");
            if(config_fifo_.front().dma_id
                !=dma_.low_reassembler_response_id())
                throw std::logic_error(
                    "Instruction DMA final response ID mismatches Config FIFO");
        }
        config_fifo_.drive(std::move(inputs));
        config_fifo_.low();
        program_load_ports_.config_fifo=
            config_fifo_.hardware_interface_view();
    }

    void prepare_distribution_reset()
    {
        if(distribution_.prepared())
            throw std::logic_error(
                "Program Distribution reset was staged twice in one LOW");
        ProgramDistributionInputs inputs{};
        inputs.reset=true;
        distribution_.drive(inputs);
        distribution_.low();
        program_load_ports_.r3.distribution=
            distribution_.hardware_interface_view();
    }

    void advance_rtl_distribution_idle(std::size_t cycle,
        bool instructionLoadDoneCurrent)
    {
        const bool endpointReady=cycle>=next_write_cycle_
            &&commits_.size()<=config_.imem_write_commit_latency_cycles;
        ProgramDistributionInputs inputs{};
        inputs.load_record_valid=instructionLoadDoneCurrent
            &&!prime_fifo_.empty();
        if(inputs.load_record_valid) inputs.load_record=prime_fifo_.front();
        inputs.endpoint_ready=[this,endpointReady](const auto& record) {
            return endpoint_ready(record,endpointReady);
        };
        inputs.outer_endpoint_ready.fill(endpointReady);
        inputs.vxm_endpoint_ready=endpointReady;
        distribution_.drive(inputs);
        distribution_.low();
        const auto& view=distribution_.hardware_interface_view();
        program_load_ports_.r3.distribution=view;
        program_load_ports_.r2.word_fifo_level_o=word_fifo_.size();
        program_load_ports_.r2.word_fifo_dequeue_ready_i=
            !inputs.load_record_valid && view.load_record_input.ready;
        if (!images_.empty()) {
            const auto image=std::find_if(images_.begin(),images_.end(),
                [](const auto& candidate) {
                    return candidate.kind==IcuMacroQueueKind::Mem;
                });
            if (image!=images_.end()) {
                const auto [hemisphere,relay]=relay_destination(*image);
                mem_program_ports_.mem_program_ready_i=
                    view.mem_endpoint_input[hemisphere][relay].ready;
            }
        }
        if(view.load_record_input.fire) prime_fifo_.pop_front();
        for(const auto& hemisphere:view.mem_endpoint_input)
            for(const auto& endpoint:hemisphere)
                accept_prime(endpoint);
        for(const auto& endpoint:view.outer_endpoint_input)
            accept_prime(endpoint);
    }

public:
    // Program-port combinational Ready, shared by distribution wiring and
    // endpoint-level drivers. This reads the selected Queue's Current state.
    bool endpoint_ready(const ProgramDistributionRecord& record,
        bool write_capacity)
    {
        const auto kind=(record.destination>>8)&15U;
        if(kind!=0 && kind!=1 && kind!=2 && kind!=3 && kind!=7 && kind!=8)
            return write_capacity;
        const auto hemisphere=(record.destination>>12)&1U;
        const auto endpoint_kind=kind==0?IcuMacroQueueKind::Mem:
            kind==1?IcuMacroQueueKind::MxmLoad:
            kind==2?IcuMacroQueueKind::MxmDequant:
            kind==3?IcuMacroQueueKind::MxmCompute:
            kind==7?IcuMacroQueueKind::MxmParamLoad:IcuMacroQueueKind::MxmBackend;
        const auto width=kind==0?InstructionControlUnit::kMemQueuesPerHemisphere:
            InstructionControlUnit::kMxmQueuesPerHemisphere;
        const auto local=record.destination&0xffU;
        if(hemisphere>=hw::kHemispheres || local>=width) return true;
        bool ready=false;
        icu_.visit_program_queue(endpoint_kind,hemisphere*width+local,false,
            [&](const auto& queue) { ready=queue.program_endpoint_state().ready(); });
        return ready && (record.control_prime || write_capacity);
    }

private:
    void accept_prime(const ProgramDistributionChannelView& endpoint)
    {
        if(!endpoint.fire || !endpoint.payload.control_prime) return;
        const auto image=std::find_if(images_.begin(),images_.end(),[&](const auto& target) {
            return program_destination(target)==endpoint.payload.destination;
        });
        if(image==images_.end())
            throw std::logic_error("PRIME reached an unconfigured Program Endpoint");
        configure_target(*image);
    }

    bool distribution_idle() const noexcept
    {
        return reassembly_.empty() && commits_.empty()
            && word_fifo_.empty() && distribution_.idle()
            && std::none_of(route_.begin(),route_.end(),
                [](const auto& packet){return packet.has_value();})
            ;
    }

    bool load_pipeline_idle() const noexcept
    {
        return dma_.drained() && config_fifo_.empty()
            && distribution_idle();
    }

    void update_program_load_view()
    {
        program_load_ports_.canonical_path=canonical_submission_active_
            || low_load_request_active_;
        program_load_ports_.load_request_active=low_load_request_active_;
        program_load_ports_.load_pipeline_idle=
            program_load_ports_.command.load_pipeline_idle_i;
        program_load_ports_.distribution_idle=low_distribution_idle_;
        program_load_ports_.program_load_complete_o=
            !low_program_reset_ && low_instruction_load_done_
            && !low_descriptor_fault_
            && !program_load_ports_.command.fault_o && !dma_.fault()
            && !distribution_.fault();
        const bool program_done=started()
            &&(low_state_for_view_==State::Draining
                ||low_state_for_view_==State::Completed);
        program_load_ports_.program_ready_o=!started()
            &&
            program_load_ports_.program_load_complete_o
            && icu_.program_frontends_ready();
        program_load_ports_.program_running_o=started() && !program_done;
        program_load_ports_.program_done_o=program_done;
        program_load_ports_.program_fault_o=low_descriptor_fault_
            || program_load_ports_.command.fault_o || dma_.fault()
            || distribution_.fault();
        program_load_ports_.program_fault_code_o=low_descriptor_fault_
            ? static_cast<std::uint8_t>(ProgramLoadFaultCode::Descriptor)
            : program_load_ports_.command.fault_o
                ? static_cast<std::uint8_t>(ProgramLoadFaultCode::Command)
                : distribution_.fault()
                    ? static_cast<std::uint8_t>(ProgramLoadFaultCode::Route)
                    : dma_.fault_code();
    }

    static void validate_load_transaction(
        const ProgramLoadCommandHardwareInterfaceView& command,
        const ProgramImage& image)
    {
        const auto expected_format=image.word_bytes()==12
            ? ProgramLoadFormat::Word96 : ProgramLoadFormat::Word128;
        if(command.ctrl_load_txn_source_address_o!=image.source_address
            || command.ctrl_load_txn_byte_count_o!=image.byte_count
            || command.ctrl_load_txn_destination_o
                !=program_destination(image)
            || command.ctrl_load_txn_format_o!=expected_format
            || command.ctrl_load_txn_expected_word_count_o
                !=image.word_count())
            throw std::logic_error(
                "Program Load command no longer matches submitted image");
    }

    static ProgramImage program_image_from_config(
        const ProgramLoadConfigDescriptor& descriptor)
    {
        const auto hemisphere=
            static_cast<std::size_t>(descriptor.destination>>12)&1;
        const auto kind_bits=
            static_cast<std::uint8_t>((descriptor.destination>>8)&15);
        const auto local=static_cast<std::size_t>(descriptor.destination&0xff);
        IcuMacroQueueKind kind{};
        std::size_t queues_per_hemisphere=0;
        switch(kind_bits) {
        case 0:
            kind=IcuMacroQueueKind::Mem;
            queues_per_hemisphere=InstructionControlUnit::kMemQueuesPerHemisphere;
            break;
        case 1:
            kind=IcuMacroQueueKind::MxmLoad;
            queues_per_hemisphere=InstructionControlUnit::kMxmQueuesPerHemisphere;
            break;
        case 2:
            kind=IcuMacroQueueKind::MxmDequant;
            queues_per_hemisphere=InstructionControlUnit::kMxmQueuesPerHemisphere;
            break;
        case 3:
            kind=IcuMacroQueueKind::MxmCompute;
            queues_per_hemisphere=InstructionControlUnit::kMxmQueuesPerHemisphere;
            break;
        case 7:
            kind=IcuMacroQueueKind::MxmParamLoad;
            queues_per_hemisphere=InstructionControlUnit::kMxmQueuesPerHemisphere;
            break;
        case 8:
            kind=IcuMacroQueueKind::MxmBackend;
            queues_per_hemisphere=InstructionControlUnit::kMxmQueuesPerHemisphere;
            break;
        default:
            throw std::logic_error(
                "Config FIFO descriptor has an unsupported destination kind");
        }
        if(local>=queues_per_hemisphere)
            throw std::logic_error(
                "Config FIFO descriptor local destination is out of range");
        ProgramImage image{};
        image.source_address=descriptor.source_address;
        image.kind=kind;
        image.queue_index=hemisphere*queues_per_hemisphere+local;
        image.byte_count=descriptor.byte_count;
        image.mode=IcuQueueMode::Macro;
        image.storage=ProgramStorage::Imem;
        image.physical_word_bytes=
            descriptor.format==ProgramLoadFormat::Word96?12:16;
        return image;
    }

    static void validate_config_descriptor(
        const ProgramLoadConfigDescriptor& descriptor,
        const ProgramImage& submitted)
    {
        const auto image=program_image_from_config(descriptor);
        if(image.source_address!=submitted.source_address
            || image.kind!=submitted.kind
            || image.queue_index!=submitted.queue_index
            || image.byte_count!=submitted.byte_count
            || image.word_bytes()!=submitted.word_bytes()
            || descriptor.destination!=program_destination(submitted)
            || descriptor.expected_word_count!=submitted.word_count())
            throw std::logic_error(
                "Config FIFO descriptor differs from submitted ProgramImage");
    }

    void initialize_canonical_statistics()
    {
        stats_={};
        stats_.request_accepted=pending_submission_cycle_;
        for(const auto& image:pending_images_) {
            stats_.images.push_back(ProgramImageStatistics{image});
            stats_.images.back().effective_bytes=image.byte_count;
            stats_.effective_bytes+=image.byte_count;
            stats_.stores[static_cast<std::size_t>(image.storage)]
                .effective_bytes+=image.byte_count;
        }
    }

    void advance_reassembly_and_route(const ProgramImage& image,
        ProgramImageStatistics& record,std::size_t cycle,
        bool rtl_distribution)
    {
        if (reassembly_.size() >= image.word_bytes()) {
            const bool destination_ready = rtl_distribution
                ? word_fifo_.size() < hw::config::kProgramLoadWordFifoDepth
                : cycle >= next_write_cycle_
                    && commits_.size()
                        <= config_.imem_write_commit_latency_cycles;
            if (destination_ready) {
                PendingWrite write{};
                write.program_imem_address = active_config_->imem_base_address
                    + accepted_words_++;
                for (std::size_t b = 0; b < image.word_bytes(); ++b) {
                    write.bytes[b] = reassembly_.front();
                    reassembly_.pop_front();
                }
                if (rtl_distribution) {
                    word_fifo_.push_back(write);
                    ++stats_.word_fifo_writes;
                } else {
                    write.due = cycle
                        + config_.imem_write_commit_latency_cycles;
                    commits_.push_back(write);
                    if (!record.first_write_accepted)
                        record.first_write_accepted = cycle;
                    record.last_write_accepted = cycle;
                    next_write_cycle_ = cycle
                        + config_.imem_write_ii_cycles;
                }
            } else ++stats_.target_wait_cycles;
        }

        if (route_.back() && reassembly_.size() + route_.back()->size
            <= kReassemblyBytes) {
            const auto& packet = *route_.back();
            for (std::size_t b = 0; b < packet.size; ++b)
                reassembly_.push_back(packet.bytes[b]);
            route_.back().reset();
        }
        for (std::size_t s = route_.size() - 1; s > 0; --s)
            if (!route_[s] && route_[s - 1]) {
                route_[s] = std::move(route_[s - 1]);
                route_[s - 1].reset();
            }
        if (dma_.word_ready() && routed_bytes_ < image.byte_count
            && !route_.front()) {
            Packet packet{};
            packet.size = std::min({config_.bus_bytes_per_cycle,
                image.word_bytes() - vector_offset_,
                image.byte_count - routed_bytes_});
            const auto word = dma_.front_word();
            for (std::size_t b = 0; b < packet.size; ++b)
                packet.bytes[b] = word.bytes[vector_offset_ + b];
            vector_offset_ += packet.size;
            routed_bytes_ += packet.size;
            if (vector_offset_ == image.word_bytes()) {
                dma_.pop_word();
                vector_offset_ = 0;
            }
            route_.front() = packet;
            if (routed_bytes_ < image.byte_count && vector_offset_ != 0)
                ++stats_.bus_wait_cycles;
        }
    }
    struct Packet { std::array<std::uint8_t, hw::kPhysicalVectorBytes> bytes{}; std::size_t size{0}; };
    struct PendingWrite {
        std::array<std::uint8_t, kReassemblyBytes> bytes{};
        std::size_t program_imem_address{0};
        std::size_t due{0};
    };

    ProgramLoadWordInterfaceRecord word_interface_record(
        const PendingWrite& write,const ProgramImage& image) const
    {
        ProgramLoadWordInterfaceRecord record{};
        std::copy_n(write.bytes.begin(),image.word_bytes(),record.data.begin());
        record.strb=static_cast<std::uint8_t>(
            (std::uint32_t{1}<<((image.word_bytes()+3)/4))-1);
        if(active_config_) {
            record.scope=active_config_->scope;
            record.destination=active_config_->destination;
        }
        record.imem_address=write.program_imem_address;
        record.image_last=write.program_imem_address+1
            ==(active_config_?active_config_->imem_base_address:0)
                +image.word_count();
        return record;
    }

    void advance_canonical_reassembler_word(const ProgramImage& image)
    {
        auto& view=program_load_ports_.r2;
        view.reassembler_word_valid_o=dma_.word_ready();
        view.reassembler_word_ready_i=word_fifo_.size()
            <hw::config::kProgramLoadWordFifoDepth;
        view.reassembler_word_fire=view.reassembler_word_valid_o
            &&view.reassembler_word_ready_i;
        view.word_fifo_enqueue_valid_i=view.reassembler_word_valid_o;
        view.word_fifo_enqueue_ready_o=view.reassembler_word_ready_i;
        view.word_fifo_enqueue_fire=view.reassembler_word_fire;
        if(!view.reassembler_word_valid_o) return;

        const auto word=dma_.front_word();
        PendingWrite write{};
        write.program_imem_address=(active_config_
            ?active_config_->imem_base_address:0)+accepted_words_;
        std::copy_n(word.bytes.begin(),image.word_bytes(),write.bytes.begin());
        const auto record=word_interface_record(write,image);
        view.reassembler_word=record;
        view.word_fifo_enqueue=record;
        view.reassembler_image_complete=view.reassembler_word_fire
            &&record.image_last;
        if(!view.reassembler_word_fire) {
            ++stats_.target_wait_cycles;
            return;
        }
        word_fifo_.push_back(std::move(write));
        dma_.pop_word();
        ++accepted_words_;
        routed_bytes_+=word.byte_count;
        ++stats_.word_fifo_writes;
    }
    struct Snapshot {
        State state{State::Idle};
        ProgramLoadStatistics stats{};
        std::vector<ProgramImage> images{};
        std::vector<std::optional<Packet>> route{};
        std::deque<std::uint8_t> reassembly{};
        std::deque<PendingWrite> commits{};
        std::deque<PendingWrite> word_fifo{};
        std::deque<ProgramDistributionRecord> prime_fifo{};
        std::size_t current{0};
        std::size_t routed_bytes{0};
        std::size_t vector_offset{0};
        std::size_t accepted_words{0};
        std::size_t next_write_cycle{0};
        std::uint64_t last_generation{0};
        bool start_requested{false};
        bool dependencies_ready{false};
        MemProgramPortSignals mem_program_ports{};
        std::vector<ProgramImage> pending_images{};
        std::optional<std::uint64_t> pending_generation{};
        std::size_t pending_submission_cycle{0};
        bool canonical_submission_active{false};
        bool load_request_active{false};
        bool descriptor_fault{false};
        bool instruction_load_done{false};
        ProgramLoadHardwareInterfaceView program_load_ports{};
        std::optional<ProgramLoadConfigDescriptor> active_config{};
    };

    Snapshot capture_snapshot() const
    {
        return Snapshot {state_,stats_,images_,route_,reassembly_,commits_,
            word_fifo_,prime_fifo_,
            current_,routed_bytes_,vector_offset_,accepted_words_,
            next_write_cycle_,last_generation_,start_requested_,
            dependencies_ready_,mem_program_ports_,pending_images_,
            pending_generation_,pending_submission_cycle_,
            canonical_submission_active_,load_request_active_,
            descriptor_fault_,instruction_load_done_,
            program_load_ports_,active_config_};
    }
    void restore_snapshot(Snapshot snapshot)
    {
        state_=snapshot.state;
        stats_=std::move(snapshot.stats);
        images_=std::move(snapshot.images);
        route_=std::move(snapshot.route);
        reassembly_=std::move(snapshot.reassembly);
        commits_=std::move(snapshot.commits);
        word_fifo_=std::move(snapshot.word_fifo);
        prime_fifo_=std::move(snapshot.prime_fifo);
        current_=snapshot.current;
        routed_bytes_=snapshot.routed_bytes;
        vector_offset_=snapshot.vector_offset;
        accepted_words_=snapshot.accepted_words;
        next_write_cycle_=snapshot.next_write_cycle;
        last_generation_=snapshot.last_generation;
        start_requested_=snapshot.start_requested;
        dependencies_ready_=snapshot.dependencies_ready;
        mem_program_ports_=std::move(snapshot.mem_program_ports);
        pending_images_=std::move(snapshot.pending_images);
        pending_generation_=snapshot.pending_generation;
        pending_submission_cycle_=snapshot.pending_submission_cycle;
        canonical_submission_active_=snapshot.canonical_submission_active;
        load_request_active_=snapshot.load_request_active;
        descriptor_fault_=snapshot.descriptor_fault;
        instruction_load_done_=snapshot.instruction_load_done;
        program_load_ports_=std::move(snapshot.program_load_ports);
        active_config_=std::move(snapshot.active_config);
    }

    void validate_image(const ProgramImage& image) {
        icu_.validate_program_image(image);
        if(image.byte_count) {
            const auto requests=(image.byte_count+hw::kPhysicalVectorBytes-1)
                / hw::kPhysicalVectorBytes;
            if(requests-1>(UINT64_MAX-image.source_address)
                    / hw::kPhysicalVectorBytes
                || image.source_address+(requests-1)*hw::kPhysicalVectorBytes
                    > UINT64_MAX-(hw::kPhysicalVectorBytes-1))
                throw std::invalid_argument("program DMA image crosses the address space");
        }
    }

    static bool uses_rtl_distribution(const ProgramImage& image) noexcept
    {
        return image.storage==ProgramStorage::Imem
            && image.mode==IcuQueueMode::Macro
            && (image.kind==IcuMacroQueueKind::Mem
                || image.kind==IcuMacroQueueKind::MxmLoad
                || image.kind==IcuMacroQueueKind::MxmDequant
                || image.kind==IcuMacroQueueKind::MxmCompute
                || image.kind==IcuMacroQueueKind::MxmParamLoad
                || image.kind==IcuMacroQueueKind::MxmBackend);
    }

    static std::uint16_t program_destination(const ProgramImage& image)
    {
        std::size_t queues_per_hemisphere=0;
        std::uint16_t kind=0;
        switch(image.kind) {
        case IcuMacroQueueKind::Mem:
            queues_per_hemisphere=InstructionControlUnit::kMemQueuesPerHemisphere;
            kind=0;break;
        case IcuMacroQueueKind::MxmLoad:
            queues_per_hemisphere=InstructionControlUnit::kMxmQueuesPerHemisphere;
            kind=1;break;
        case IcuMacroQueueKind::MxmDequant:
            queues_per_hemisphere=InstructionControlUnit::kMxmQueuesPerHemisphere;
            kind=2;break;
        case IcuMacroQueueKind::MxmCompute:
            queues_per_hemisphere=InstructionControlUnit::kMxmQueuesPerHemisphere;
            kind=3;break;
        case IcuMacroQueueKind::MxmParamLoad:
            queues_per_hemisphere=InstructionControlUnit::kMxmQueuesPerHemisphere;
            kind=7;break;
        case IcuMacroQueueKind::MxmBackend:
            queues_per_hemisphere=InstructionControlUnit::kMxmQueuesPerHemisphere;
            kind=8;break;
        default:
            throw std::invalid_argument(
                "Program destination is not connected by RTL Program Load");
        }
        const auto hemisphere=image.queue_index/queues_per_hemisphere;
        const auto local=image.queue_index%queues_per_hemisphere;
        if(hemisphere>=hw::kHemispheres || local>0xff)
            throw std::invalid_argument("Program destination cannot be packed");
        return static_cast<std::uint16_t>((hemisphere<<12)|(kind<<8)|local);
    }

    static std::pair<std::size_t,std::size_t> relay_destination(
        const ProgramImage& image)
    {
        constexpr auto relay_count=
            hw::config::kProgramLoadRelaysPerHemisphere;
        if (image.kind==IcuMacroQueueKind::Mem) {
            constexpr auto queues_per_hemisphere=
                InstructionControlUnit::kMemQueuesPerHemisphere;
            constexpr auto banks_per_relay=
                hw::kProgramLoadMemBanksPerRelay;
            const auto hemisphere=image.queue_index/queues_per_hemisphere;
            const auto local=image.queue_index%queues_per_hemisphere;
            return {hemisphere,std::min(local/banks_per_relay,
                relay_count-1)};
        }
        constexpr auto queues_per_hemisphere=
            InstructionControlUnit::kMxmQueuesPerHemisphere;
        return {image.queue_index/queues_per_hemisphere,relay_count-1};
    }

    ProgramDistributionRecord distribution_record(
        const PendingWrite& write,const ProgramImage& image) const
    {
        ProgramDistributionRecord record{};
        std::copy_n(write.bytes.begin(),image.word_bytes(),record.data.begin());
        record.strb=static_cast<std::uint8_t>(
            (std::uint32_t{1}<<((image.word_bytes()+3)/4))-1);
        if(active_config_) {
            record.scope=active_config_->scope;
            record.destination=active_config_->destination;
        }
        record.imem_address=write.program_imem_address;
        record.image_last=write.program_imem_address+1
            ==(active_config_?active_config_->imem_base_address:0)
                +image.word_count();
        return record;
    }

    static PendingWrite pending_write(
        const ProgramDistributionRecord& record,std::size_t due)
    {
        PendingWrite write{};
        std::copy(record.data.begin(),record.data.end(),write.bytes.begin());
        write.program_imem_address=record.imem_address;
        write.due=due;
        return write;
    }

    void advance_rtl_distribution(const ProgramImage& image,
        ProgramImageStatistics& record,std::size_t cycle,bool canonical=false)
    {
        if(distribution_.prepared())
            throw std::logic_error(
                "Program Distribution LOW executed twice in one cycle");
        stats_.peak_word_fifo=std::max(
            stats_.peak_word_fifo,word_fifo_.size());
        stats_.word_fifo_full_cycles+=word_fifo_.size()
            == hw::config::kProgramLoadWordFifoDepth;

        const bool endpointReady=cycle>=next_write_cycle_
            &&commits_.size()<=config_.imem_write_commit_latency_cycles;
        ProgramDistributionInputs inputs{};
        inputs.load_record_valid=!word_fifo_.empty();
        if(inputs.load_record_valid)
            inputs.load_record=distribution_record(word_fifo_.front(),image);
        for(auto& hemisphere:inputs.mem_endpoint_ready)
            hemisphere.fill(endpointReady);
        inputs.outer_endpoint_ready.fill(endpointReady);
        inputs.vxm_endpoint_ready=endpointReady;
        if(canonical)
            inputs.endpoint_ready=[this,endpointReady](const auto& payload) {
                return endpoint_ready(payload,endpointReady);
            };
        distribution_.drive(inputs);
        distribution_.low();
        const auto& view=distribution_.hardware_interface_view();
        program_load_ports_.r3.distribution=view;

        auto& r2=program_load_ports_.r2;
        r2.word_fifo_level_o=word_fifo_.size();
        r2.word_fifo_dequeue_valid_o=view.load_record_input.valid;
        r2.word_fifo_dequeue_ready_i=view.load_record_input.ready;
        r2.word_fifo_dequeue_fire=view.load_record_input.fire;
        if(r2.word_fifo_dequeue_valid_o)
            r2.word_fifo_dequeue=word_interface_record(
                word_fifo_.front(),image);
        if (r2.word_fifo_dequeue_fire) {
            word_fifo_.pop_front();
            ++stats_.word_fifo_reads;
            ++stats_.ingress_fifo_writes;
        }

        stats_.peak_ingress_fifo=std::max(
            stats_.peak_ingress_fifo,view.elastic_level);
        stats_.ingress_fifo_full_cycles+=view.elastic_level
            ==ProgramDistributionTransport::kElasticDepth;
        stats_.ingress_fifo_reads+=view.elastic_output.fire;
        for(std::size_t h=0;h<ProgramDistributionTransport::kHemispheres;++h)
            for(std::size_t relay=0;
                relay<ProgramDistributionTransport::kRelays;++relay) {
                stats_.peak_relay_level=std::max(stats_.peak_relay_level,
                    view.relay_level[h][relay]);
                stats_.relay_full_cycles+=view.relay_level[h][relay]
                    ==ProgramDistributionTransport::kRelayDepth;
                stats_.relay_writes+=view.relay_input[h][relay].fire;
                stats_.relay_reads+=view.relay_forward_output[h][relay].fire
                    ||view.mem_endpoint_input[h][relay].fire;
            }

        std::optional<ProgramDistributionRecord> completed{};
        ProgramDistributionChannelView* completed_port=nullptr;
        for(std::size_t h=0;h<ProgramDistributionTransport::kHemispheres;++h)
            for(std::size_t relay=0;
                relay<ProgramDistributionTransport::kRelays;++relay) {
                const auto& endpoint=view.mem_endpoint_input[h][relay];
                if(image.kind==IcuMacroQueueKind::Mem&&endpoint.valid) {
                    mem_program_ports_.mem_program_valid_o=endpoint.valid;
                    mem_program_ports_.mem_program_ready_i=endpoint.ready;
                    mem_program_ports_.mem_program_fire=endpoint.fire;
                    mem_program_ports_.mem_program_imem_address_o=
                        endpoint.payload.imem_address;
                    mem_program_ports_.mem_program_data_o=endpoint.payload.data;
                    mem_program_ports_.mem_program_strb_o=endpoint.payload.strb;
                    mem_program_ports_.mem_program_last_o=
                        endpoint.payload.image_last;
                }
                if(endpoint.fire) {
                    if(completed) throw std::logic_error(
                        "multiple Program Endpoints completed one serialized record");
                    completed=endpoint.payload;
                    completed_port=&program_load_ports_.r3.distribution.mem_endpoint_input[h][relay];
                }
            }
        for(std::size_t h=0;h<view.outer_endpoint_input.size();++h) {
            const auto& endpoint=view.outer_endpoint_input[h];
            if(endpoint.fire) {
                if(completed) throw std::logic_error(
                    "multiple Program Endpoints completed one serialized record");
                completed=endpoint.payload;
                completed_port=&program_load_ports_.r3.distribution.outer_endpoint_input[h];
            }
        }
        if(view.vxm_endpoint_input.fire) {
            if(completed) throw std::logic_error(
                "multiple Program Endpoints completed one serialized record");
            completed=view.vxm_endpoint_input.payload;
            completed_port=&program_load_ports_.r3.distribution.vxm_endpoint_input;
        }
        if(completed) {
            const auto write=pending_write(*completed,
                cycle+config_.imem_write_commit_latency_cycles);
            if(canonical) {
                const auto expected_strb=(1U<<(image.word_bytes()/4))-1;
                if(completed->control_prime || completed->strb!=expected_strb
                    || completed->destination!=program_destination(image)
                    || completed->imem_address>=image.word_count()) {
                    completed_port->error=true;
                    descriptor_fault_=true;
                    return;
                }
                commit_word(image,write);
            }
            commits_.push_back(write);
            if(!record.first_write_accepted)
                record.first_write_accepted=cycle;
            record.last_write_accepted=cycle;
            next_write_cycle_=cycle+config_.imem_write_ii_cycles;
        }
    }

    void begin_queue()
    {
        routed_bytes_ = vector_offset_ = accepted_words_ = next_write_cycle_ = 0;
        const auto& image = images_[current_];
        if(image.word_count()) {
            const auto width=image.word_bytes()==4
                ? PhysicalInstructionWordWidth::Bits32
                : image.word_bytes()==8
                    ? PhysicalInstructionWordWidth::Bits64
                    : image.word_bytes()==12
                        ? PhysicalInstructionWordWidth::Bits96
                        : PhysicalInstructionWordWidth::Bits128;
            dma_.start(InstructionFetchProgram {
                image.source_address,image.word_count(),width});
        }
        else stats_.images[current_].source_done=stats_.request_accepted;
        state_ = State::LoadQueue;
    }

    void commit_word(const ProgramImage& image,const PendingWrite& write) {
        icu_.stage_program_word(image,write.program_imem_address,
            write.bytes.data());
    }
    void configure_target(const ProgramImage& image) {icu_.stage_program_seal(image);}

    InstructionFetchDma dma_;
    Ddr4Model& ddr_;
    std::function<bool()> drain_checker_{};
    std::function<void()> program_reset_{};
    InstructionControlUnit& icu_;
    ProgramLoadConfig config_;
    State state_{State::Idle};
    ProgramLoadStatistics stats_{};
    std::vector<ProgramImage> images_{};
    std::vector<std::optional<Packet>> route_{};
    std::deque<std::uint8_t> reassembly_{};
    std::deque<PendingWrite> commits_{};
    std::deque<PendingWrite> word_fifo_{};
    std::deque<ProgramDistributionRecord> prime_fifo_{};
    ProgramDistributionTransport distribution_{};
    std::size_t current_{0}, routed_bytes_{0}, vector_offset_{0}, accepted_words_{0}, next_write_cycle_{0};
    std::uint64_t last_generation_{0};
    bool start_requested_{false}, dependencies_ready_{false};
    MemProgramPortSignals mem_program_ports_{};
    std::optional<Snapshot> next_snapshot_{};
    bool clock_managed_{false};
    bool low_started_{false}, low_complete_{false};
    bool low_observe_dma_{false}, observe_dma_low_{false};
    ProgramLoadCommandController command_controller_{};
    ProgramLoadConfigFifo config_fifo_{};
    ProgramLoadHostAdapter host_adapter_{};
    std::vector<ProgramImage> pending_images_{};
    std::optional<std::uint64_t> pending_generation_{};
    std::size_t pending_submission_cycle_{0};
    bool canonical_submission_active_{false};
    bool load_request_active_{false};
    bool descriptor_fault_{false};
    bool instruction_load_done_{false};
    ProgramLoadHardwareInterfaceView program_load_ports_{};
    std::optional<ProgramLoadConfigDescriptor> active_config_{};
    std::optional<ProgramLoadConfigDescriptor> low_config_enqueue_{};
    bool low_config_reset_{false};
    State low_state_for_view_{State::Idle};
    bool low_distribution_idle_{true};
    bool low_load_request_active_{false};
    bool low_descriptor_fault_{false};
    bool low_instruction_load_done_{false};
    bool low_program_reset_{false};
};

} // namespace ftlpu
