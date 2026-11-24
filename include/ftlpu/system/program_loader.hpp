#pragma once

#include "ftlpu/icu/instruction_fetch_dma.hpp"
#include "ftlpu/system/icu.hpp"

#include "ftlpu/system/mem_program_endpoint.hpp"

#include <limits>
#include <functional>
#include <ostream>

namespace ftlpu {

struct ProgramLoadConfig {
    InstructionFetchDmaConfig instruction_dma{};
    std::size_t bus_bytes_per_cycle{hw::config::kProgramLoadBusBytesPerCycle};
    std::size_t route_latency_cycles{hw::config::kProgramLoadRouteLatencyCycles};
    std::size_t imem_write_ii_cycles{hw::config::kProgramLoadImemWriteIiCycles};
    std::size_t imem_write_commit_latency_cycles{
        hw::config::kProgramLoadImemWriteCommitLatencyCycles};

    void validate() const
    {
        instruction_dma.validate();
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
    std::optional<std::size_t> first_write_accepted{}, first_write_committed{}, last_write_accepted{}, prime_accepted{};
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
    std::size_t peak_reassembly_bytes{0};
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

// One active program, one active destination. DDR is ticked by the enclosing
// system, exactly once per physical cycle. This engine shares that DDR with
// weight/data DMA. Program bytes take the dedicated bounded read/distribution path; data MEM is never a staging store.
class ProgramLoader : public timing::ClockedModule {
public:
    enum class State { Idle, LoadQueue, WaitCommit, AllCommitted, Prime, Ready, Started, Draining, Completed, Failed };
    static constexpr std::size_t kReassemblyBytes = 64;

    ProgramLoader(Ddr4Model& ddr, InstructionControlUnit& icu,
        ProgramLoadConfig config = {})
        : dma_(ddr, config.instruction_dma), ddr_(ddr), icu_(icu), config_(config)
    { config_.validate(); }

    void enable_clocked_mode() {
        if(clock_managed_ || next_snapshot_)throw std::logic_error("loader clock attachment");
        dma_.enable_clocked_mode();clock_managed_=true;
    }
    bool clocked_mode() const noexcept {return clock_managed_;}
    void begin_clocked_low(std::size_t cycle) {
        if(!clock_managed_ || next_snapshot_)throw std::logic_error("loader LOW twice");
        auto current=capture_snapshot();tick_before_dma(cycle);
        next_snapshot_=capture_snapshot();restore_snapshot(std::move(current));
    }
    void advance_clocked_dma_low(std::size_t cycle) {
        if(!next_snapshot_ || low_complete_)throw std::logic_error("loader DMA phase");
        dma_.low();
        auto current=capture_snapshot();restore_snapshot(std::move(*next_snapshot_));
        observe_dma_after_low(cycle);next_snapshot_=capture_snapshot();
        restore_snapshot(std::move(current));low_complete_=true;
    }
    void low() override {throw std::logic_error("loader needs physical cycle input");}
    void validate_posedge() const override {
        if(!next_snapshot_ || !low_complete_)throw std::logic_error("loader edge before LOW");
    }
    void posedge() override {
        validate_posedge();restore_snapshot(std::move(*next_snapshot_));
        next_snapshot_.reset();low_complete_=false;
    }
    void high() override {}
    static std::pair<std::size_t,std::size_t> relay_destination_for_test(const ProgramImage& im) {return relay_destination(im);}

    void configure(ProgramLoadConfig config)
    {
        if (busy()) throw std::logic_error("program loader is busy");
        config.validate();
        dma_.configure(config.instruction_dma);
        config_ = config;
    }
    const ProgramLoadConfig& config() const noexcept { return config_; }
    void set_program_reset(std::function<void()> reset) {program_reset_=std::move(reset);}
    State state() const noexcept { return state_; }
    const MemProgramPortSignals& mem_program_port_signals() const noexcept { return mem_program_ports_; }
    bool failed() const noexcept {return state_==State::Failed;}
    void recover_failed_program() {
        if(!failed() || !dma_.drained())throw std::logic_error("program error not drained");
        dma_.recover();
        icu_.reset_program(); // abort unexecuted/partial slot before the drained task reset
        if(program_reset_)program_reset_();
        // Keep the generation watermark and physical clocks; retry is a new
        // task. Never resume a partially overwritten private instruction slot.
        mem_endpoint_={}; mem_program_ports_={};
        stats_={};images_.clear();state_=State::Idle;segment_loading_=false;
        start_requested_=dependencies_ready_=false;
    }
    void begin_task_abort() {
        last_generation_=last_generation();dma_.begin_abort();state_=State::Failed;
        stats_.kernel_done_cycle.reset();images_.clear();route_.clear();
        reassembly_.clear();commits_.clear();clear_distribution();start_requested_=dependencies_ready_=false;
        icu_.stage_program_issue_enabled(false);icu_.stage_program_loading(true);
    }
    bool busy() const noexcept
    { return state_ == State::LoadQueue || state_ == State::WaitCommit
        || state_ == State::AllCommitted || state_ == State::Prime || state_ == State::Ready; }
    bool all_images_committed() const noexcept
    { return stats_.all_images_committed.has_value(); }
    bool ready_to_start() const noexcept { return state_ == State::Ready; }
    bool started() const noexcept { return stats_.program_start.has_value(); }
    bool kernel_done() const noexcept {return stats_.kernel_done_cycle.has_value();}
    void set_drain_checker(std::function<bool()> checker) {drain_checker_=std::move(checker);}
    bool can_load_program() const {
        return !failed() && !busy() && (!started() || kernel_done()) && dma_.drained()
            && commits_.empty() && reassembly_.empty()
            && std::none_of(route_.begin(),route_.end(),[](const auto& p){return p.has_value();})
            && icu_.program_queues_done() && (!drain_checker_ || drain_checker_());
    }
    void observe_completion(std::size_t edge) {
        if(failed() || !started() || kernel_done()) return;
        if(icu_.program_queues_done()) state_=State::Draining;
        if(state_==State::Draining && !more_segments_pending_ && dma_.drained()
            && (!drain_checker_ || drain_checker_())) {
            stats_.kernel_done_cycle=edge;state_=State::Completed;
        }
    }
    std::uint64_t last_generation() const noexcept {return std::max(last_generation_,icu_.program_generation());}
    void begin_direct_program(std::size_t cycle,std::uint64_t generation=0) {
        more_segments_pending_=false;
        last_generation_=std::max(last_generation_,generation);
        stats_={};stats_.program_start=cycle;state_=State::Started;
        images_.clear();route_.clear();reassembly_.clear();commits_.clear();clear_distribution();
        icu_.stage_program_loading(false);icu_.stage_program_issue_enabled(true);
    }
    std::size_t total_committed_bytes() const noexcept {return total_committed_bytes_;}
    const ProgramLoadStatistics& statistics() const noexcept { return stats_; }
    InstructionFetchDma& dma() noexcept {return dma_;}
    const InstructionFetchDma& dma() const noexcept { return dma_; }

    void reset()
    {
        mem_endpoint_={}; mem_program_ports_={};
        dma_.reset();
        state_ = State::Idle;
        stats_ = {};
        last_generation_=0;
        images_.clear(); route_.clear(); reassembly_.clear(); commits_.clear();clear_distribution();
        start_requested_ = dependencies_ready_ = false;
        segment_loading_=more_segments_pending_=false;
    }

    // Validation precedes any mutation, including clearing the old program.
    void submit(std::vector<ProgramImage> images, std::size_t physical_cycle,
                bool compute_segment=false, bool more_segments=false)
    {
        if(ddr_.axi().isolated())throw AxiResponseProtocolError("program submission on isolated DDR link");
        if (!compute_segment && !can_load_program()) throw std::logic_error("previous program has not completed and drained");
        if(compute_segment) {
            // A continuation belongs to a previously started, non-failed task.
            // Retaining its generation watermark across recovery is for stale
            // event rejection, not permission to patch its retired iMEM slot.
            if(failed() || !started() || !more_segments_pending_)
                throw std::logic_error("compute segment has no resumable program");
            if(busy() || !dma_.drained() || !commits_.empty() || !reassembly_.empty())
                throw std::logic_error("previous segment source not retired");
            for(const auto& im:images) {
                // Continuations may rebind an empty QCTL partition's local end, never
                // replace event records or an active queue in this generation.
                const bool extent = im.storage==ProgramStorage::Qctl && !im.byte_count;
                if((im.storage!=ProgramStorage::Imem && !extent) || im.generation!=last_generation_
                    || im.kind==IcuMacroQueueKind::C2cDma || im.kind==IcuMacroQueueKind::C2cTx
                    || im.kind==IcuMacroQueueKind::C2cRx)
                    throw std::invalid_argument("compute segment target/generation");
                each_segment_target(im,[](auto& q){if(!q.done())throw std::logic_error("active single-slot overwrite");});
            }
        }
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
        if(generation && (compute_segment?*generation!=last_generation():*generation<=last_generation()))
            throw std::invalid_argument("program generation must increase across task switches");
        if((qctl && !queueEvents && !compute_segment) || (barr && (!barrierEvents || !barrierRam)))
            throw std::invalid_argument("missing committed event or barrier configuration");
        if (!dma_.drained())
            throw std::logic_error("program DMA still has unconsumed traffic");
        std::size_t total_bytes = 0;
        for (const auto& image : images) {
            if (total_bytes > std::numeric_limits<std::size_t>::max() - image.byte_count)
                throw std::overflow_error("aggregate program image size overflow");
            total_bytes += image.byte_count;
        }
        segment_loading_=compute_segment;
        // A quiescent non-final segment is a safe reload point, not KERNEL_DONE.
        more_segments_pending_=more_segments;
        if(compute_segment) {for(const auto& im:images) if(im.storage==ProgramStorage::Imem) icu_.stage_segment_begin(im);}
        else if(program_reset_) program_reset_(); else icu_.stage_program_reset();
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
        commits_.clear();clear_distribution();
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

    void tick(std::size_t cycle) {
        if(clock_managed_)throw std::logic_error("clock-managed loader requires SystemPhase");
        tick_before_dma(cycle);dma_.tick();observe_dma_after_low(cycle);
    }
private:
    void tick_before_dma(std::size_t cycle)
    {
        mem_program_ports_={};
        observe_completion(cycle+1);
        if(failed()) {return;}
        if (!busy()) return;
        icu_.stage_program_issue_enabled(false);
        icu_.stage_program_loading(!segment_loading_);
        if (state_ == State::Ready) {
            // Control images and IQ are committed. Permit their transport
            // programs to satisfy initial MEM fences while compute waits for
            // START; otherwise page-zero readiness and START would deadlock.
            icu_.stage_program_loading(false);
            if (start_requested_ && dependencies_ready_) {
                stats_.program_start = cycle;
                state_ = State::Started;
                icu_.stage_program_loading(false);
                icu_.stage_program_issue_enabled(true);
            }
            return;
        }
        if (state_ == State::AllCommitted) { state_ = State::Prime; }
        if (state_ == State::Prime) {
            bool primed=true;
            if(segment_loading_) {
                for(const auto& im:images_) {if(im.storage!=ProgramStorage::Imem)continue;icu_.stage_segment_prime(im);each_segment_target(im,[&](auto& q){primed &= q.imem_occupancy()==0 || q.frontend_ready();});}
            } else {icu_.stage_program_frontend_tick();primed=icu_.program_frontends_ready();}
            if (primed) {
                stats_.frontend_ready = cycle;
                state_ = State::Ready;
            }
            return;
        }

        auto& record = stats_.images[current_];
        const auto& image = images_[current_];

        if(dma_.fault()) {
            state_=State::Failed;route_.clear();commits_.clear();clear_distribution();reassembly_.clear();
            icu_.stage_program_issue_enabled(false);icu_.stage_program_loading(true);return;
        }
        // Commit is independent of accepting the next word: latency is a
        // pipeline, never a per-word round trip.
        while (!commits_.empty() && commits_.front().due <= cycle) {
            const auto w=commits_.front();commits_.pop_front();
            if(w.prime) {
                if(record.written_words!=image.word_count())throw std::logic_error("PRIME overtook WRITE commits");
                configure_target(image);record.image_committed=cycle;
                if(segment_loading_ && image.storage==ProgramStorage::Imem)icu_.stage_segment_end(image);
            } else {
                if(is_mem_macro(image)) {
                    mem_program_ports_.mem_program_queue_o=image.queue_index;mem_program_ports_.mem_program_generation_o=last_generation_;
                    if(!mem_endpoint_.commit(w.address)){fail_mem_endpoint();return;}
                    mem_program_ports_.mem_program_commit_i=true;mem_program_ports_.mem_program_commit_address_i=w.address;
                }
                if(!record.first_write_committed)record.first_write_committed=cycle;
                commit_word(image,w);total_committed_bytes_+=image.word_bytes();++record.written_words;++stats_.written_words;
                ++stats_.stores[static_cast<std::size_t>(image.storage)].written_words;
                stats_.stores[static_cast<std::size_t>(image.storage)].last_commit=cycle;
            }
        }
        if(record.image_committed) {
            if(!record.source_done) record.source_done=cycle;
                        if (++current_ == images_.size()) {
                stats_.all_images_committed = cycle;
                state_ = State::AllCommitted;
            } else begin_queue();
            return; // queue n+1 cannot issue before queue n commits
        }

        const bool rtl_distribution=uses_rtl_distribution(image);
        if (rtl_distribution) {
            advance_rtl_distribution(image,record,cycle);
            if(failed())return;
        }

        if(accepted_words_==image.word_count() && !prime_sent_) {
            PendingWrite prime{};prime.prime=true;
            if(uses_rtl_distribution(image)) {
                if(word_fifo_.size()<hw::config::kProgramLoadWordFifoDepth) {
                    word_fifo_.push_back(prime);++stats_.word_fifo_writes;prime_sent_=true;
                }
            } else if(commits_.size()<=config_.imem_write_commit_latency_cycles) {
                prime.due=cycle+config_.imem_write_commit_latency_cycles;
                commits_.push_back(prime);prime_sent_=true;
            }
        }

        if (reassembly_.size() >= image.word_bytes()) {
            const bool destination_ready = rtl_distribution
                ? word_fifo_.size() < hw::config::kProgramLoadWordFifoDepth
                : cycle >= next_write_cycle_
                    && commits_.size()
                        <= config_.imem_write_commit_latency_cycles;
            if (destination_ready) {
                PendingWrite write{};
                write.address = accepted_words_++;
                write.last = accepted_words_==image.word_count();
                write.strb = (1u<<(image.word_bytes()/4))-1;
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

    }
    // Observe the prepared image after DMA LOW, INCLUDING begin_queue() on
    // a preceding image's commit cycle (c223ed7 transition-accounting fix).
    void observe_dma_after_low(std::size_t cycle) {
        if(current_>=images_.size() || failed())return;
        auto& record=stats_.images[current_];const auto& image=images_[current_];
        if(dma_.low_last_read_request()) {
            if(!stats_.first_ddr_request)stats_.first_ddr_request=cycle;
            if(!record.first_ddr_request)record.first_ddr_request=cycle;
            const auto bytes=dma_.low_read_request_bytes();
            record.dma_bytes+=bytes;stats_.dma_bytes+=bytes;
            stats_.stores[static_cast<std::size_t>(image.storage)].dma_bytes+=bytes;
        }
        if(dma_.low_last_beat()) {
            if(!record.first_data_return)record.first_data_return=cycle;
            if(!stats_.first_data_return)stats_.first_data_return=cycle;
        }
        if(dma_.low_load_credit_wait())++stats_.dma_credit_wait_cycles;
        if(dma_.fault()) {
            state_=State::Failed;route_.clear();commits_.clear();clear_distribution();reassembly_.clear();
            icu_.stage_program_issue_enabled(false);icu_.stage_program_loading(true);return;
        }
        if(dma_.low_idle() && !record.source_done) {record.source_done=cycle;state_=State::WaitCommit;}
        stats_.peak_reassembly_bytes=std::max(stats_.peak_reassembly_bytes,reassembly_.size());
        stats_.peak_instruction_dma_outstanding=std::max(stats_.peak_instruction_dma_outstanding,dma_.low_outstanding_reads());
        stats_.peak_instruction_dma_response_fifo=std::max(stats_.peak_instruction_dma_response_fifo,dma_.low_receive_queue_size());
    }

private:
    template<class Fn> void each_segment_target(const ProgramImage& im,Fn fn) {
        icu_.visit_program_queue(im.kind,im.queue_index,false,fn);
    }
    struct Packet { std::array<std::uint8_t, hw::kPhysicalVectorBytes> bytes{}; std::size_t size{0}; };
    struct PendingWrite { std::array<std::uint8_t, kReassemblyBytes> bytes{}; std::size_t address{0}, due{0}; bool prime{}, last{}; unsigned strb{}; };

    struct Snapshot {
        State state{State::Idle};
        ProgramLoadStatistics stats{};
        std::vector<ProgramImage> images{};
        std::vector<std::optional<Packet>> route{};
        std::deque<std::uint8_t> reassembly{};
        std::deque<PendingWrite> commits{};
        std::deque<PendingWrite> word_fifo{};
        std::deque<PendingWrite> ingress_fifo{};
        std::array<std::array<std::deque<PendingWrite>,
            hw::config::kProgramLoadRelaysPerHemisphere>,
            hw::kHemispheres> relay_fifos{};
        std::size_t current{0};
        std::size_t routed_bytes{0};
        std::size_t vector_offset{0};
        std::size_t accepted_words{0};
        std::size_t next_write_cycle{0};
        std::uint64_t last_generation{0};
        bool start_requested{false};
        bool dependencies_ready{false};
        bool segment_loading{},prime_sent{};
        std::size_t total_committed_bytes{};
        MemProgramEndpoint endpoint{}; MemProgramPortSignals edge{};
    };

    Snapshot capture_snapshot() const
    {
        return Snapshot {state_,stats_,images_,route_,reassembly_,commits_,
            word_fifo_,ingress_fifo_,relay_fifos_,
            current_,routed_bytes_,vector_offset_,accepted_words_,
            next_write_cycle_,last_generation_,start_requested_,
            dependencies_ready_,segment_loading_,prime_sent_,total_committed_bytes_,mem_endpoint_,mem_program_ports_};
    }
    void restore_snapshot(Snapshot snapshot)
    {
        mem_endpoint_=snapshot.endpoint; mem_program_ports_=snapshot.edge;
        state_=snapshot.state;
        stats_=std::move(snapshot.stats);
        images_=std::move(snapshot.images);
        route_=std::move(snapshot.route);
        reassembly_=std::move(snapshot.reassembly);
        commits_=std::move(snapshot.commits);
        word_fifo_=std::move(snapshot.word_fifo);
        ingress_fifo_=std::move(snapshot.ingress_fifo);
        relay_fifos_=std::move(snapshot.relay_fifos);
        current_=snapshot.current;
        routed_bytes_=snapshot.routed_bytes;
        vector_offset_=snapshot.vector_offset;
        accepted_words_=snapshot.accepted_words;
        next_write_cycle_=snapshot.next_write_cycle;
        last_generation_=snapshot.last_generation;
        start_requested_=snapshot.start_requested;
        dependencies_ready_=snapshot.dependencies_ready;
        segment_loading_=snapshot.segment_loading;prime_sent_=snapshot.prime_sent;total_committed_bytes_=snapshot.total_committed_bytes;
    }

    void validate_image(const ProgramImage& image) {
        icu_.validate_program_image(image);
        if(image.source_address%ddr_.config().beat_bytes) throw std::invalid_argument("unaligned program source");
        if(image.vector_count()) {
            const auto dma=C2cDmaInstruction::Load(image.source_address,image.vector_count());
            if(dma.vector_address(image.vector_count()-1)>UINT64_MAX-(hw::kPhysicalVectorBytes-1))
                throw std::invalid_argument("program DMA image crosses the address space");
        }
    }

    static bool uses_rtl_distribution(const ProgramImage& image) noexcept
    {
        return image.storage==ProgramStorage::Imem
            && (image.kind==IcuMacroQueueKind::Mem
                || image.kind==IcuMacroQueueKind::MxmLoad
                || image.kind==IcuMacroQueueKind::MxmDequant
                || image.kind==IcuMacroQueueKind::MxmCompute);
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

    void advance_rtl_distribution(const ProgramImage& image,
        ProgramImageStatistics& record,std::size_t cycle)
    {
        constexpr auto relay_count=
            hw::config::kProgramLoadRelaysPerHemisphere;
        constexpr auto relay_depth=
            hw::config::kProgramLoadRelayBufferDepth;
        const auto [hemisphere,target_relay]=relay_destination(image);
        auto& relays=relay_fifos_.at(hemisphere);
        std::array<bool,relay_count> relay_ready{};
        for (std::size_t relay=0;relay<relay_count;++relay) {
            relay_ready[relay]=relays[relay].size()<relay_depth;
            stats_.peak_relay_level=std::max(
                stats_.peak_relay_level,relays[relay].size());
            stats_.relay_full_cycles+=
                relays[relay].size()==relay_depth;
        }
        stats_.peak_word_fifo=std::max(
            stats_.peak_word_fifo,word_fifo_.size());
        stats_.peak_ingress_fifo=std::max(
            stats_.peak_ingress_fifo,ingress_fifo_.size());
        stats_.word_fifo_full_cycles+=word_fifo_.size()
            == hw::config::kProgramLoadWordFifoDepth;
        stats_.ingress_fifo_full_cycles+=ingress_fifo_.size()
            == hw::config::kProgramLoadIngressFifoDepth;

        auto& endpoint=relays[target_relay];
        bool ready=cycle>=next_write_cycle_ &&
            commits_.size()<=config_.imem_write_commit_latency_cycles;
        if (is_mem_macro(image)) {
            mem_program_ports_.mem_program_ready_i=ready && !mem_endpoint_.last_seen;
            mem_program_ports_.mem_prime_ready_i=ready && mem_endpoint_.last_seen && !mem_endpoint_.primed
                && mem_endpoint_.accepted==mem_endpoint_.committed;
        }
        if (!endpoint.empty() && is_mem_macro(image)) {
            const auto& w=endpoint.front();auto& edge=mem_program_ports_;
            edge.mem_program_queue_o=image.queue_index;edge.mem_program_generation_o=last_generation_;
            // A delayed ACK retains ownership; PRIME can wait on the relay.
            ready &= !w.prime || mem_endpoint_.accepted==mem_endpoint_.committed;
            edge.mem_prime_valid_o=w.prime;edge.mem_program_valid_o=!w.prime;
            if(!w.prime) {
                edge.mem_program_imem_address_o=w.address;edge.mem_program_strb_o=w.strb;edge.mem_program_last_o=w.last;
                std::copy_n(w.bytes.begin(),image.word_bytes(),edge.mem_program_data_o.begin());
            }
            if(ready) {
                if(!(w.prime ? mem_endpoint_.prime(image.word_count())
                             : mem_endpoint_.write(w.address,w.strb,w.last,image.word_count()))) {
                    fail_mem_endpoint();return;
                }
                edge.mem_prime_fire=w.prime;edge.mem_program_fire=!w.prime;
            }
        }
        if (!endpoint.empty() && ready) {
            auto write=std::move(endpoint.front());
            endpoint.pop_front();
            ++stats_.relay_reads;
            write.due=cycle+config_.imem_write_commit_latency_cycles;
            if(write.prime && is_mem_macro(image)) {
                record.prime_accepted=cycle;
                configure_target(image);record.image_committed=cycle;
                if(segment_loading_ && image.storage==ProgramStorage::Imem)icu_.stage_segment_end(image);
            } else {
                commits_.push_back(std::move(write));
                if(commits_.back().prime)record.prime_accepted=cycle;
                else {
                    if(!record.first_write_accepted)record.first_write_accepted=cycle;
                    record.last_write_accepted=cycle;
                }
            }
            next_write_cycle_=cycle+config_.imem_write_ii_cycles;
        }

        // Each registered-ready Relay used its occupancy from the beginning
        // of LOW. Tail-to-head movement permits at most one hop per cycle and
        // deliberately forbids full+dequeue replacement in a Relay.
        for (std::size_t relay=relay_count-1;relay>0;--relay) {
            auto& upstream=relays[relay-1];
            if (relay-1>=target_relay || upstream.empty()
                || !relay_ready[relay])
                continue;
            relays[relay].push_back(std::move(upstream.front()));
            upstream.pop_front();
            ++stats_.relay_reads;
            ++stats_.relay_writes;
        }
        if (!ingress_fifo_.empty() && relay_ready[0]) {
            relays[0].push_back(std::move(ingress_fifo_.front()));
            ingress_fifo_.pop_front();
            ++stats_.ingress_fifo_reads;
            ++stats_.relay_writes;
        }
        if (!word_fifo_.empty() && ingress_fifo_.size()
            < hw::config::kProgramLoadIngressFifoDepth) {
            ingress_fifo_.push_back(std::move(word_fifo_.front()));
            word_fifo_.pop_front();
            ++stats_.word_fifo_reads;
            ++stats_.ingress_fifo_writes;
        }
    }

    void begin_queue()
    {
        mem_endpoint_={};
        prime_sent_=false;
        routed_bytes_ = vector_offset_ = accepted_words_ = next_write_cycle_ = 0;
        const auto& image = images_[current_];
        if(image.word_count()) {
            dma_.start({image.source_address,image.word_count(),
                static_cast<PhysicalInstructionWordWidth>(image.word_bytes()*8),last_generation_});
        } else stats_.images[current_].source_done=stats_.request_accepted;
        state_ = State::LoadQueue;
    }

    void commit_word(const ProgramImage& image,const PendingWrite& write) {
        icu_.stage_program_word(image,write.address,write.bytes.data());
    }
    void configure_target(const ProgramImage& image) {icu_.stage_program_seal(image);}

    static bool is_mem_macro(const ProgramImage& im) {
        return im.storage==ProgramStorage::Imem && im.kind==IcuMacroQueueKind::Mem && im.mode==IcuQueueMode::Macro;
    }
    void fail_mem_endpoint() {
        mem_program_ports_.mem_program_error_i=true;
        mem_program_ports_.mem_program_fault_code_i=mem_endpoint_.error;
        begin_task_abort(); // no new admission or PRIME, retain published AXI obligations
    }
    MemProgramEndpoint mem_endpoint_{};
    MemProgramPortSignals mem_program_ports_{};
    void clear_distribution() {
        word_fifo_.clear();ingress_fifo_.clear();
        for(auto& h:relay_fifos_)for(auto& r:h)r.clear();
        prime_sent_=false;
    }
    std::deque<PendingWrite> word_fifo_,ingress_fifo_;
    std::array<std::array<std::deque<PendingWrite>,hw::config::kProgramLoadRelaysPerHemisphere>,hw::kHemispheres> relay_fifos_;
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
    std::size_t current_{0}, routed_bytes_{0}, vector_offset_{0}, accepted_words_{0}, next_write_cycle_{0};
    std::uint64_t last_generation_{0};
    bool start_requested_{false}, dependencies_ready_{false};
    bool segment_loading_{false},prime_sent_{false},clock_managed_{false},low_complete_{false};
    bool more_segments_pending_{false}; // admission metadata, constant across LOW/posedge
    std::optional<Snapshot> next_snapshot_;
    std::size_t total_committed_bytes_{};
};

} // namespace ftlpu
