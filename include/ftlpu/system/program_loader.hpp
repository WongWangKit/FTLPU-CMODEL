#pragma once

#include "ftlpu/c2c/dma.hpp"
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
    std::vector<ProgramImageStatistics> images{};
    std::array<ProgramStorageStatistics,7> stores{};

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
           << " target_wait_cycles=" << target_wait_cycles << '\n';
    }
};

// One active program, one active destination. DDR is ticked by the enclosing
// system, exactly once per physical cycle. This engine shares that DDR with
// weight/data DMA and reuses C2cDmaEngine's actual FIFO and outstanding credits.
class ProgramLoader {
public:
    enum class State { Idle, LoadQueue, WaitCommit, AllCommitted, Prime, Ready, Started, Draining, Completed };
    static constexpr std::size_t kReassemblyBytes = 64;

    ProgramLoader(Ddr4Model& ddr, InstructionControlUnit& icu,
        std::size_t fifo_depth = hw::config::kDmaFifoDepthVectors,
        ProgramLoadConfig config = {})
        : dma_(ddr, fifo_depth), ddr_(ddr), icu_(icu), config_(config)
    { config_.validate(); }

    void configure(ProgramLoadConfig config)
    {
        if (busy()) throw std::logic_error("program loader is busy");
        config.validate();
        config_ = config;
    }
    const ProgramLoadConfig& config() const noexcept { return config_; }
    C2cDmaEngine& dma_trace_source() noexcept {return dma_;}
    const C2cDmaEngine& dma_trace_source() const noexcept {return dma_;}
    void set_program_reset(std::function<void()> reset) {program_reset_=std::move(reset);}
    State state() const noexcept { return state_; }
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
        return !busy() && (!started() || kernel_done()) && dma_.drained()
            && commits_.empty() && reassembly_.empty()
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
        icu_.set_program_loading(false);icu_.set_program_issue_enabled(true);
    }
    const ProgramLoadStatistics& statistics() const noexcept { return stats_; }
    const C2cDmaEngine& dma() const noexcept { return dma_; }

    void reset()
    {
        dma_.reset();
        state_ = State::Idle;
        stats_ = {};
        last_generation_=0;
        images_.clear(); route_.clear(); reassembly_.clear(); commits_.clear();
        start_requested_ = dependencies_ready_ = false;
    }

    // Validation precedes any mutation, including clearing the old program.
    void submit(std::vector<ProgramImage> images, std::size_t physical_cycle)
    {
        if (!can_load_program()) throw std::logic_error("previous program has not completed and drained");
        const auto group=[](ProgramStorage store) {
            return store==ProgramStorage::Imem?0:(store==ProgramStorage::TransportImem || store==ProgramStorage::RoutingDescriptors)?1:
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
        if (!dma_.idle() || dma_.receive_ready())
            throw std::logic_error("program DMA still has unconsumed traffic");
        std::size_t total_bytes = 0;
        for (const auto& image : images) {
            if (total_bytes > std::numeric_limits<std::size_t>::max() - image.byte_count)
                throw std::overflow_error("aggregate program image size overflow");
            total_bytes += image.byte_count;
        }
        while (dma_.take_completion_notification()) {}
        while (dma_.take_completion_sync_tag()) {}
        dma_.clear_completion_history();
        if(program_reset_) program_reset_(); else icu_.reset_program();
        if(generation) {last_generation_=*generation;icu_.set_program_generation(*generation);}
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
        if (!busy()) return;
        icu_.set_program_issue_enabled(false);
        if (state_ == State::Ready) {
            if (start_requested_ && dependencies_ready_) {
                stats_.program_start = cycle;
                state_ = State::Started;
                icu_.set_program_loading(false);
                icu_.set_program_issue_enabled(true);
            }
            return;
        }
        if (state_ == State::AllCommitted) { state_ = State::Prime; }
        if (state_ == State::Prime) {
            icu_.tick_program_frontends();
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
            while (dma_.take_completion_notification()) {}
            while (dma_.take_completion_sync_tag()) {}
            dma_.clear_completion_history();
            if (++current_ == images_.size()) {
                stats_.all_images_committed = cycle;
                state_ = State::AllCommitted;
            } else begin_queue();
            return; // queue n+1 cannot issue before queue n commits
        }

        if (reassembly_.size() >= image.word_bytes()) {
            if (cycle >= next_write_cycle_
                && commits_.size() <= config_.imem_write_commit_latency_cycles) {
                PendingWrite write{};
                write.address = accepted_words_++;
                write.due = cycle + config_.imem_write_commit_latency_cycles;
                for (std::size_t b = 0; b < image.word_bytes(); ++b) {
                    write.bytes[b] = reassembly_.front();
                    reassembly_.pop_front();
                }
                commits_.push_back(write);
                if (!record.first_write_accepted) record.first_write_accepted = cycle;
                record.last_write_accepted = cycle;
                next_write_cycle_ = cycle + config_.imem_write_ii_cycles;
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
        if (dma_.receive_ready() && routed_bytes_ < image.byte_count) {
            if (!route_.front()) {
                Packet packet{};
                packet.size = std::min({config_.bus_bytes_per_cycle,
                    hw::kPhysicalVectorBytes - vector_offset_,
                    image.byte_count - routed_bytes_});
                const auto& vector = dma_.front_received();
                for (std::size_t b = 0; b < packet.size; ++b) {
                    const auto offset = vector_offset_ + b;
                    packet.bytes[b] = vector.payload[offset / hw::kLanesPerTile]
                        [offset % hw::kLanesPerTile];
                }
                vector_offset_ += packet.size;
                routed_bytes_ += packet.size;
                if (vector_offset_ == hw::kPhysicalVectorBytes || routed_bytes_ == image.byte_count) {
                    dma_.pop_received();
                    vector_offset_ = 0;
                }
                route_.front() = packet;
                if (routed_bytes_ < image.byte_count && vector_offset_ != 0)
                    ++stats_.bus_wait_cycles;
            }
        }

        dma_.tick();
        if (dma_.last_read_request()) {
            if (!stats_.first_ddr_request) stats_.first_ddr_request = cycle;
            if (!record.first_ddr_request) record.first_ddr_request = cycle;
            record.dma_bytes += hw::kPhysicalVectorBytes;
            stats_.dma_bytes += hw::kPhysicalVectorBytes;
            stats_.stores[static_cast<std::size_t>(image.storage)].dma_bytes+=hw::kPhysicalVectorBytes;
        }
        if (dma_.last_beat()) {
            if (!stats_.first_data_return) stats_.first_data_return = cycle;
            if (!record.first_data_return) record.first_data_return = cycle;
        }
        if (dma_.load_credit_wait()) ++stats_.dma_credit_wait_cycles;
        if (dma_.idle() && !record.source_done) {
            record.source_done = cycle;
            state_ = State::WaitCommit;
        }
        stats_.peak_reassembly_bytes = std::max(stats_.peak_reassembly_bytes, reassembly_.size());
        stats_.peak_dma_vectors = std::max(stats_.peak_dma_vectors,
            dma_.outstanding_reads() + dma_.receive_queue_size());
    }

private:
    struct Packet { std::array<std::uint8_t, hw::kPhysicalVectorBytes> bytes{}; std::size_t size{0}; };
    struct PendingWrite { std::array<std::uint8_t, kReassemblyBytes> bytes{}; std::size_t address{0}, due{0}; };

    void validate_image(const ProgramImage& image) {
        icu_.validate_program_image(image);
        if(image.vector_count()) {
            const auto dma=C2cDmaInstruction::Load(image.source_address,image.vector_count());
            if(dma.vector_address(image.vector_count()-1)>UINT64_MAX-(hw::kPhysicalVectorBytes-1))
                throw std::invalid_argument("program DMA image crosses the address space");
        }
    }

    void begin_queue()
    {
        routed_bytes_ = vector_offset_ = accepted_words_ = next_write_cycle_ = 0;
        const auto& image = images_[current_];
        if(image.vector_count()) dma_.issue(C2cDmaInstruction::Load(image.source_address, image.vector_count()));
        else stats_.images[current_].source_done=stats_.request_accepted;
        state_ = State::LoadQueue;
    }

    void commit_word(const ProgramImage& image,const PendingWrite& write) {
        icu_.write_program_word(image,write.address,write.bytes.data());
    }
    void configure_target(const ProgramImage& image) {icu_.seal_program_image(image);}

    C2cDmaEngine dma_;
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
};

} // namespace ftlpu
