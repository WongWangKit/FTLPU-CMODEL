#pragma once

#include "ftlpu/c2c/dma.hpp"
#include "ftlpu/system/icu.hpp"

#include <limits>
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

// Runtime has already resolved the queue and encoded its physical image.
// byte_count includes the physical control word and word padding, but excludes
// DMA vector tail padding. Version 1 accepts only the 96/128-bit Macro codec.
struct ProgramImage {
    std::uint64_t source_address{0};
    IcuMacroQueueKind kind{IcuMacroQueueKind::Mem};
    std::size_t queue_index{0};
    std::size_t byte_count{0};
    IcuQueueMode mode{IcuQueueMode::Macro};
    std::size_t word_bytes() const noexcept
    { return kind == IcuMacroQueueKind::Mem ? 12 : 16; }
    std::size_t word_count() const noexcept { return byte_count / word_bytes(); }
    std::size_t vector_count() const noexcept
    { return byte_count / hw::kPhysicalVectorBytes
        + (byte_count % hw::kPhysicalVectorBytes != 0); }
};

struct ProgramImageStatistics {
    ProgramImage image{};
    std::optional<std::size_t> first_ddr_request{}, first_data_return{};
    std::optional<std::size_t> source_done{}, image_committed{};
    std::optional<std::size_t> first_write_accepted{}, first_write_committed{}, last_write_accepted{};
    std::size_t effective_bytes{0}, dma_bytes{0}, written_words{0};
};

struct ProgramLoadStatistics {
    std::optional<std::size_t> request_accepted{}, first_ddr_request{}, first_data_return{};
    std::optional<std::size_t> all_images_committed{}, frontend_ready{}, program_start{};
    std::size_t effective_bytes{0}, dma_bytes{0}, written_words{0};
    std::size_t dma_credit_wait_cycles{0}, bus_wait_cycles{0}, target_wait_cycles{0};
    std::size_t peak_reassembly_bytes{0}, peak_dma_vectors{0};
    std::vector<ProgramImageStatistics> images{};

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
    enum class State { Idle, LoadQueue, WaitCommit, AllCommitted, Prime, Ready, Started };
    static constexpr std::size_t kReassemblyBytes = 64;

    ProgramLoader(Ddr4Model& ddr, InstructionControlUnit& icu,
        std::size_t fifo_depth = hw::config::kDmaFifoDepthVectors,
        ProgramLoadConfig config = {})
        : dma_(ddr, fifo_depth), icu_(icu), config_(config)
    { config_.validate(); }

    void configure(ProgramLoadConfig config)
    {
        if (busy()) throw std::logic_error("program loader is busy");
        config.validate();
        config_ = config;
    }
    const ProgramLoadConfig& config() const noexcept { return config_; }
    State state() const noexcept { return state_; }
    bool busy() const noexcept
    { return state_ != State::Idle && state_ != State::Started; }
    bool all_images_committed() const noexcept
    { return stats_.all_images_committed.has_value(); }
    bool ready_to_start() const noexcept { return state_ == State::Ready; }
    bool started() const noexcept { return state_ == State::Started; }
    const ProgramLoadStatistics& statistics() const noexcept { return stats_; }
    const C2cDmaEngine& dma() const noexcept { return dma_; }

    void reset()
    {
        dma_.reset();
        state_ = State::Idle;
        stats_ = {};
        images_.clear(); route_.clear(); reassembly_.clear(); commits_.clear();
        start_requested_ = dependencies_ready_ = false;
    }

    // Validation precedes any mutation, including clearing the old program.
    void submit(std::vector<ProgramImage> images, std::size_t physical_cycle)
    {
        if (busy()) throw std::logic_error("only one program load may be active");
        for (std::size_t i = 0; i < images.size(); ++i) {
            validate_image(images[i]);
            for (std::size_t j = 0; j < i; ++j)
                if (images[i].kind == images[j].kind
                    && images[i].queue_index == images[j].queue_index)
                    throw std::invalid_argument("duplicate program queue destination");
        }
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
        icu_.reset_compute_program();
        images_ = std::move(images);
        stats_ = {};
        stats_.request_accepted = physical_cycle;
        for (auto image : images_) {
            stats_.images.push_back(ProgramImageStatistics{image});
            stats_.images.back().effective_bytes = image.byte_count;
            stats_.effective_bytes += image.byte_count;
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
        if (state_ == State::Idle || state_ == State::Started) return;
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
            icu_.tick_raw_macro_frontends();
            if (icu_.raw_macro_frontends_ready()) {
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
        }
        if (record.written_words == image.word_count()) {
            configure_target(image);
            record.image_committed = cycle;
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
    struct PendingWrite { std::array<std::uint8_t, 16> bytes{}; std::size_t address{0}, due{0}; };

    static void validate_image(const ProgramImage& image)
    {
        if (image.mode != IcuQueueMode::Macro
            || static_cast<unsigned>(image.kind) > static_cast<unsigned>(IcuMacroQueueKind::MxmDequant))
            throw std::invalid_argument("unsupported program image physical format");
        const auto depth = image.kind == IcuMacroQueueKind::Mem
            ? hw::kIcuMemImemDepth : hw::kIcuMxmImemDepth;
        const auto queues = image.kind == IcuMacroQueueKind::Mem
            ? InstructionControlUnit::kMemQueues : InstructionControlUnit::kMxmQueues;
        if (image.queue_index >= queues || image.byte_count == 0
            || image.byte_count % image.word_bytes() != 0 || image.word_count() > depth)
            throw std::invalid_argument("invalid program image queue, length or i-MEM capacity");
        const auto dma = C2cDmaInstruction::Load(image.source_address, image.vector_count());
        if (dma.vector_address(image.vector_count() - 1)
            > std::numeric_limits<std::uint64_t>::max() - (hw::kPhysicalVectorBytes - 1))
            throw std::invalid_argument("program DMA image crosses the address space");
    }

    void begin_queue()
    {
        routed_bytes_ = vector_offset_ = accepted_words_ = next_write_cycle_ = 0;
        const auto& image = images_[current_];
        dma_.issue(C2cDmaInstruction::Load(image.source_address, image.vector_count()));
        state_ = State::LoadQueue;
    }

    template <typename Word> static Word decode_word(const PendingWrite& write)
    {
        Word word{};
        for (std::size_t b = 0; b < Word::bit_count / 8; ++b)
            word.lanes[b / 4] |= static_cast<std::uint32_t>(write.bytes[b]) << (8 * (b % 4));
        return word;
    }
    void commit_word(const ProgramImage& image, const PendingWrite& write)
    {
        const auto q = image.queue_index;
        if (image.kind == IcuMacroQueueKind::Mem) {
            icu_.write_mem_raw_macro_imem(q, write.address,
                decode_word<InstructionControlUnit::MemRawImemWord>(write));
            return;
        }
        const auto word = decode_word<InstructionControlUnit::MxmRawImemWord>(write);
        switch (image.kind) {
        case IcuMacroQueueKind::MxmLoad: icu_.write_mxm_load_raw_macro_imem(q, write.address, word); break;
        case IcuMacroQueueKind::MxmCompute: icu_.write_mxm_compute_raw_macro_imem(q, write.address, word); break;
        case IcuMacroQueueKind::MxmDequant: icu_.write_mxm_dequant_raw_macro_imem(q, write.address, word); break;
        default: throw std::logic_error("unsupported program target");
        }
    }
    void configure_target(const ProgramImage& image)
    {
        const auto q = image.queue_index, words = image.word_count();
        switch (image.kind) {
        case IcuMacroQueueKind::Mem: icu_.configure_mem_raw_macro_imem(q, words); break;
        case IcuMacroQueueKind::MxmLoad: icu_.configure_mxm_load_raw_macro_imem(q, words); break;
        case IcuMacroQueueKind::MxmCompute: icu_.configure_mxm_compute_raw_macro_imem(q, words); break;
        case IcuMacroQueueKind::MxmDequant: icu_.configure_mxm_dequant_raw_macro_imem(q, words); break;
        }
    }

    C2cDmaEngine dma_;
    InstructionControlUnit& icu_;
    ProgramLoadConfig config_;
    State state_{State::Idle};
    ProgramLoadStatistics stats_{};
    std::vector<ProgramImage> images_{};
    std::vector<std::optional<Packet>> route_{};
    std::deque<std::uint8_t> reassembly_{};
    std::deque<PendingWrite> commits_{};
    std::size_t current_{0}, routed_bytes_{0}, vector_offset_{0}, accepted_words_{0}, next_write_cycle_{0};
    bool start_requested_{false}, dependencies_ready_{false};
};

} // namespace ftlpu
