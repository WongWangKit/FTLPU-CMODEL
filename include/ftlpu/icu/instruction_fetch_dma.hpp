#pragma once

#include "ftlpu/c2c/ddr4.hpp"
#include "ftlpu/icu/instruction_dma_protocol_core.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ftlpu {

enum class PhysicalInstructionWordWidth : std::size_t {
    Bits32 = 32,
    Bits64 = 64,
    Bits96 = 96,
    Bits128 = 128,
};

constexpr std::size_t physical_instruction_word_bytes(
    PhysicalInstructionWordWidth width) noexcept
{
    return static_cast<std::size_t>(width) / 8;
}

struct PhysicalInstructionWord {
    static constexpr std::size_t kMaximumBytes = 16;
    std::size_t index{0};
    PhysicalInstructionWordWidth width{PhysicalInstructionWordWidth::Bits96};
    std::size_t byte_count{0};
    std::array<std::uint8_t, kMaximumBytes> bytes{};

    std::uint32_t lane32(std::size_t lane) const
    {
        const auto offset = lane * 4;
        if (offset + 4 > byte_count)
            throw std::out_of_range("physical instruction lane is outside the word");
        return static_cast<std::uint32_t>(bytes[offset])
            | (static_cast<std::uint32_t>(bytes[offset + 1]) << 8)
            | (static_cast<std::uint32_t>(bytes[offset + 2]) << 16)
            | (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
    }
};

struct InstructionFetchProgram {
    std::uint64_t base_address{0};
    std::size_t word_count{0};
    PhysicalInstructionWordWidth word_width{PhysicalInstructionWordWidth::Bits96};
    std::uint64_t generation{1};
};

// Defaults match lpu_instruction_dma.sv: 256-bit responses, eight-bit IDs,
// 16 command-outstanding entries and a 16-entry response FIFO.
struct InstructionFetchDmaConfig {
    std::size_t data_width_bits{256};
    std::size_t id_width_bits{8};
    std::size_t max_outstanding{hw::config::kProgramLoadOutstandingDepth};
    std::size_t response_fifo_depth{hw::config::kC2cDmaSizing.p};

    [[nodiscard]] std::size_t data_bytes() const noexcept
    {
        return data_width_bits / 8;
    }

    [[nodiscard]] InstructionDmaProtocolConfig protocol_config() const noexcept
    {
        return {data_width_bits, id_width_bits, max_outstanding,
            response_fifo_depth};
    }

    void validate() const
    {
        protocol_config().validate();

    }
};

class InstructionWordAssembler {
public:
    InstructionWordAssembler(std::size_t word_bytes,
        std::size_t response_bytes)
        : word_bytes_(word_bytes), storage_(response_bytes + word_bytes - 1, 0)
    {
        if ((word_bytes != 4 && word_bytes != 8 && word_bytes != 12
                && word_bytes != 16)
            || response_bytes == 0)
            throw std::invalid_argument("instruction assembler requires 4/8/12/16-byte records and a response width");
    }

    [[nodiscard]] std::size_t capacity() const noexcept { return storage_.size(); }
    [[nodiscard]] std::size_t occupancy() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] bool word_ready() const noexcept { return size_ >= word_bytes_; }
    [[nodiscard]] bool can_accept(std::size_t count) const noexcept
    {
        return count <= capacity() - size_;
    }

    void append(std::span<const std::uint8_t> bytes)
    {
        if (!can_accept(bytes.size()))
            throw std::logic_error("instruction assembler capacity was exceeded");
        for (auto byte : bytes) {
            storage_[(head_ + size_) % capacity()] = byte;
            ++size_;
        }
    }

    [[nodiscard]] PhysicalInstructionWord front(std::size_t index,
        PhysicalInstructionWordWidth width) const
    {
        if (!word_ready())
            throw std::logic_error("instruction assembler has no complete word");
        PhysicalInstructionWord word{};
        word.index = index;
        word.width = width;
        word.byte_count = word_bytes_;
        for (std::size_t byte = 0; byte < word_bytes_; ++byte)
            word.bytes[byte] = storage_[(head_ + byte) % capacity()];
        return word;
    }

    void pop_word()
    {
        if (!word_ready())
            throw std::logic_error("instruction assembler cannot pop an incomplete word");
        head_ = (head_ + word_bytes_) % capacity();
        size_ -= word_bytes_;
    }

private:
    std::size_t word_bytes_{0};
    std::vector<std::uint8_t> storage_{};
    std::size_t head_{0};
    std::size_t size_{0};
};

struct InstructionFetchBurstTrace {
    std::size_t request_cycle{0};
    std::uint8_t dma_id{0};
    std::uint64_t address{0};
    std::size_t valid_bytes{0};
    bool last{false};
};

struct InstructionFetchDmaTrace {
    std::vector<InstructionFetchBurstTrace> bursts{};
    std::vector<std::size_t> response_input_cycles{};
    std::vector<std::size_t> word_assembled_cycles{};
    std::vector<std::size_t> word_enqueue_cycles{};
};

struct InstructionFetchDmaStatistics {
    std::size_t cycle_count{0};
    std::size_t command_handshakes{0};
    std::size_t ar_handshakes{0};
    std::size_t r_handshakes{0};
    std::size_t bytes_received{0};
    std::size_t words_assembled{0};
    std::size_t words_enqueued{0};
    std::size_t response_fifo_full_cycles{0};
    std::size_t queue_backpressure_cycles{0};
    std::size_t ddr_credit_wait_cycles{0};
    std::size_t outstanding_peak{0};
    std::size_t response_fifo_peak{0};
    std::size_t assembler_peak_bytes{0};

    [[nodiscard]] double effective_payload_bytes_per_cycle() const noexcept
    {
        return cycle_count == 0 ? 0.0
                                : static_cast<double>(bytes_received)
                / static_cast<double>(cycle_count);
    }
};

// Ddr4Model remains the single memory/bandwidth authority. ProtocolCore is the
// logical command-ID, response-FIFO and fault authority; the shared AXI Master
// alone owns physical AXI IDs/credits. This wrapper only
// adapts accepted commands to DDR reads and reassembles instruction records.
class InstructionFetchDma final : public timing::ClockedModule {
public:
    explicit InstructionFetchDma(Ddr4Model& ddr,
        InstructionFetchDmaConfig config = {})
        : ddr_(ddr), config_(config), protocol_(config.protocol_config())
    {
        config_.validate();
    }

    void enable_clocked_mode()
    {
        if (clock_managed_ || state_.cycle != 0 || next_)
            throw std::logic_error("instruction DMA clocked mode must be enabled before execution");
        clock_managed_ = true;
    }

    [[nodiscard]] bool clocked_mode() const noexcept { return clock_managed_; }

    [[nodiscard]] bool can_accept_program() const noexcept
    {
        return !ddr_.axi().isolated() && !aborting_ && !protocol_.fault() && !pending_command_
            && state_.commands.size() < config_.max_outstanding;
    }

    std::uint8_t start(InstructionFetchProgram program)
    {
        require_input_window("command");
        validate_program(program);
        if (!can_accept_program())
            throw std::logic_error("instruction DMA command window is full");
        const auto id = allocate_dma_id();
        pending_command_ = make_command(id, program);
        return id;
    }

    [[nodiscard]] bool word_ready() const noexcept
    {
        return !state_.commands.empty()
            && state_.commands.front().assembler.word_ready();
    }

    [[nodiscard]] PhysicalInstructionWord front_word() const
    {
        if (!word_ready())
            throw std::logic_error("instruction DMA has no assembled word");
        const auto& command = state_.commands.front();
        return command.assembler.front(command.words_enqueued,
            command.program.word_width);
    }

    void pop_word()
    {
        require_input_window("word dequeue");
        if (pending_word_pop_ || !word_ready())
            throw std::logic_error("instruction DMA word dequeue is invalid or duplicated");
        pending_word_pop_ = true;
    }

    void low() override
    {
        if (!clock_managed_) {
            tick_immediate();
            return;
        }
        if (next_)
            throw std::logic_error("instruction DMA LOW executed more than once");
        prepare_next();
    }

    void validate_posedge() const override
    {
        if (clock_managed_ && !next_)
            throw std::logic_error("instruction DMA posedge requires one complete LOW");
        protocol_.validate_posedge();
    }

    void posedge() override
    {
        if (!clock_managed_) return;
        validate_posedge();
        state_ = std::move(*next_);
        next_.reset();
        protocol_.posedge();
        pending_command_.reset();
        pending_word_pop_ = false;
    }

    void high() override { protocol_.high(); }

    void tick()
    {
        if (clock_managed_)
            throw std::logic_error("clock-managed instruction DMA must run through the system clock");
        tick_immediate();
    }

    void configure(InstructionFetchDmaConfig config) {
        if(next_ || !drained())throw std::logic_error("instruction DMA configure before drain");
        config.validate();config_=config;protocol_=InstructionDmaProtocolCore(config.protocol_config());
    }
    // Task cancellation drains every accepted shared-Master ticket. It does
    // not reset DDR/time or resume a partially overwritten instruction slot.
    void begin_abort() {
        if(next_)throw std::logic_error("instruction DMA abort outside cycle boundary");
        ddr_.axi().abort_client(Ddr4Model::kProgramClient);
        reset();aborting_=true;
    }
    void recover() {
        if(next_ || !fault() || !drained())throw std::logic_error("instruction DMA fault not drained");
        ddr_.axi().resume_client(Ddr4Model::kProgramClient);
        reset();aborting_=false;
    }
    std::size_t pending_reads() const noexcept {return state_.transport.size()+state_.orphan_requests.size();}
    std::size_t low_read_request_bytes() const noexcept {return low_observation_.request_bytes;}
    void set_trace_enabled(bool enabled) {trace_enabled_=enabled;if(!enabled)trace_={};}
    void reset()
    {
        if (next_)
            throw std::logic_error("instruction DMA reset cannot interrupt LOW");
        State clean{};
        clean.next_dma_id = state_.next_dma_id;
        clean.cycle = state_.cycle;
        clean.orphan_requests = state_.orphan_requests;
        for (const auto& request : state_.transport)
            clean.orphan_requests.push_back(request.ddr_id);
        state_ = std::move(clean);
        protocol_.reset();
        pending_command_.reset();
        pending_word_pop_ = false;
        low_observation_ = {};
    }

    [[nodiscard]] bool idle() const noexcept
    {
        return is_drained(state_, protocol_.idle());
    }
    [[nodiscard]] bool drained() const noexcept
    {
        return idle() && !pending_command_;
    }
    [[nodiscard]] bool complete() const noexcept { return drained(); }
    [[nodiscard]] bool fault() const noexcept { return protocol_.fault() || aborting_; }
    [[nodiscard]] std::uint8_t fault_code() const noexcept
    {
        return protocol_.fault_code();
    }
    [[nodiscard]] std::size_t cycle() const noexcept { return state_.cycle; }
    [[nodiscard]] std::size_t outstanding() const noexcept
    {
        return protocol_.outstanding();
    }
    [[nodiscard]] std::size_t response_fifo_occupancy() const noexcept
    {
        return protocol_.response_fifo_occupancy();
    }
    [[nodiscard]] std::size_t assembler_capacity() const noexcept
    {
        return state_.commands.empty() ? 0
                                       : state_.commands.front().assembler.capacity();
    }
    [[nodiscard]] std::size_t assembler_occupancy() const noexcept
    {
        return state_.commands.empty() ? 0
                                       : state_.commands.front().assembler.occupancy();
    }
    [[nodiscard]] const InstructionFetchDmaConfig& config() const noexcept
    {
        return config_;
    }
    [[nodiscard]] const InstructionFetchDmaTrace& trace() const noexcept
    {
        return trace_;
    }
    [[nodiscard]] const InstructionFetchDmaStatistics& statistics() const noexcept
    {
        return statistics_;
    }
    [[nodiscard]] bool low_last_read_request() const noexcept
    {
        return low_observation_.read_request;
    }
    [[nodiscard]] bool low_last_beat() const noexcept
    {
        return low_observation_.response_input;
    }
    [[nodiscard]] bool low_load_credit_wait() const noexcept
    {
        return low_observation_.credit_wait;
    }
    [[nodiscard]] bool low_idle() const noexcept
    {
        return next_ ? is_drained(*next_, protocol_.next_idle()) : idle();
    }
    [[nodiscard]] std::size_t low_outstanding_reads() const noexcept
    {
        return protocol_.next_outstanding();
    }
    [[nodiscard]] std::size_t low_receive_queue_size() const noexcept
    {
        return protocol_.next_response_fifo_occupancy();
    }

private:
    struct Command {
        std::uint8_t id{0};
        InstructionFetchProgram program{};
        std::size_t total_bytes{0};
        std::size_t bytes_assembled{0};
        std::size_t words_assembled{0};
        std::size_t words_enqueued{0};
        bool protocol_issued{false};
        bool last_response_seen{false};
        InstructionWordAssembler assembler;

        Command(std::uint8_t id_, InstructionFetchProgram program_,
            std::size_t response_bytes)
            : id(id_), program(program_),
              total_bytes(program_.word_count
                  * physical_instruction_word_bytes(program_.word_width)),
              assembler(physical_instruction_word_bytes(program_.word_width),
                  response_bytes)
        {
        }
    };

    struct DdrCommand {
        std::uint8_t dma_id{0};
        std::uint64_t address{0};
        std::size_t total_bytes{0};
        std::size_t bytes_requested{0};
        std::size_t bytes_completed{0};
        std::uint64_t generation{0};
    };

    struct TransportRequest {
        Ddr4Model::RequestId ddr_id{0};
        std::uint8_t dma_id{0};
        std::uint64_t address{0};
        std::size_t valid_bytes{0};
        bool last{false};
    };

    struct State {
        std::deque<Command> commands{};
        std::deque<DdrCommand> ddr_commands{};
        std::deque<TransportRequest> transport{};
        std::deque<Ddr4Model::RequestId> orphan_requests{};
        std::uint16_t next_dma_id{0};
        std::size_t cycle{0};
    };

    struct LowObservation {
        bool read_request{false};
        bool response_input{false};
        bool credit_wait{false};
        std::size_t request_bytes{};
    };

    static void validate_program(const InstructionFetchProgram& program)
    {
        const auto wordBytes = physical_instruction_word_bytes(program.word_width);
        if (wordBytes != 4 && wordBytes != 8 && wordBytes != 12
            && wordBytes != 16)
            throw std::invalid_argument("instruction-fetch record width must be 32, 64, 96 or 128 bits");
        if (program.word_count > std::numeric_limits<std::size_t>::max() / wordBytes)
            throw std::out_of_range("instruction-fetch byte count overflows");
        const auto bytes = program.word_count * wordBytes;
        if (bytes > std::numeric_limits<std::uint32_t>::max())
            throw std::out_of_range("instruction-fetch command length exceeds the RTL field");
        if (bytes) {
            const auto requests = (bytes + hw::kPhysicalVectorBytes - 1)
                / hw::kPhysicalVectorBytes;
            if (requests - 1
                > (std::numeric_limits<std::uint64_t>::max() - program.base_address)
                    / hw::kPhysicalVectorBytes)
                throw std::out_of_range("instruction-fetch padded DDR reads cross the address space");
        }
    }

    [[nodiscard]] Command make_command(std::uint8_t id,
        const InstructionFetchProgram& program) const
    {
        return Command{id, program, config_.data_bytes()};
    }

    [[nodiscard]] std::uint8_t allocate_dma_id() const
    {
        const auto count = std::size_t{1} << config_.id_width_bits;
        for (std::size_t offset = 0; offset < count; ++offset) {
            const auto candidate = static_cast<std::uint8_t>(
                (state_.next_dma_id + offset) % count);
            const auto used = std::any_of(state_.commands.begin(),
                state_.commands.end(), [&](const auto& command) {
                    return command.id == candidate;
                });
            if (!used) return candidate;
        }
        throw std::logic_error("instruction DMA ID space is exhausted");
    }

    void require_input_window(const char* operation) const
    {
        if (clock_managed_ && next_)
            throw std::logic_error(std::string("instruction DMA ")
                + operation + " arrived after LOW");
    }

    static bool command_complete(const Command& command) noexcept
    {
        return command.protocol_issued && command.last_response_seen
            && command.words_enqueued == command.program.word_count
            && command.assembler.empty();
    }

    static bool is_drained(const State& state, bool protocolIdle) noexcept
    {
        return protocolIdle && state.commands.empty()
            && state.ddr_commands.empty() && state.transport.empty()
            && state.orphan_requests.empty();
    }

    static void retire_commands(State& state)
    {
        while (!state.commands.empty() && command_complete(state.commands.front()))
            state.commands.pop_front();
    }

    static std::uint32_t keep_mask(std::size_t validBytes)
    {
        if (validBytes > InstructionDmaResponseBeat::kDataBytes)
            throw std::logic_error("instruction DMA response byte count is invalid");
        return validBytes == InstructionDmaResponseBeat::kDataBytes
            ? std::numeric_limits<std::uint32_t>::max()
            : (std::uint32_t{1} << validBytes) - 1;
    }

    static std::size_t keep_bytes(std::uint32_t keep) noexcept
    {
        return static_cast<std::size_t>(std::popcount(keep));
    }

    static auto find_command(State& state, std::uint8_t id)
    {
        return std::find_if(state.commands.begin(), state.commands.end(),
            [id](const auto& command) { return command.id == id; });
    }

    void apply_pending_inputs(State& state)
    {
        if (pending_word_pop_) {
            if (state.commands.empty()
                || !state.commands.front().assembler.word_ready())
                throw std::logic_error("staged instruction word disappeared before LOW");
            auto& command = state.commands.front();
            command.assembler.pop_word();
            if(trace_enabled_)trace_.word_enqueue_cycles.push_back(state.cycle);
            ++command.words_enqueued;
            ++statistics_.words_enqueued;
        }
        retire_commands(state);

        if (pending_command_) {
            auto command = *pending_command_;
            const auto count = std::size_t{1} << config_.id_width_bits;
            state.next_dma_id = static_cast<std::uint16_t>(
                (static_cast<std::size_t>(command.id) + 1) % count);
            if (command.total_bytes == 0) {
                command.protocol_issued = true;
                command.last_response_seen = true;
            }
            state.commands.push_back(std::move(command));
            retire_commands(state);
        }
    }

    void drain_orphan_response(State& state)
    {
        if (state.orphan_requests.empty()) return;
        const auto id = state.orphan_requests.front();
        if (!ddr_.axi().completion_ready(Ddr4Model::kProgramClient)
            || ddr_.axi().peek(Ddr4Model::kProgramClient).ticket!=id) return;
        (void)ddr_.axi().pop(Ddr4Model::kProgramClient);
        state.orphan_requests.pop_front();
    }

    [[nodiscard]] std::optional<InstructionDmaResponseBeat>
    pending_ddr_response(const State& state) const
    {
        if (state.transport.empty()) return std::nullopt;
        const auto& request = state.transport.front();
        if (!ddr_.axi().completion_ready(Ddr4Model::kProgramClient)) return std::nullopt;
        const auto completion = ddr_.axi().peek(Ddr4Model::kProgramClient);
        if (completion.ticket!=request.ddr_id) return std::nullopt;
        InstructionDmaResponseBeat beat{};
        beat.keep = keep_mask(request.valid_bytes);
        beat.last = request.last;
        beat.id = request.dma_id;
        beat.error=completion.error || completion.aborted;
        const auto prefix=request.address%ddr_.axi().config().beat_bytes;
        for(std::size_t byte=0;!beat.error && byte<request.valid_bytes;++byte)
            beat.data[byte]=completion.bytes.at(prefix+byte);
        return beat;
    }

    [[nodiscard]] bool reassembler_ready(const State& state) const
    {
        if (!protocol_.response_valid()) return false;
        const auto& response = protocol_.response_front();
        const auto found = std::find_if(state.commands.begin(),
            state.commands.end(), [&](const auto& command) {
                return command.id == response.id;
            });
        if (response.error || found == state.commands.end())
            return true;
        return found->assembler.can_accept(keep_bytes(response.keep));
    }

    void consume_protocol_response(State& state)
    {
        if (!protocol_.response_fire()) return;
        const auto& response = protocol_.response_front();
        auto found = find_command(state, response.id);
        if (response.error || found == state.commands.end())
            return;

        std::array<std::uint8_t, InstructionDmaResponseBeat::kDataBytes> compact{};
        std::size_t count = 0;
        for (std::size_t lane = 0;
             lane < InstructionDmaResponseBeat::kDataBytes; ++lane) {
            if ((response.keep >> lane) & 1U)
                compact[count++] = response.data[lane];
        }
        if (!found->assembler.can_accept(count))
            throw std::logic_error("instruction DMA consumed a response without assembler capacity");
        found->assembler.append(std::span<const std::uint8_t>(compact.data(), count));
        found->bytes_assembled += count;
        if (response.last) {
            if (found->bytes_assembled != found->total_bytes)
                throw std::logic_error("instruction DMA last response length is inconsistent");
            found->last_response_seen = true;
        }
        const auto complete = found->bytes_assembled
            / physical_instruction_word_bytes(found->program.word_width);
        while (found->words_assembled < complete) {
            if(trace_enabled_)trace_.word_assembled_cycles.push_back(state.cycle);
            ++found->words_assembled;
            ++statistics_.words_assembled;
        }
    }

    void accept_ddr_response(State& state)
    {
        if (!protocol_.response_input_fire()) return;
        if (state.transport.empty())
            throw std::logic_error("instruction DMA accepted a response without a DDR request");
        const auto request = state.transport.front();
        (void)ddr_.axi().pop(Ddr4Model::kProgramClient);
        state.transport.pop_front();

        auto command = std::find_if(state.ddr_commands.begin(),
            state.ddr_commands.end(), [&](const auto& item) {
                return item.dma_id == request.dma_id;
            });
        if (command == state.ddr_commands.end())
            throw std::logic_error("instruction DMA DDR response has no adapter command");
        command->bytes_completed += request.valid_bytes;
        if (command->bytes_completed == command->total_bytes)
            state.ddr_commands.erase(command);

        if(trace_enabled_)trace_.response_input_cycles.push_back(state.cycle);
        ++statistics_.r_handshakes;
        statistics_.bytes_received += request.valid_bytes;
        low_observation_.response_input = true;
    }

    void accept_protocol_command(State& state)
    {
        if (!protocol_.request_fire()) return;
        auto found = find_command(state, protocol_.request_id());
        if (found == state.commands.end() || found->protocol_issued)
            throw std::logic_error("instruction DMA protocol command has no program owner");
        found->protocol_issued = true;
        state.ddr_commands.push_back({protocol_.request_id(),
            protocol_.request_address(), protocol_.request_length(), 0, 0,found->program.generation});
        ++statistics_.command_handshakes;
    }

    void issue_ddr_request(State& state)
    {
        if(aborting_ || protocol_.fault())return;
        // Reserve each protocol response slot before submitting a physical
        // read. Account for the input just removed from transport this LOW.
        if(state.transport.size()+protocol_.response_fifo_occupancy()
            +(protocol_.response_input_fire()?1:0)>=config_.response_fifo_depth) {
            low_observation_.credit_wait=true;++statistics_.ddr_credit_wait_cycles;return;
        }
        auto found = std::find_if(state.ddr_commands.begin(),
            state.ddr_commands.end(), [](const auto& command) {
                return command.bytes_requested < command.total_bytes;
            });
        if (found == state.ddr_commands.end()) return;
        const auto remaining=found->total_bytes-found->bytes_requested;
        const auto address=found->address+found->bytes_requested;
        const auto& axi=ddr_.axi().config();
        const auto prefixBytes=address%axi.beat_bytes;
        const auto budget=std::min(axi.max_burst_beats,axi.rx_beats)*axi.beat_bytes-prefixBytes;
        const auto valid=std::min({remaining,config_.data_bytes(),budget,
            static_cast<std::size_t>(4096-address%4096)});
        const auto prefix=address%ddr_.axi().config().beat_bytes;
        const auto id=ddr_.axi().submit_read(Ddr4Model::kProgramClient,
            address-prefix,prefix+valid,found->generation);
        if(!id) {
            low_observation_.credit_wait=true;
            ++statistics_.ddr_credit_wait_cycles;return;
        }
        const auto ddrId=*id;const auto last=valid==remaining;
        state.transport.push_back({ddrId, found->dma_id, address, valid, last});
        found->bytes_requested += valid;
        if(trace_enabled_)trace_.bursts.push_back({state.cycle, found->dma_id, address, valid, last});
        ++statistics_.ar_handshakes;
        low_observation_.read_request = true;
        low_observation_.request_bytes=((prefix+valid+axi.beat_bytes-1)/axi.beat_bytes)*axi.beat_bytes;
    }

    void prepare_next()
    {
        next_ = state_;
        low_observation_ = {};
        if (word_ready() && !pending_word_pop_)
            ++statistics_.queue_backpressure_cycles;
        apply_pending_inputs(*next_);
        drain_orphan_response(*next_);

        InstructionDmaProtocolInputs inputs{};
        const auto command = std::find_if(next_->commands.begin(),
            next_->commands.end(), [](const auto& item) {
                return !item.protocol_issued && item.total_bytes != 0;
            });
        if (command != next_->commands.end()) {
            inputs.command_valid = true;
            inputs.command_address = command->program.base_address;
            inputs.command_length = static_cast<std::uint32_t>(command->total_bytes);
            inputs.command_id = command->id;
        }
        inputs.request_ready = true;
        if (const auto response = pending_ddr_response(*next_)) {
            inputs.response_valid = true;
            inputs.response = *response;
        }
        inputs.reassembler_ready = reassembler_ready(*next_);

        const bool fullAtCycleStart = protocol_.response_fifo_occupancy()
            == config_.response_fifo_depth;
        if (fullAtCycleStart)
            ++statistics_.response_fifo_full_cycles;

        protocol_.drive(inputs);
        consume_protocol_response(*next_);
        accept_ddr_response(*next_);
        accept_protocol_command(*next_);
        issue_ddr_request(*next_);
        retire_commands(*next_);
        protocol_.low();

        if (inputs.response_valid && !protocol_.response_input_fire())
            low_observation_.credit_wait = true;
        statistics_.outstanding_peak = std::max(statistics_.outstanding_peak,
            protocol_.next_outstanding());
        statistics_.response_fifo_peak = std::max(statistics_.response_fifo_peak,
            protocol_.next_response_fifo_occupancy());
        if (!next_->commands.empty())
            statistics_.assembler_peak_bytes = std::max(
                statistics_.assembler_peak_bytes,
                next_->commands.front().assembler.occupancy());
        ++next_->cycle;
        statistics_.cycle_count = next_->cycle;
    }

    void tick_immediate()
    {
        prepare_next();
        protocol_.validate_posedge();
        state_ = std::move(*next_);
        next_.reset();
        protocol_.posedge();
        pending_command_.reset();
        pending_word_pop_ = false;
    }

    Ddr4Model& ddr_;
    InstructionFetchDmaConfig config_{};
    InstructionDmaProtocolCore protocol_;
    State state_{};
    std::optional<State> next_{};
    std::optional<Command> pending_command_{};
    bool pending_word_pop_{false};
    bool clock_managed_{false};
    bool aborting_{false},trace_enabled_{false};
    LowObservation low_observation_{};
    InstructionFetchDmaTrace trace_{};
    InstructionFetchDmaStatistics statistics_{};
};

} // namespace ftlpu
