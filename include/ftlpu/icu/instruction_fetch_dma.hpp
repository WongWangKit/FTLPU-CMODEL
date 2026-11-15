#pragma once

#include "ftlpu/c2c/ddr4.hpp"

#include <algorithm>
#include <array>
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
    Bits32=32, Bits64=64, Bits96=96, Bits128=128
};
constexpr std::size_t physical_instruction_word_bytes(PhysicalInstructionWordWidth width) noexcept
{ return static_cast<std::size_t>(width)/8; }

struct PhysicalInstructionWord {
    static constexpr std::size_t kMaximumBytes=16;
    std::size_t index{0};
    PhysicalInstructionWordWidth width{PhysicalInstructionWordWidth::Bits96};
    std::size_t byte_count{0};
    std::array<std::uint8_t,kMaximumBytes> bytes{};
    std::uint32_t lane32(std::size_t lane) const {
        const auto offset=lane*4;
        if(offset+4>byte_count) throw std::out_of_range("physical instruction lane is outside the word");
        return static_cast<std::uint32_t>(bytes[offset])
            |(static_cast<std::uint32_t>(bytes[offset+1])<<8)
            |(static_cast<std::uint32_t>(bytes[offset+2])<<16)
            |(static_cast<std::uint32_t>(bytes[offset+3])<<24);
    }
};

struct InstructionFetchProgram {
    std::uint64_t base_address{0};
    std::size_t word_count{0};
    PhysicalInstructionWordWidth word_width{PhysicalInstructionWordWidth::Bits96};
};

// Defaults match lpu_instruction_dma.sv: 256-bit responses, eight-bit IDs,
// 16 command-outstanding entries and a 16-entry response FIFO.
struct InstructionFetchDmaConfig {
    std::size_t data_width_bits{256};
    std::size_t id_width_bits{8};
    std::size_t max_outstanding{hw::config::kProgramLoadOutstandingDepth};
    std::size_t response_fifo_depth{hw::config::kProgramLoadResponseFifoDepth};
    std::size_t data_bytes() const noexcept {return data_width_bits/8;}
    void validate() const {
        if(data_width_bits!=256 || data_bytes()!=hw::kPhysicalVectorBytes)
            throw std::invalid_argument("instruction DMA must use the RTL 256-bit response width");
        if(id_width_bits==0 || id_width_bits>8)
            throw std::invalid_argument("instruction DMA ID width must be in [1, 8]");
        if(max_outstanding==0 || max_outstanding>(std::size_t{1}<<id_width_bits))
            throw std::invalid_argument("instruction DMA outstanding depth exceeds its ID space");
        if(response_fifo_depth==0)
            throw std::invalid_argument("instruction DMA response FIFO depth must be non-zero");
    }
};

class InstructionWordAssembler {
public:
    InstructionWordAssembler(std::size_t word_bytes,std::size_t response_bytes)
        :word_bytes_(word_bytes),storage_(response_bytes+word_bytes-1,0) {
        if((word_bytes!=4&&word_bytes!=8&&word_bytes!=12&&word_bytes!=16)||response_bytes==0)
            throw std::invalid_argument("instruction assembler requires 4/8/12/16-byte records and a response width");
    }
    std::size_t capacity() const noexcept{return storage_.size();}
    std::size_t occupancy() const noexcept{return size_;}
    bool empty() const noexcept{return size_==0;}
    bool word_ready() const noexcept{return size_>=word_bytes_;}
    bool can_accept(std::size_t n) const noexcept{return n<=capacity()-size_;}
    void append(std::span<const std::uint8_t> bytes) {
        if(!can_accept(bytes.size())) throw std::logic_error("instruction assembler capacity was exceeded");
        for(auto byte:bytes){storage_[(head_+size_)%capacity()]=byte;++size_;}
    }
    PhysicalInstructionWord front(std::size_t index,PhysicalInstructionWordWidth width) const {
        if(!word_ready()) throw std::logic_error("instruction assembler has no complete word");
        PhysicalInstructionWord word{};word.index=index;word.width=width;word.byte_count=word_bytes_;
        for(std::size_t b=0;b<word_bytes_;++b) word.bytes[b]=storage_[(head_+b)%capacity()];
        return word;
    }
    void pop_word(){
        if(!word_ready()) throw std::logic_error("instruction assembler cannot pop an incomplete word");
        head_=(head_+word_bytes_)%capacity();size_-=word_bytes_;
    }
private:
    std::size_t word_bytes_{0};
    std::vector<std::uint8_t> storage_{};
    std::size_t head_{0},size_{0};
};

struct InstructionFetchBurstTrace {
    std::size_t request_cycle{0};std::uint8_t dma_id{0};std::uint64_t address{0};
    std::size_t valid_bytes{0};bool last{false};
};
struct InstructionFetchDmaTrace {
    std::vector<InstructionFetchBurstTrace> bursts{};
    std::vector<std::size_t> response_input_cycles{},word_assembled_cycles{},word_enqueue_cycles{};
};
struct InstructionFetchDmaStatistics {
    std::size_t cycle_count{0},command_handshakes{0},ar_handshakes{0},r_handshakes{0};
    std::size_t bytes_received{0},words_assembled{0},words_enqueued{0};
    std::size_t response_fifo_full_cycles{0},queue_backpressure_cycles{0},ddr_credit_wait_cycles{0};
    std::size_t outstanding_peak{0},response_fifo_peak{0},assembler_peak_bytes{0};
    double effective_payload_bytes_per_cycle() const noexcept {
        return cycle_count==0?0.0:static_cast<double>(bytes_received)/static_cast<double>(cycle_count);
    }
};

// The DMA owns the RTL command-ID and response FIFOs. Ddr4Model remains the
// one shared memory/bandwidth authority for instruction and C2C data traffic.
class InstructionFetchDma:public timing::ClockedModule {
    friend struct InstructionFetchDmaTestAccess;
public:
    explicit InstructionFetchDma(Ddr4Model& ddr,InstructionFetchDmaConfig config={})
        :ddr_(ddr),config_(config){config_.validate();}
    void enable_clocked_mode(){
        if(clock_managed_||state_.cycle!=0||next_) throw std::logic_error("instruction DMA clocked mode must be enabled before execution");
        clock_managed_=true;
    }
    bool clocked_mode() const noexcept{return clock_managed_;}
    bool can_accept_program() const noexcept {
        return !state_.fault&&!pending_command_&&state_.commands.size()<config_.max_outstanding;
    }
    std::uint8_t start(InstructionFetchProgram program){
        require_input_window("command");validate_program(program);
        if(!can_accept_program()) throw std::logic_error("instruction DMA outstanding command FIFO is full");
        const auto id=allocate_dma_id();pending_command_=make_command(id,program);return id;
    }
    bool word_ready() const noexcept {
        return !state_.commands.empty()&&state_.commands.front().assembler.word_ready();
    }
    PhysicalInstructionWord front_word() const {
        if(!word_ready()) throw std::logic_error("instruction DMA has no assembled word");
        const auto& c=state_.commands.front();return c.assembler.front(c.words_enqueued,c.program.word_width);
    }
    void pop_word(){
        require_input_window("word dequeue");
        if(pending_word_pop_||!word_ready()) throw std::logic_error("instruction DMA word dequeue is invalid or duplicated");
        pending_word_pop_=true;
    }
    void low() override {
        if(!clock_managed_){tick_immediate();return;}
        if(next_) throw std::logic_error("instruction DMA LOW executed more than once");
        prepare_next();
    }
    void validate_posedge() const override {
        if(clock_managed_&&!next_) throw std::logic_error("instruction DMA posedge requires one complete LOW");
    }
    void posedge() override {
        if(!clock_managed_)return;validate_posedge();state_=std::move(*next_);next_.reset();
        pending_command_.reset();pending_word_pop_=false;
    }
    void high() override{}
    void tick(){
        if(clock_managed_) throw std::logic_error("clock-managed instruction DMA must run through the system clock");
        tick_immediate();
    }
    void reset(){
        if(next_) throw std::logic_error("instruction DMA reset cannot interrupt LOW");
        for(const auto& request:state_.transport) state_.orphan_requests.push_back(request.ddr_id);
        state_.commands.clear();state_.outstanding_ids.clear();state_.responses.clear();state_.transport.clear();
        state_.fault=false;state_.fault_code=0;pending_command_.reset();pending_word_pop_=false;low_observation_={};
    }
    bool idle() const noexcept{return is_drained(state_);}
    bool drained() const noexcept{return is_drained(state_)&&!pending_command_;}
    bool complete() const noexcept{return drained();}
    bool fault() const noexcept{return state_.fault;}
    std::uint8_t fault_code() const noexcept{return state_.fault_code;}
    std::size_t cycle() const noexcept{return state_.cycle;}
    std::size_t outstanding() const noexcept{return state_.outstanding_ids.size();}
    std::size_t response_fifo_occupancy() const noexcept{return state_.responses.size();}
    std::size_t assembler_capacity() const noexcept{return state_.commands.empty()?0:state_.commands.front().assembler.capacity();}
    std::size_t assembler_occupancy() const noexcept{return state_.commands.empty()?0:state_.commands.front().assembler.occupancy();}
    const InstructionFetchDmaConfig& config() const noexcept{return config_;}
    const InstructionFetchDmaTrace& trace() const noexcept{return trace_;}
    const InstructionFetchDmaStatistics& statistics() const noexcept{return statistics_;}
    bool low_last_read_request() const noexcept{return low_observation_.read_request;}
    bool low_last_beat() const noexcept{return low_observation_.response_input;}
    bool low_load_credit_wait() const noexcept{return low_observation_.credit_wait;}
    bool low_idle() const noexcept{return next_?is_drained(*next_):is_drained(state_);}
    std::size_t low_outstanding_reads() const noexcept{return (next_?next_->transport:state_.transport).size();}
    std::size_t low_receive_queue_size() const noexcept{return (next_?next_->responses:state_.responses).size();}

private:
    struct Command {
        std::uint8_t id{0};InstructionFetchProgram program{};
        std::size_t total_bytes{0},bytes_requested{0},bytes_assembled{0},words_assembled{0},words_enqueued{0};
        bool last_response_seen{false};InstructionWordAssembler assembler;
        Command(std::uint8_t id_,InstructionFetchProgram p,std::size_t response_bytes)
            :id(id_),program(p),total_bytes(p.word_count*physical_instruction_word_bytes(p.word_width)),
             assembler(physical_instruction_word_bytes(p.word_width),response_bytes){}
    };
    struct TransportRequest {Ddr4Model::RequestId ddr_id{0};std::uint8_t dma_id{0};std::uint64_t address{0};std::size_t valid_bytes{0};bool last{false};};
    struct ResponseBeat {std::array<std::uint8_t,hw::kPhysicalVectorBytes> data{};std::size_t valid_bytes{0};std::uint8_t dma_id{0};bool last{false},error{false};};
    struct State {
        std::deque<Command> commands{};std::deque<std::uint8_t> outstanding_ids{};
        std::deque<ResponseBeat> responses{};std::deque<TransportRequest> transport{};
        std::deque<Ddr4Model::RequestId> orphan_requests{};
        std::uint16_t next_dma_id{0};std::size_t cycle{0};bool fault{false};std::uint8_t fault_code{0};
    };
    struct LowObservation {bool read_request{false},response_input{false},credit_wait{false};};
    static void validate_program(const InstructionFetchProgram& p){
        const auto wb=physical_instruction_word_bytes(p.word_width);
        if(wb!=4&&wb!=8&&wb!=12&&wb!=16)
            throw std::invalid_argument("instruction-fetch record width must be 32, 64, 96 or 128 bits");
        if(p.word_count>std::numeric_limits<std::size_t>::max()/wb) throw std::out_of_range("instruction-fetch byte count overflows");
        const auto bytes=p.word_count*wb;
        if(bytes){const auto requests=(bytes+hw::kPhysicalVectorBytes-1)/hw::kPhysicalVectorBytes;
            if(requests-1>(std::numeric_limits<std::uint64_t>::max()-p.base_address)/hw::kPhysicalVectorBytes)
                throw std::out_of_range("instruction-fetch padded DDR reads cross the address space");}
    }
    Command make_command(std::uint8_t id,const InstructionFetchProgram& p) const{return Command{id,p,config_.data_bytes()};}
    std::uint8_t allocate_dma_id() const {
        const auto count=std::size_t{1}<<config_.id_width_bits;
        for(std::size_t off=0;off<count;++off){const auto candidate=static_cast<std::uint8_t>((state_.next_dma_id+off)%count);
            const auto used=std::any_of(state_.commands.begin(),state_.commands.end(),[&](const auto& c){return c.id==candidate;});
            if(!used)return candidate;}
        throw std::logic_error("instruction DMA ID space is exhausted");
    }
    void require_input_window(const char* op) const {
        if(clock_managed_&&next_) throw std::logic_error(std::string("instruction DMA ")+op+" arrived after LOW");
    }
    static bool command_complete(const Command& c) noexcept {
        return c.last_response_seen&&c.words_enqueued==c.program.word_count&&c.assembler.empty();
    }
    static bool is_drained(const State& s) noexcept {
        return s.commands.empty()&&s.outstanding_ids.empty()&&s.responses.empty()&&s.transport.empty()&&s.orphan_requests.empty();
    }
    static void set_fault(State& s,std::uint8_t code){if(!s.fault){s.fault=true;s.fault_code=code;}}
    static void retire_commands(State& s){while(!s.commands.empty()&&command_complete(s.commands.front()))s.commands.pop_front();}
    void apply_pending_inputs(State& s){
        if(pending_word_pop_){
            if(s.commands.empty()||!s.commands.front().assembler.word_ready()) throw std::logic_error("staged instruction word disappeared before LOW");
            auto& c=s.commands.front();c.assembler.pop_word();trace_.word_enqueue_cycles.push_back(s.cycle);++c.words_enqueued;++statistics_.words_enqueued;
        }
        retire_commands(s);
        if(pending_command_){auto c=*pending_command_;const auto count=std::size_t{1}<<config_.id_width_bits;
            s.next_dma_id=static_cast<std::uint16_t>((static_cast<std::size_t>(c.id)+1)%count);
            if(c.total_bytes==0)c.last_response_seen=true;
            else s.outstanding_ids.push_back(c.id);
            s.commands.push_back(std::move(c));++statistics_.command_handshakes;
            retire_commands(s);}
    }
    void consume_response(State& s){
        if(s.responses.empty()||s.commands.empty())return;auto& c=s.commands.front();const auto& r=s.responses.front();
        if(r.dma_id!=c.id){set_fault(s,1);return;}if(r.error){set_fault(s,2);return;}
        if(!c.assembler.can_accept(r.valid_bytes))return;
        c.assembler.append(std::span<const std::uint8_t>(r.data.data(),r.valid_bytes));c.bytes_assembled+=r.valid_bytes;
        if(r.last){if(c.bytes_assembled!=c.total_bytes)set_fault(s,3);c.last_response_seen=true;}s.responses.pop_front();
        const auto complete=c.bytes_assembled/physical_instruction_word_bytes(c.program.word_width);
        while(c.words_assembled<complete){trace_.word_assembled_cycles.push_back(s.cycle);++c.words_assembled;++statistics_.words_assembled;}
    }
    static bool can_consume_response(const State& s) noexcept {
        if(s.responses.empty()||s.commands.empty())return false;
        const auto& c=s.commands.front();const auto& r=s.responses.front();
        return r.dma_id==c.id&&!r.error
            &&c.assembler.can_accept(r.valid_bytes);
    }
    void accept_ddr_response(State& s,bool ready){
        if(!ready||s.transport.empty())return;const auto request=s.transport.front();
        if(!ddr_.read_completion_ready(request.ddr_id))return;const auto completion=ddr_.pop_read_completion(request.ddr_id);
        ResponseBeat response{};response.valid_bytes=request.valid_bytes;response.dma_id=request.dma_id;response.last=request.last;
        for(std::size_t b=0;b<hw::kPhysicalVectorBytes;++b)response.data[b]=completion.vector.payload[b/hw::kLanesPerTile][b%hw::kLanesPerTile];
        s.responses.push_back(std::move(response));s.transport.pop_front();trace_.response_input_cycles.push_back(s.cycle);
        ++statistics_.r_handshakes;statistics_.bytes_received+=request.valid_bytes;low_observation_.response_input=true;
        if(request.last){if(s.outstanding_ids.empty()||s.outstanding_ids.front()!=request.dma_id)set_fault(s,1);else s.outstanding_ids.pop_front();}
    }
    void issue_ddr_request(State& s){
        auto found=std::find_if(s.commands.begin(),s.commands.end(),[](const auto& c){return c.bytes_requested<c.total_bytes;});
        if(found==s.commands.end()||s.fault)return;if(!ddr_.can_accept_request()){low_observation_.credit_wait=true;++statistics_.ddr_credit_wait_cycles;return;}
        const auto remaining=found->total_bytes-found->bytes_requested;const auto valid=std::min(remaining,hw::kPhysicalVectorBytes);
        const auto address=found->program.base_address+found->bytes_requested;const auto last=valid==remaining;const auto ddrId=ddr_.request_read(address);
        s.transport.push_back({ddrId,found->id,address,valid,last});found->bytes_requested+=valid;
        trace_.bursts.push_back({s.cycle,found->id,address,valid,last});++statistics_.ar_handshakes;low_observation_.read_request=true;
    }
    void drain_orphan_response(State& s){
        if(s.orphan_requests.empty())return;const auto id=s.orphan_requests.front();if(!ddr_.read_completion_ready(id))return;
        (void)ddr_.pop_read_completion(id);s.orphan_requests.pop_front();
    }
    void prepare_next(){
        next_=state_;low_observation_={};
        if(word_ready()&&!pending_word_pop_)++statistics_.queue_backpressure_cycles;
        apply_pending_inputs(*next_);drain_orphan_response(*next_);
        const bool fullAtCycleStart=
            state_.responses.size()==config_.response_fifo_depth;
        const bool responseReady=!fullAtCycleStart
            ||can_consume_response(*next_);
        if(fullAtCycleStart)++statistics_.response_fifo_full_cycles;
        if(!responseReady)low_observation_.credit_wait=true;
        consume_response(*next_);
        accept_ddr_response(*next_,responseReady);issue_ddr_request(*next_);retire_commands(*next_);
        statistics_.outstanding_peak=std::max(statistics_.outstanding_peak,next_->outstanding_ids.size());
        statistics_.response_fifo_peak=std::max(statistics_.response_fifo_peak,next_->responses.size());
        if(!next_->commands.empty())statistics_.assembler_peak_bytes=std::max(statistics_.assembler_peak_bytes,next_->commands.front().assembler.occupancy());
        ++next_->cycle;statistics_.cycle_count=next_->cycle;
    }
    void tick_immediate(){prepare_next();state_=std::move(*next_);next_.reset();pending_command_.reset();pending_word_pop_=false;}

    Ddr4Model& ddr_;InstructionFetchDmaConfig config_{};State state_{};std::optional<State> next_{};
    std::optional<Command> pending_command_{};bool pending_word_pop_{false},clock_managed_{false};
    LowObservation low_observation_{};InstructionFetchDmaTrace trace_{};InstructionFetchDmaStatistics statistics_{};
};

} // namespace ftlpu
