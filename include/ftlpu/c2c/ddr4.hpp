#pragma once

#include "ftlpu/c2c/types.hpp"
#include "ftlpu/core/timing.hpp"
#include "ftlpu/c2c/axi_memory.hpp"
#include <memory>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ftlpu {

struct Ddr4Config {
    std::size_t beat_bytes{hw::kC2cBytesPerStreamPerCycle};
    // Default platform: a 500 MHz LPU attached to dual-channel DDR4-3200.
    // 51.2 GB/s corresponds to an average 102.4 bytes per LPU cycle.
    std::size_t read_latency_cycles{35};
    std::size_t write_latency_cycles{25};
    std::size_t request_queue_depth{hw::config::kC2cDmaSizing.q};
    // Platform logical lane count, retained in the platform configuration.
    // This never multiplies the physical AXI width: at most one R and one W
    // beat per edge, additionally limited by the aggregate DDR byte budget.
    std::size_t transfer_channels{
        hw::kHemispheres * hw::kC2cStreamsPerDirection};
    std::uint64_t lpu_clock_hz{500'000'000};
    std::uint64_t peak_bandwidth_bytes_per_second{51'200'000'000};
    std::size_t read_latency_jitter_cycles{15};
    std::size_t write_latency_jitter_cycles{10};
    std::uint64_t latency_random_seed{0x46544c5055444452ULL};

    void validate() const
    {
        if (beat_bytes == 0 || beat_bytes > 64 || (beat_bytes & (beat_bytes-1))) {
            throw std::invalid_argument(
                "DDR4 beat_bytes must be a power of two up to 64 bytes");
        }
        if (request_queue_depth == 0) {
            throw std::invalid_argument(
                "DDR4 request queue depth must be non-zero");
        }
        if (transfer_channels == 0
            || transfer_channels
                > hw::kHemispheres * hw::kC2cStreamsPerDirection) {
            throw std::invalid_argument(
                "DDR4 transfer channel count exceeds the aggregate C2C fabric");
        }
        if (lpu_clock_hz == 0
            || peak_bandwidth_bytes_per_second == 0) {
            throw std::invalid_argument(
                "DDR4 clock and peak bandwidth must be non-zero");
        }
    }
};

// Vector endpoint requests and program bytes share this one physical master.
// Completed data retain their admission credits until the owner consumes them.
class Ddr4Model : public timing::ClockedModule {
public:
    using RequestId=AxiDmaMaster::Ticket;
    static constexpr std::size_t kLaneClients=2*hw::kHemispheres*hw::kC2cStreamsPerDirection;
    static constexpr std::size_t kDirectClient=kLaneClients, kProgramClient=kLaneClients+1;
    struct ReadCompletion {RequestId id{};std::uint64_t address{};C2cVector vector{};bool error{},aborted{};std::size_t transferred_bytes{};};
    struct WriteCompletion {RequestId id{};std::uint64_t address{};bool error{},aborted{};std::size_t transferred_bytes{};};
    struct BeatTrace {RequestId id{};bool write{};std::uint64_t address{};std::size_t byte_count{};};
    explicit Ddr4Model(Ddr4Config config={}) : config_(config), master_(geometry(config)) {make_memory();}
    Ddr4Model(const Ddr4Model&)=delete;
    Ddr4Model& operator=(const Ddr4Model&)=delete;
    void enable_clocked_mode() {
        if(clock_managed_ || cycle_)throw std::logic_error("DDR clock attachment after start");
        clock_managed_=true;
    }
    bool clocked_mode() const noexcept {return clock_managed_;}
    // Client submission reserves bounded ownership during LOW. Physical bus
    // handshakes, payload writes and response visibility occur only at POSEDGE.
    void low() override {
        if(master_.isolated())throw AxiResponseProtocolError("DDR link isolated");
        if(low_prepared_)throw std::logic_error("DDR LOW twice");
        low_prepared_=true;
    }
    void validate_posedge() const override {
        if(!low_prepared_)throw std::logic_error("DDR edge without LOW");
    }
    void posedge() override {validate_posedge();tick_edge();low_prepared_=false;}
    void high() override {}
    AxiDmaMaster& axi() {return master_;}
    const AxiDmaMaster& axi() const {return master_;}
    void set_response_fault(std::function<std::uint8_t(std::uint64_t,bool)> fault) {
        memory_->set_response_fault(std::move(fault));
    }
    // Test environment control; configure/reset deliberately clears schedules.
    void set_channel_service(AxiMemory::ServiceSchedule schedule) {
        memory_->set_channel_service(std::move(schedule));
    }
    void set_response_wire(AxiMemory::ResponseWire hook) {memory_->set_response_wire(std::move(hook));}
    void configure(Ddr4Config config) {
        if(!drained())throw std::logic_error("DDR reconfigure before drain");
        config.validate();config_=config;master_=AxiDmaMaster(geometry(config));make_memory();cycle_=0;
    }
    void reset_execution_state() {
        if(!drained())throw std::logic_error("drain accepted DDR transactions before reset");
        master_=AxiDmaMaster(geometry(config_));owners_.clear();make_memory();cycle_=0;last_beat_.reset();
    }
    const Ddr4Config& config() const {return config_;}
    std::size_t cycle() const {return cycle_;}
    bool idle() const {return master_.transport_idle() && memory_->drained();}
    bool drained() const {return !master_.isolated() && master_.drained() && owners_.empty() && memory_->drained();}
    const std::optional<BeatTrace>& last_beat() const {return last_beat_;}
    std::uint64_t read_bytes_transferred() const {return memory_->read_bytes();}
    std::uint64_t write_bytes_transferred() const {return memory_->write_bytes();}
    std::uint64_t bytes_transferred() const {return read_bytes_transferred()+write_bytes_transferred();}
    std::size_t read_vector_service_cycles() const {
        return config_.read_latency_cycles+config_.read_latency_jitter_cycles
            +(hw::kPhysicalVectorBytes+config_.beat_bytes-1)/config_.beat_bytes+2;
    }
    bool can_accept_request(std::uint64_t address=0) const {
        const auto slots=hw::kPhysicalVectorBytes>4096-address%4096?2u:1u;
        return owners_.size()<config_.request_queue_depth
            && master_.counters().live+slots<=master_.config().transactions
            && master_.counters().live+slots<=master_.config().ids*master_.config().per_id
            && master_.can_submit(hw::kPhysicalVectorBytes,false)
            && master_.can_submit(hw::kPhysicalVectorBytes,true);
    }
    RequestId request_read(std::uint64_t address,std::optional<std::size_t> lane=std::nullopt) {
        validate(address);const auto client=client_for(lane);
        Owner o{};o.client=client;o.address=address;
        for(const auto& range:plan_axi_bursts(master_.config(),address,hw::kPhysicalVectorBytes)) {
            auto id=master_.submit_read(client,range.address,range.bytes);
            if(!id)throw std::logic_error("DDR read admission without credit");
            o.parts[o.count++]=*id;
        }
        o.ticket=o.parts[0];owners_.push_back(o);return o.ticket;
    }
    RequestId request_write(std::uint64_t address,C2cVector vector,std::optional<std::size_t> lane=std::nullopt) {
        validate(address);const auto client=client_for(lane);std::vector<std::uint8_t> data;
        for(const auto& tile:vector.payload)for(auto b:tile)data.push_back(b);
        Owner o{};o.client=client;o.address=address;o.write=true;std::size_t offset=0;
        for(const auto& range:plan_axi_bursts(master_.config(),address,data.size())) {
            std::vector<std::uint8_t> fragment(data.begin()+offset,data.begin()+offset+range.bytes);
            auto id=master_.submit_write(client,range.address,fragment);
            if(!id)throw std::logic_error("DDR write admission without credit");
            o.parts[o.count++]=*id;offset+=range.bytes;
        }
        o.ticket=o.parts[0];owners_.push_back(o);return o.ticket;
    }
    bool read_completion_ready(RequestId id) const {return ready(id,false);}
    bool write_completion_ready(RequestId id) const {return ready(id,true);}
    ReadCompletion pop_read_completion(RequestId id) {
        auto c=pop(id,false);ReadCompletion r{id,c.address,{},c.error,c.aborted,c.transferred_bytes};
        for(std::size_t b=0;b<c.bytes.size();++b)r.vector.payload[b/hw::kLanesPerTile][b%hw::kLanesPerTile]=c.bytes[b];
        return r;
    }
    WriteCompletion pop_write_completion(RequestId id) {auto c=pop(id,true);return {id,c.address,c.error,c.aborted,c.transferred_bytes};}
    void initialize_vector(std::uint64_t address,const C2cVector& vector) {
        if(address>UINT64_MAX-hw::kPhysicalVectorBytes)throw std::out_of_range("DDR initialization range");
        for(std::size_t b=0;b<hw::kPhysicalVectorBytes;++b)put(address+b,vector.payload[b/hw::kLanesPerTile][b%hw::kLanesPerTile]);
    }
    C2cVector read_vector(std::uint64_t address) const {
        if(address>UINT64_MAX-hw::kPhysicalVectorBytes)throw std::out_of_range("DDR inspection range");
        C2cVector v;for(std::size_t b=0;b<hw::kPhysicalVectorBytes;++b)v.payload[b/hw::kLanesPerTile][b%hw::kLanesPerTile]=byte(address+b);return v;
    }
    void tick() {low();posedge();high();}
private:
    void tick_edge() {
        try {memory_->tick(master_);}
        catch(const AxiResponseProtocolError&) {master_.isolate_link();throw;}
        last_beat_.reset();
        if(auto t=memory_->trace())last_beat_=BeatTrace{0,t->write,t->address,t->bytes};
        ++cycle_;
    }
private:
    struct Owner {RequestId ticket{};std::size_t client{};std::uint64_t address{};bool write{};
        std::array<RequestId,2> parts{};std::size_t count{};};
    static AxiDmaConfig geometry(const Ddr4Config& c) {
        c.validate();AxiDmaConfig a;a.beat_bytes=c.beat_bytes;a.max_burst_beats=256;
        a.ids=8;a.per_id=(c.request_queue_depth+7)/8;a.transactions=c.request_queue_depth;
        a.rx_beats=a.tx_beats=c.request_queue_depth*((hw::kPhysicalVectorBytes+c.beat_bytes-1)/c.beat_bytes);
        a.clients=kLaneClients+2;a.address_limit=UINT64_MAX;return a;
    }
    void make_memory() {
        AxiMemory::Config c;c.capacity=config_.request_queue_depth;c.read_latency=config_.read_latency_cycles;
        c.write_latency=config_.write_latency_cycles;c.read_jitter=config_.read_latency_jitter_cycles;
        c.write_jitter=config_.write_latency_jitter_cycles;c.clock_hz=config_.lpu_clock_hz;
        c.bytes_per_second=config_.peak_bandwidth_bytes_per_second;c.seed=config_.latency_random_seed;
        memory_=std::make_unique<AxiMemory>(master_.config(),c,[this](auto a){return byte(a);},[this](auto a,auto b){put(a,b);});
    }
    std::size_t client_for(std::optional<std::size_t> lane) const {
        if(lane && *lane>=kLaneClients)throw std::out_of_range("DDR lane");return lane.value_or(kDirectClient);
    }
    void validate(std::uint64_t address) const {
        if(!can_accept_request(address))throw std::logic_error("DDR request table full");
        if(address%config_.beat_bytes || address>UINT64_MAX-hw::kPhysicalVectorBytes)
            throw std::invalid_argument("DMA vector alignment/range");
    }
    bool ready(RequestId id,bool write) const {
        for(const auto& o:owners_)if(o.ticket==id && o.write==write) {
            for(std::size_t i=0;i<o.count;++i)if(!master_.ticket_done(o.parts[i]))return false;
            return master_.completion_ready(o.client) && master_.peek(o.client).ticket==id;
        }
        return false;
    }
    AxiDmaMaster::Completion pop(RequestId id,bool write) {
        if(!ready(id,write))throw std::logic_error("DDR owner completion not ready");
        const auto it=std::find_if(owners_.begin(),owners_.end(),[&](const auto& o){return o.ticket==id;});
        AxiDmaMaster::Completion c;c.ticket=id;c.address=it->address;c.write=write;
        for(std::size_t i=0;i<it->count;++i) {
            auto part=master_.pop(it->client);
            if(part.ticket!=it->parts[i])throw std::logic_error("vector burst ownership");
            c.bytes.insert(c.bytes.end(),part.bytes.begin(),part.bytes.end());
            c.error|=part.error;c.aborted|=part.aborted;c.transferred_bytes+=part.transferred_bytes;
        }
        owners_.erase(it);
        return c;
    }
    std::uint8_t byte(std::uint64_t address) const {auto p=pages_.find(address/4096);return p==pages_.end()?0:p->second[address%4096];}
    void put(std::uint64_t address,std::uint8_t value) {pages_[address/4096][address%4096]=value;}
    bool clock_managed_{},low_prepared_{};
    Ddr4Config config_;AxiDmaMaster master_;std::unique_ptr<AxiMemory> memory_;
    std::unordered_map<std::uint64_t,std::array<std::uint8_t,4096>> pages_;
    std::deque<Owner> owners_;std::size_t cycle_{};std::optional<BeatTrace> last_beat_;
};
} // namespace ftlpu
