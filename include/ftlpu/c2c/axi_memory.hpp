#pragma once
#include "ftlpu/c2c/axi_dma.hpp"
#include <functional>

namespace ftlpu {
// Bounded AXI slave/controller timing model, separate from the master.
// The backing byte store is external DRAM, not an on-chip queue. One R and
// one W transfer at most per edge; aggregate byte service is token limited.
class AxiMemory {
public:
    struct Config {
        std::size_t capacity{256}, read_latency{35}, write_latency{25};
        std::size_t read_jitter{15}, write_jitter{10};
        std::uint64_t clock_hz{500000000}, bytes_per_second{51200000000ULL}, seed{1};
    };
    AxiMemory(AxiDmaConfig geometry, Config config,
        std::function<std::uint8_t(std::uint64_t)> read,
        std::function<void(std::uint64_t,std::uint8_t)> write)
        : g_(geometry),c_(config),read_(std::move(read)),write_(std::move(write)) {
        g_.validate();
        if(!c_.capacity || !c_.clock_hz || !c_.bytes_per_second)throw std::invalid_argument("AXI memory capacity/rate");
    }
    bool drained() const {return reads_.empty() && writes_.empty() && responses_.empty();}
    std::uint64_t read_bytes() const {return read_bytes_;}
    std::uint64_t write_bytes() const {return write_bytes_;}
    struct Trace {bool write{};std::uint64_t address{};std::size_t bytes{};};
    const std::optional<Trace>& trace() const {return trace_;}
    // Environment-only legal RRESP/BRESP fault injection. Writes can have
    // partial external effects even when B reports an error; no rollback.
    void set_response_fault(std::function<std::uint8_t(std::uint64_t,bool)> fault) {
        fault_=std::move(fault);
    }
    // Environment service availability, evaluated once per physical edge.
    // R/B gates delay offering a response; they never retract an offered beat.
    // No backpressure is introduced on the reserved-response master.
    struct ChannelService { bool ar{true}, aw{true}, w{true}, r{true}, b{true}; };
    using ServiceSchedule = std::function<ChannelService(std::size_t)>;
    void set_channel_service(ServiceSchedule schedule) {service_=std::move(schedule);}
    // Testbench hook on the slave-to-master wire, after slave service. Unlike
    // RRESP/BRESP this may corrupt identity/framing and make retirement unsafe.
    using ResponseWire = std::function<void(std::optional<AxiReadBeat>&,std::optional<AxiWriteResponse>&)>;
    void set_response_wire(ResponseWire hook) {response_wire_=std::move(hook);}
    // This environment uses an always-ready master. VALID hold behavior is
    // covered separately by the adversarial channel tests of AxiDmaMaster.
    void tick(AxiDmaMaster& master) {
        const auto o=master.eval();
        if(!o.rready || !o.bready)throw std::logic_error("reserved-response master required");
        trace_.reset();
        remainder_+=c_.bytes_per_second;
        credit_=std::min<std::uint64_t>(2*g_.beat_bytes,credit_+remainder_/c_.clock_hz);
        remainder_%=c_.clock_hz;
        AxiSlaveSignals s;
        const auto service=service_?service_(cycle_):ChannelService{};
        s.arready=service.ar && reads_.size()<c_.capacity;
        s.awready=service.aw && writes_.size()+responses_.size()<c_.capacity;
        const auto service_read=[&] {
        if(!service.r)return;
        // Round-robin across IDs, preserving each ID's order.
        for(std::size_t k=0;k<reads_.size();++k) {
            const auto n=(rr_+k)%reads_.size();auto& r=reads_[n];
            if(r.due>cycle_ || credit_<g_.beat_bytes)continue;
            bool head=true;
            for(std::size_t p=0;p<n;++p)if(reads_[p].a.id==r.a.id)head=false;
            if(!head)continue;
            AxiReadBeat beat;beat.id=r.a.id;beat.last=r.sent+1==r.a.beats;beat.resp=r.response;
            const auto address=r.a.address+r.sent*g_.beat_bytes;
            for(std::size_t b=0;b<g_.beat_bytes;++b)beat.bytes.push_back(read_(address+b));
            s.r=std::move(beat);credit_-=g_.beat_bytes;read_bytes_+=g_.beat_bytes;
            trace_=Trace{false,address,g_.beat_bytes};++r.sent;
            if(r.sent==r.a.beats)reads_.erase(reads_.begin()+n);
            rr_=reads_.empty()?0:(n+1)%reads_.size();break;
        }
        };
        const auto service_write=[&] {
        s.wready=service.w && !writes_.empty() && writes_.front().due<=cycle_ && credit_>=g_.beat_bytes;
        if(o.w && s.wready) {
            auto& w=writes_.front();
            if(o.w->last!=(w.sent+1==w.a.beats))throw std::logic_error("AXI memory WLAST mismatch");
            const auto address=w.a.address+w.sent*g_.beat_bytes;std::size_t bytes=0;
            for(std::size_t b=0;b<g_.beat_bytes;++b)if(o.w->strobe[b]) {write_(address+b,o.w->bytes[b]);++bytes;}
            credit_-=g_.beat_bytes;write_bytes_+=bytes;trace_=Trace{true,address,bytes};
            if(++w.sent==w.a.beats) {responses_.push_back({w.a.id,cycle_+1,w.response});writes_.pop_front();}
        }
        };
        // A sustained read stream must not starve stores when the DDR budget
        // permits only one beat. Alternate after service, not after time:
        // fractional token rates must not phase-lock to a fixed priority.
        if(write_first_) {service_write();service_read();}
        else {service_read();service_write();}
        if(s.r || (o.w && s.wready))write_first_=s.r.has_value();
        // B can be reordered between IDs, never within an ID.
        for(std::size_t n=responses_.size();service.b && n-->0;) {
            if(responses_[n].due>cycle_)continue;
            bool head=true;for(std::size_t p=0;p<n;++p)if(responses_[p].id==responses_[n].id)head=false;
            if(!head)continue;
            s.b=AxiWriteResponse{responses_[n].id,responses_[n].response};responses_.erase(responses_.begin()+n);break;
        }
        if(o.ar && s.arready)reads_.push_back({*o.ar,0,cycle_+1+c_.read_latency+jitter(*o.ar,false),fault_?fault_(o.ar->address,false):std::uint8_t{0}});
        if(o.aw && s.awready)writes_.push_back({*o.aw,0,cycle_+1+c_.write_latency+jitter(*o.aw,true),fault_?fault_(o.aw->address,true):std::uint8_t{0}});
        if(response_wire_)response_wire_(s.r,s.b);
        master.tick(s);++cycle_;
    }
private:
    ServiceSchedule service_;
    ResponseWire response_wire_;
    struct Request {AxiAddress a;std::size_t sent{},due{};std::uint8_t response{};};
    struct Response {std::size_t id{},due{};std::uint8_t response{};};
    std::function<std::uint8_t(std::uint64_t,bool)> fault_;
    std::size_t jitter(const AxiAddress& a,bool write) const {
        auto x=c_.seed^a.address^(a.id*0x9e3779b97f4a7c15ULL);
        x^=x>>30;x*=0xbf58476d1ce4e5b9ULL;x^=x>>27;
        return x%((write?c_.write_jitter:c_.read_jitter)+1);
    }
    AxiDmaConfig g_;Config c_;
    std::function<std::uint8_t(std::uint64_t)> read_;
    std::function<void(std::uint64_t,std::uint8_t)> write_;
    std::deque<Request> reads_,writes_;std::deque<Response> responses_;
    std::size_t cycle_{},rr_{};
    bool write_first_{};
    std::uint64_t credit_{},remainder_{},read_bytes_{},write_bytes_{};
    std::optional<Trace> trace_;
};
} // namespace ftlpu
