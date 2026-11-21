#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ftlpu {

// Cycle model of one physical AXI4 INCR master. eval() observes registered
// outputs; tick() samples all five handshakes at one edge. No time advances
// on submit/pop. Aligned full-width addresses with a byte-enabled tail.
// Payload, transaction, per-ID and completion storage share explicit limits.
struct AxiDmaConfig {
    std::size_t beat_bytes{16}, max_burst_beats{16}, ids{4}, per_id{2};
    std::size_t transactions{8}, rx_beats{128}, tx_beats{128}, clients{3};
    std::uint64_t address_limit{1ULL << 40}; // exclusive
    void validate() const {
        if (!beat_bytes || beat_bytes > 128 || (beat_bytes & (beat_bytes-1))
            || !max_burst_beats || max_burst_beats > 256 || !ids || !per_id
            || !transactions || !rx_beats || !tx_beats || !clients
            || address_limit < beat_bytes)
            throw std::invalid_argument("invalid AXI DMA geometry/capacity");
    }
};
struct AxiAddress {
    std::uint64_t address{};
    std::size_t id{}, beats{}, beat_bytes{};
    bool operator==(const AxiAddress&) const = default;
};
struct AxiWriteBeat {
    std::vector<std::uint8_t> bytes{}, strobe{};
    bool last{};
    bool operator==(const AxiWriteBeat&) const = default;
};
struct AxiReadBeat {
    std::size_t id{};
    std::vector<std::uint8_t> bytes{};
    bool last{};
    std::uint8_t resp{};
};
struct AxiWriteResponse { std::size_t id{}; std::uint8_t resp{}; };
struct AxiMasterSignals {
    std::optional<AxiAddress> ar{}, aw{};
    std::optional<AxiWriteBeat> w{};
    bool rready{true}, bready{true};
};
struct AxiSlaveSignals {
    bool arready{}, awready{}, wready{};
    std::optional<AxiReadBeat> r{};
    std::optional<AxiWriteResponse> b{};
};
struct AxiResponseProtocolError : std::logic_error { using std::logic_error::logic_error; };
struct AxiBurstRange { std::uint64_t address{}; std::size_t bytes{}; };

// Host-side planning only, not an unbounded hardware descriptor queue.
inline std::vector<AxiBurstRange> plan_axi_bursts(
    const AxiDmaConfig& c, std::uint64_t address, std::size_t bytes)
{
    c.validate();
    if (address % c.beat_bytes || address >= c.address_limit
        || bytes > c.address_limit-address)
        throw std::invalid_argument("AXI request alignment/range");
    std::vector<AxiBurstRange> out;
    while (bytes) {
        const auto n = std::min({bytes, c.max_burst_beats*c.beat_bytes,
            static_cast<std::size_t>(4096-(address%4096))});
        out.push_back({address,n});
        address += n; bytes -= n;
    }
    return out;
}

class AxiDmaMaster {
public:
    using Ticket = std::uint64_t;
    struct Completion {
        Ticket ticket{};
        std::size_t client{};
        bool write{}, error{}, aborted{};
        std::uint64_t address{}, generation{};
        std::vector<std::uint8_t> bytes{};
        std::size_t transferred_bytes{};
    };
    struct Counters {
        std::size_t live{}, rx_reserved{}, tx_reserved{}, peak_live{}, peak_rx{}, peak_tx{};
        std::uint64_t ar{}, aw{}, r{}, w{}, b{}, cycles{};
    };
    explicit AxiDmaMaster(AxiDmaConfig config = {}) : c_(config) {
        c_.validate(); clients_.resize(c_.clients); stopped_.resize(c_.clients);
    }
    const AxiDmaConfig& config() const { return c_; }
    const Counters& counters() const { return counts_; }
    // A corrupted wire response has no trustworthy owner. Preserve every
    // reservation for diagnosis; software abort/retry cannot resynchronise IDs.
    void isolate_link() noexcept { isolated_ = true; }
    bool isolated() const noexcept { return isolated_; }
    bool drained() const { return transactions_.empty(); }
    bool transport_idle() const {
        return std::all_of(transactions_.begin(),transactions_.end(),[](const auto& t){return t.done;});
    }
    bool client_stopped(std::size_t client) const { return stopped_.at(client); }
    bool ticket_done(Ticket ticket) const {
        for(const auto& t:transactions_)if(t.ticket==ticket)return t.done;
        return false;
    }
    bool can_submit(std::size_t bytes,bool write) const {
        const auto beats=(bytes+c_.beat_bytes-1)/c_.beat_bytes;
        return !isolated_ && bytes && counts_.live<c_.transactions && counts_.live<c_.ids*c_.per_id
            && beats<=(write?c_.tx_beats-counts_.tx_reserved:c_.rx_beats-counts_.rx_reserved);
    }

    std::optional<Ticket> submit_read(std::size_t client, std::uint64_t address,
        std::size_t bytes, std::uint64_t generation=1)
    { return submit(client,address,bytes,generation,false,{}); }
    std::optional<Ticket> submit_write(std::size_t client, std::uint64_t address,
        const std::vector<std::uint8_t>& bytes, std::uint64_t generation=1)
    { return submit(client,address,bytes.size(),generation,true,bytes); }

    // Stop new admission; already advertised offers and admitted transactions
    // remain live until their AXI response. No implicit retry/rollback.
    void abort_client(std::size_t client) {
        stopped_.at(client)=true;
        for(auto& t:transactions_) if(t.client==client) t.aborted=true;
    }
    void resume_client(std::size_t client) {
        if(!clients_.at(client).empty()) throw std::logic_error("resume before retirement");
        stopped_[client]=false;
    }

    AxiMasterSignals eval() const {
        AxiMasterSignals o;
        for(const auto& t:transactions_) {
            if(!t.write && !t.address_sent && !o.ar)
                o.ar=AxiAddress{t.address,t.id,t.beats,c_.beat_bytes};
            if(t.write && !t.address_sent && !o.aw)
                o.aw=AxiAddress{t.address,t.id,t.beats,c_.beat_bytes};
            if(t.write && t.sent<t.beats && !o.w) {
                AxiWriteBeat w;
                w.bytes.resize(c_.beat_bytes); w.strobe.resize(c_.beat_bytes);
                for(std::size_t i=0;i<c_.beat_bytes;++i) {
                    const auto offset=t.sent*c_.beat_bytes+i;
                    if(offset<t.length) {w.bytes[i]=t.data[offset]; w.strobe[i]=1;}
                }
                w.last=t.sent+1==t.beats; o.w=std::move(w);
            }
        }
        return o;
    }
    void tick(const AxiSlaveSignals& i) {
        if(isolated_) throw AxiResponseProtocolError("AXI link isolated; platform reset required");
        // Validate returned traffic against pre-edge state. A response cannot
        // refer to an address/WLAST first accepted at this same sampling edge.
        Transaction* r=nullptr; Transaction* b=nullptr;
        if(i.r) {
            if(i.r->id>=c_.ids || i.r->bytes.size()!=c_.beat_bytes || i.r->resp>3)
                throw AxiResponseProtocolError("invalid R payload");
            for(auto& t:transactions_)
                if(!t.write && t.id==i.r->id && !t.done) {r=&t;break;}
            if(!r || !r->address_sent || i.r->last!=(r->received+1==r->beats))
                throw AxiResponseProtocolError("unsolicited/out-of-order R or RLAST mismatch");
        }
        if(i.b) {
            if(i.b->id>=c_.ids || i.b->resp>3) throw AxiResponseProtocolError("invalid B payload");
            for(auto& t:transactions_)
                if(t.write && t.id==i.b->id && !t.done) {b=&t;break;}
            if(!b || !b->address_sent || b->sent!=b->beats)
                throw AxiResponseProtocolError("B before AW/WLAST or unsolicited BID");
        }
        const auto out=eval();
        if(out.ar && i.arready) {
            for(auto& t:transactions_) if(!t.write && !t.address_sent) {
                t.address_sent=true; ++counts_.ar; break;
            }
        }
        if(out.aw && i.awready) {
            for(auto& t:transactions_) if(t.write && !t.address_sent) {
                t.address_sent=true; ++counts_.aw; break;
            }
        }
        if(out.w && i.wready) {
            for(auto& t:transactions_) if(t.write && t.sent<t.beats) {
                ++t.sent; ++counts_.w; break;
            }
        }
        if(r) {
            const auto offset=r->received*c_.beat_bytes;
            const auto n=std::min(c_.beat_bytes,r->length-offset);
            std::copy_n(i.r->bytes.begin(),n,r->data.begin()+offset);
            ++r->received; ++counts_.r;
            r->error |= i.r->resp>=2;
            if(r->error) stopped_[r->client]=true;
            r->done=r->received==r->beats;
        }
        if(b) {
            b->done=true; b->error=i.b->resp>=2; ++counts_.b;
            if(b->error) stopped_[b->client]=true;
        }
        ++counts_.cycles;
        check_invariants();
    }
    const Completion peek(std::size_t client) const {
        const auto* t=front(client);
        if(!t || !t->done) throw std::logic_error("client completion not ready");
        return {t->ticket,t->client,t->write,t->error,t->aborted,t->address,
            t->generation,t->write?std::vector<std::uint8_t>{}:t->data,
            std::min(t->length,(t->write?t->sent:t->received)*c_.beat_bytes)};
    }
    bool completion_ready(std::size_t client) const {
        const auto* t=front(client);return t && t->done;
    }
    Completion pop(std::size_t client) {
        auto result=peek(client);
        auto it=std::find_if(transactions_.begin(),transactions_.end(),
            [&](const auto& t){return t.ticket==result.ticket;});
        (it->write?counts_.tx_reserved:counts_.rx_reserved)-=it->beats;
        transactions_.erase(it); clients_[client].pop_front(); --counts_.live;
        check_invariants(); return result;
    }
    void check_invariants() const {
        std::size_t rx=0,tx=0,total=0;
        std::vector<std::size_t> ids(c_.ids),queued(c_.clients);
        for(const auto& t:transactions_) {
            (t.write?tx:rx)+=t.beats; ++ids.at(t.id); ++queued.at(t.client); ++total;
            if(t.received>t.beats || t.sent>t.beats || t.data.size()!=t.length)
                throw std::logic_error("transaction accounting");
        }
        if(total!=counts_.live || total>c_.transactions || rx!=counts_.rx_reserved
            || tx!=counts_.tx_reserved || rx>c_.rx_beats || tx>c_.tx_beats)
            throw std::logic_error("AXI capacity invariant");
        for(auto n:ids) if(n>c_.per_id) throw std::logic_error("per-ID capacity");
        for(std::size_t c=0;c<queued.size();++c)
            if(queued[c]!=clients_[c].size()) throw std::logic_error("completion ownership");
    }
private:
    struct Transaction {
        Ticket ticket{};
        std::size_t client{},id{},length{},beats{},sent{},received{};
        std::uint64_t address{},generation{};
        bool write{},address_sent{},done{},error{},aborted{};
        std::vector<std::uint8_t> data{};
    };
    const Transaction* front(std::size_t client) const {
        if(clients_.at(client).empty()) return nullptr;
        const auto ticket=clients_[client].front();
        for(const auto& t:transactions_) if(t.ticket==ticket) return &t;
        throw std::logic_error("missing completion owner");
    }
    std::optional<Ticket> submit(std::size_t client,std::uint64_t address,
        std::size_t bytes,std::uint64_t generation,bool write,
        const std::vector<std::uint8_t>& data)
    {
        if(client>=c_.clients || !generation) throw std::invalid_argument("client/generation");
        // Admission validates a single burst without constructing the host-side
        // plan. A malformed huge descriptor must not allocate an unbounded list.
        if(!bytes || address%c_.beat_bytes || address>=c_.address_limit
            || bytes>c_.address_limit-address
            || bytes>c_.max_burst_beats*c_.beat_bytes || bytes>4096-address%4096)
            throw std::invalid_argument("submit exactly one aligned, in-range INCR burst");
        const auto beats=(bytes+c_.beat_bytes-1)/c_.beat_bytes;
        if(isolated_ || stopped_[client] || counts_.live==c_.transactions
            || beats>(write?c_.tx_beats-counts_.tx_reserved:c_.rx_beats-counts_.rx_reserved))
            return std::nullopt;
        std::optional<std::size_t> id;
        for(std::size_t n=0;n<c_.ids;++n) {
            const auto candidate=(next_id_+n)%c_.ids;
            const auto count=std::count_if(transactions_.begin(),transactions_.end(),
                [&](const auto& t){return t.id==candidate;});
            if(static_cast<std::size_t>(count)<c_.per_id) {id=candidate;break;}
        }
        if(!id) return std::nullopt;
        if(next_ticket_==std::numeric_limits<Ticket>::max())
            throw std::overflow_error("ticket generation exhausted");
        Transaction t; t.ticket=next_ticket_++;t.client=client;t.id=*id;
        t.length=bytes;t.beats=beats;t.address=address;t.generation=generation;t.write=write;
        t.data=write?data:std::vector<std::uint8_t>(bytes);
        const auto ticket=t.ticket;
        transactions_.push_back(std::move(t));clients_[client].push_back(ticket);
        (write?counts_.tx_reserved:counts_.rx_reserved)+=beats; ++counts_.live;
        counts_.peak_live=std::max(counts_.peak_live,counts_.live);
        counts_.peak_rx=std::max(counts_.peak_rx,counts_.rx_reserved);
        counts_.peak_tx=std::max(counts_.peak_tx,counts_.tx_reserved);
        next_id_=(*id+1)%c_.ids;check_invariants();return ticket;
    }
    AxiDmaConfig c_;
    std::deque<Transaction> transactions_;
    std::vector<std::deque<Ticket>> clients_;
    std::vector<bool> stopped_;
    Ticket next_ticket_{1};
    std::size_t next_id_{};
    Counters counts_{};
    bool isolated_{};
};

} // namespace ftlpu

