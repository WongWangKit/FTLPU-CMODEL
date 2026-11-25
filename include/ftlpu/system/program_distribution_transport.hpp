#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/core/timing.hpp"
#include "ftlpu/system/program_load_command_controller.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>

namespace ftlpu {

struct ProgramDistributionRecord {
    bool control_prime{false};
    std::array<std::uint8_t,hw::kProgramWordBytes> data{};
    std::uint8_t strb{0};
    ProgramLoadScope scope{ProgramLoadScope::East};
    std::uint16_t destination{0};
    std::size_t imem_address{0};
    bool image_last{false};

    bool operator==(const ProgramDistributionRecord&) const = default;
};

struct ProgramDistributionChannelView {
    bool valid{false};
    bool ready{false};
    bool fire{false};
    ProgramDistributionRecord payload{};
    bool payload_known{false};
    bool error{false};
};

struct ProgramDistributionInputs {
    bool reset{false};
    bool load_record_valid{false};
    ProgramDistributionRecord load_record{};
    std::array<std::array<bool,hw::config::kProgramLoadRelaysPerHemisphere>,
        hw::kHemispheres> mem_endpoint_ready{};
    std::array<bool,hw::kHemispheres> outer_endpoint_ready{};
    bool vxm_endpoint_ready{false};
    std::function<bool(const ProgramDistributionRecord&)> endpoint_ready{};
};

struct ProgramDistributionHardwareInterfaceView {
    ProgramDistributionChannelView load_record_input{};
    ProgramDistributionChannelView elastic_output{};
    std::size_t elastic_level{0};
    std::array<ProgramDistributionChannelView,hw::kHemispheres>
        distributor_route_output{};
    std::array<bool,hw::kHemispheres> distributor_route_required{};
    std::array<std::array<ProgramDistributionChannelView,
        hw::config::kProgramLoadRelaysPerHemisphere>,hw::kHemispheres>
        relay_input{};
    std::array<std::array<ProgramDistributionChannelView,
        hw::config::kProgramLoadRelaysPerHemisphere>,hw::kHemispheres>
        relay_forward_output{};
    std::array<std::array<ProgramDistributionChannelView,
        hw::config::kProgramLoadRelaysPerHemisphere>,hw::kHemispheres>
        mem_endpoint_input{};
    std::array<ProgramDistributionChannelView,hw::kHemispheres>
        outer_endpoint_input{};
    ProgramDistributionChannelView vxm_endpoint_input{};
    std::array<std::array<std::size_t,
        hw::config::kProgramLoadRelaysPerHemisphere>,hw::kHemispheres>
        relay_level{};
    bool west_done{false};
    bool east_done{false};
    bool idle{true};
    bool fault{false};
    std::uint8_t fault_code{0};
};

// Hierarchical clock participant for lpu_program_distribute_system.sv up to,
// but not including, Program Endpoint state updates.  ProgramLoader calls LOW
// in its ordered SystemPhase slot and commits this child during the same global
// POSEDGE as its own state.
class ProgramDistributionTransport final : public timing::ClockedModule {
public:
    static constexpr std::size_t kHemispheres=hw::kHemispheres;
    static constexpr std::size_t kRelays=
        hw::config::kProgramLoadRelaysPerHemisphere;
    static constexpr std::size_t kElasticDepth=
        hw::config::kProgramLoadIngressFifoDepth;
    static constexpr std::size_t kRelayDepth=
        hw::config::kProgramLoadRelayBufferDepth;
    static constexpr std::uint8_t kRouteFault=8;

    void drive(ProgramDistributionInputs inputs)
    {
        if(next_) throw std::logic_error(
            "Program Distribution inputs arrived after LOW");
        inputs_=std::move(inputs);
        driven_=true;
    }

    void low() override
    {
        if(next_) throw std::logic_error(
            "Program Distribution LOW executed more than once");
        if(!driven_) throw std::logic_error(
            "Program Distribution LOW requires driven inputs");
        if(inputs_.reset) {
            next_=State{};
            view_={};
            view_.load_record_input.ready=true;
            view_.idle=true;
            for(std::size_t h=0;h<kHemispheres;++h) {
                view_.distributor_route_output[h].ready=true;
                for(std::size_t r=0;r<kRelays;++r) {
                    view_.relay_input[h][r].ready=true;
                    view_.relay_forward_output[h][r].ready=r+1<kRelays;
                }
            }
            return;
        }

        next_=state_;
        view_={};
        build_relay_views();
        build_distributor_view();
        apply_relay_transfers();
        apply_distributor_transfers();
        view_.idle=state_.elastic.empty();
        for(const auto& hemisphere:state_.relays)
            for(const auto& relay:hemisphere)
                view_.idle=view_.idle&&relay.empty();
        view_.idle=view_.idle&&!state_.west_done&&!state_.east_done;
        view_.fault=state_.fault||next_->fault;
        view_.fault_code=view_.fault?kRouteFault:0;
    }

    void validate_posedge() const override
    {
        if(!next_) throw std::logic_error(
            "Program Distribution posedge requires one complete LOW");
    }

    void posedge() override
    {
        validate_posedge();
        state_=std::move(*next_);
        next_.reset();
        driven_=false;
    }

    void high() override {}
    void reset()
    {
        if(next_) throw std::logic_error(
            "Program Distribution reset cannot interrupt LOW");
        state_={};
        inputs_={};
        view_={};
        driven_=false;
    }
    bool prepared() const noexcept {return next_.has_value();}
    bool idle() const noexcept
    {
        if(!state_.elastic.empty()||state_.west_done||state_.east_done)
            return false;
        for(const auto& hemisphere:state_.relays)
            for(const auto& relay:hemisphere)
                if(!relay.empty()) return false;
        return true;
    }
    bool fault() const noexcept {return state_.fault||view_.fault;}
    const ProgramDistributionHardwareInterfaceView&
    hardware_interface_view() const noexcept {return view_;}

private:
    struct RelayBuffer {
        std::array<ProgramDistributionRecord,kRelayDepth> memory{};
        std::array<bool,kRelayDepth> written{};
        std::size_t read_ptr{0};
        std::size_t write_ptr{0};
        std::size_t level{0};

        bool empty() const noexcept {return level==0;}
        std::size_t size() const noexcept {return level;}
        bool head_written() const noexcept {return written[read_ptr];}
        const ProgramDistributionRecord& front() const noexcept
        {return memory[read_ptr];}
        void pop_front()
        {
            if(empty()) throw std::logic_error("empty Program Relay dequeue");
            read_ptr=(read_ptr+1)%kRelayDepth;
            --level;
        }
        void push_back(const ProgramDistributionRecord& record)
        {
            if(level==kRelayDepth)
                throw std::logic_error("full Program Relay enqueue");
            memory[write_ptr]=record;
            written[write_ptr]=true;
            write_ptr=(write_ptr+1)%kRelayDepth;
            ++level;
        }
    };

    struct State {
        std::deque<ProgramDistributionRecord> elastic{};
        std::array<std::array<RelayBuffer,kRelays>,kHemispheres> relays{};
        bool west_done{false};
        bool east_done{false};
        bool fault{false};
    };

    static std::size_t destination_hemisphere(
        const ProgramDistributionRecord& record) noexcept
    {return (record.destination>>12)&1U;}
    static std::size_t destination_kind(
        const ProgramDistributionRecord& record) noexcept
    {return (record.destination>>8)&15U;}
    static std::size_t destination_index(
        const ProgramDistributionRecord& record) noexcept
    {return record.destination&0xffU;}
    static bool peripheral_kind(std::size_t kind) noexcept
    {return (kind>=1&&kind<=3)||kind==5||kind==6||kind==7||kind==8;}
    static bool outer_legal(const ProgramDistributionRecord& record) noexcept
    {
        const auto kind=destination_kind(record);
        const auto index=destination_index(record);
        return (((kind>=1&&kind<=3)||kind==7||kind==8)&&index<2)
            ||((kind==5||kind==6)&&index<1);
    }
    static ProgramDistributionRecord branch_record(
        ProgramDistributionRecord record,std::size_t hemisphere)
    {
        record.destination=static_cast<std::uint16_t>(
            (record.destination&0x0fffU)|(hemisphere<<12));
        record.scope=hemisphere==0?ProgramLoadScope::East:
            ProgramLoadScope::West;
        return record;
    }
    static bool scope_legal(ProgramLoadScope scope) noexcept
    {
        return scope==ProgramLoadScope::East||scope==ProgramLoadScope::West
            ||scope==ProgramLoadScope::Both;
    }
    static bool relay_route_illegal(const ProgramDistributionRecord& record,
        std::size_t hemisphere,std::size_t relay) noexcept
    {
        const auto kind=destination_kind(record);
        const auto index=destination_index(record);
        const auto base=relay*hw::kProgramLoadMemBanksPerRelay;
        const auto end=std::min(base+hw::kProgramLoadMemBanksPerRelay,
            hw::kMemSliceColumns*hw::kMemBanksPerSlice);
        const bool local=kind==0&&index>=base&&index<end;
        const bool forwardMem=kind==0&&index>=end
            &&index<hw::kMemSliceColumns*hw::kMemBanksPerSlice;
        const bool forwardPeripheral=peripheral_kind(kind);
        const bool scopeMismatch=(record.scope==ProgramLoadScope::East
                &&hemisphere!=0)||(record.scope==ProgramLoadScope::West
                &&hemisphere!=1);
        return !scope_legal(record.scope)||scopeMismatch
            ||destination_hemisphere(record)!=hemisphere
            ||(!local&&!forwardMem&&!forwardPeripheral);
    }

    void build_relay_views()
    {
        for(std::size_t h=0;h<kHemispheres;++h) {
            for(std::size_t r=0;r<kRelays;++r) {
                auto& input=view_.relay_input[h][r];
                auto& forward=view_.relay_forward_output[h][r];
                auto& endpoint=view_.mem_endpoint_input[h][r];
                const auto& queue=state_.relays[h][r];
                view_.relay_level[h][r]=queue.size();
                bool illegal=false;
                if(!queue.empty()) illegal=relay_route_illegal(queue.front(),h,r);
                input.ready=!state_.fault&&!illegal&&queue.size()<kRelayDepth;
                // ready is a continuously observable input to each output
                // channel.  It must not collapse to zero merely because that
                // channel's valid is currently low.
                endpoint.ready=false;
                forward.ready=false;
                if(r+1==kRelays) view_.outer_endpoint_input[h].ready=false;
                // lpu_program_distribute_relay retains record_mem after a
                // dequeue.  Even at level zero the retained head payload feeds
                // the endpoint ready mux.  Model that physical memory state,
                // while keeping valid strictly occupancy based.
                if(queue.head_written()) {
                    const auto& retained=queue.front();
                    const auto retainedKind=destination_kind(retained);
                    const auto retainedIndex=destination_index(retained);
                    const auto retainedBase=
                        r*hw::kProgramLoadMemBanksPerRelay;
                    const auto retainedEnd=std::min(retainedBase
                            +hw::kProgramLoadMemBanksPerRelay,
                        hw::kMemSliceColumns*hw::kMemBanksPerSlice);
                    if(retainedKind==0&&retainedIndex>=retainedBase
                            &&retainedIndex<retainedEnd) {
                        endpoint.payload=retained;
                        endpoint.payload_known=true;
                        endpoint.ready=inputs_.endpoint_ready
                            ?inputs_.endpoint_ready(retained)
                            :inputs_.mem_endpoint_ready[h][r];
                    }
                    if(r+1==kRelays) {
                        const bool legal=outer_legal(retained);
                        const bool ready=legal && (inputs_.endpoint_ready
                            ?inputs_.endpoint_ready(retained)
                            :inputs_.outer_endpoint_ready[h]);
                        forward.ready=!legal||ready;
                        if(legal) {
                            view_.outer_endpoint_input[h].payload=retained;
                            view_.outer_endpoint_input[h].payload_known=true;
                            view_.outer_endpoint_input[h].ready=ready;
                        }
                    }
                }
                if(queue.empty()) continue;
                const auto& record=queue.front();
                const auto kind=destination_kind(record);
                const auto index=destination_index(record);
                const auto base=r*hw::kProgramLoadMemBanksPerRelay;
                const auto end=std::min(base+hw::kProgramLoadMemBanksPerRelay,
                    hw::kMemSliceColumns*hw::kMemBanksPerSlice);
                const bool local=kind==0&&index>=base&&index<end;
                const bool forwardMem=kind==0&&index>=end
                    &&index<hw::kMemSliceColumns*hw::kMemBanksPerSlice;
                const bool forwardPeripheral=peripheral_kind(kind);
                if(local&&!illegal&&!state_.fault) {
                    endpoint.valid=true;
                    endpoint.fire=endpoint.valid&&endpoint.ready;
                    endpoint.payload=record;
                } else if((forwardMem||forwardPeripheral)&&!illegal&&!state_.fault) {
                    forward.valid=true;
                    forward.payload=record;
                    forward.payload_known=true;
                }
                if(r+1==kRelays&&forward.valid) {
                    if(!outer_legal(record)) {
                        forward.ready=true;
                        next_->fault=true;
                    }
                    forward.fire=forward.valid&&forward.ready;
                    auto& outer=view_.outer_endpoint_input[h];
                    outer=forward;
                }
                if(illegal) next_->fault=true;
            }
        }
        // Downstream Relay ready is a registered-occupancy decision.  Wire the
        // combinational forwarding channel only after every input Ready exists.
        for(std::size_t h=0;h<kHemispheres;++h)
            for(std::size_t r=0;r+1<kRelays;++r) {
                auto& forward=view_.relay_forward_output[h][r];
                forward.ready=view_.relay_input[h][r+1].ready;
                forward.fire=forward.valid&&forward.ready;
            }
    }

    void build_distributor_view()
    {
        auto& input=view_.load_record_input;
        input.valid=inputs_.load_record_valid;
        input.payload=inputs_.load_record;
        for(std::size_t h=0;h<kHemispheres;++h)
            view_.distributor_route_output[h].ready=
                view_.relay_input[h][0].ready;
        view_.elastic_level=state_.elastic.size();
        view_.west_done=state_.west_done;
        view_.east_done=state_.east_done;

        bool headReady=false;
        if(!state_.elastic.empty()) {
            const auto head=state_.elastic.front();
            auto& elastic=view_.elastic_output;
            elastic.valid=true;
            elastic.payload=head;
            const auto kind=destination_kind(head);
            const bool vxm=kind==4;
            const bool scopeIllegal=!scope_legal(head.scope);
            const bool scopeMismatch=!vxm&&(
                (head.scope==ProgramLoadScope::East
                    &&destination_hemisphere(head)!=0)
                ||(head.scope==ProgramLoadScope::West
                    &&destination_hemisphere(head)!=1));
            const bool illegal=scopeIllegal||scopeMismatch;
            const bool required[2]={vxm||head.scope==ProgramLoadScope::East
                    ||head.scope==ProgramLoadScope::Both,
                !vxm&&(head.scope==ProgramLoadScope::West
                    ||head.scope==ProgramLoadScope::Both)};
            view_.distributor_route_required[0]=required[0];
            view_.distributor_route_required[1]=required[1];
            bool complete[2]={!required[0]||state_.east_done,
                !required[1]||state_.west_done};
            for(std::size_t h=0;h<kHemispheres;++h) {
                auto& route=view_.distributor_route_output[h];
                const bool done=h==0?state_.east_done:state_.west_done;
                route.valid=elastic.valid&&required[h]&&!done&&!state_.fault;
                if(vxm&&h==0) route.ready=inputs_.vxm_endpoint_ready;
                route.fire=route.valid&&route.ready;
                route.payload=branch_record(head,h);
                if(route.fire) complete[h]=true;
                if(vxm&&h==0) {
                    view_.vxm_endpoint_input=route;
                    view_.vxm_endpoint_input.ready=inputs_.vxm_endpoint_ready;
                    view_.vxm_endpoint_input.fire=
                        view_.vxm_endpoint_input.valid
                        &&view_.vxm_endpoint_input.ready;
                }
            }
            if(illegal) next_->fault=true;
            headReady=elastic.valid&&(illegal||(complete[0]&&complete[1]
                &&!state_.fault));
            elastic.ready=headReady;
            elastic.fire=headReady;
        }
        input.ready=state_.elastic.size()<kElasticDepth||headReady;
        input.fire=input.valid&&input.ready;

        for(std::size_t h=0;h<kHemispheres;++h) {
            auto& relayInput=view_.relay_input[h][0];
            const auto& route=view_.distributor_route_output[h];
            relayInput.valid=route.valid&&destination_kind(route.payload)!=4;
            relayInput.fire=relayInput.valid&&relayInput.ready;
            if(relayInput.valid) relayInput.payload=route.payload;
        }
        for(std::size_t h=0;h<kHemispheres;++h)
            for(std::size_t r=1;r<kRelays;++r) {
                auto& inputView=view_.relay_input[h][r];
                const auto& upstream=view_.relay_forward_output[h][r-1];
                inputView.valid=upstream.valid;
                inputView.fire=upstream.fire;
                if(inputView.valid) inputView.payload=upstream.payload;
            }
    }

    void apply_relay_transfers()
    {
        for(std::size_t h=0;h<kHemispheres;++h)
            for(std::size_t r=0;r<kRelays;++r) {
                const bool dequeue=view_.mem_endpoint_input[h][r].fire
                    ||view_.relay_forward_output[h][r].fire
                    ||(!state_.relays[h][r].empty()
                        &&relay_route_illegal(state_.relays[h][r].front(),h,r));
                const auto& input=view_.relay_input[h][r];
                if(dequeue) next_->relays[h][r].pop_front();
                if(input.fire) next_->relays[h][r].push_back(input.payload);
            }
    }

    void apply_distributor_transfers()
    {
        if(view_.elastic_output.fire) next_->elastic.pop_front();
        if(view_.load_record_input.fire)
            next_->elastic.push_back(view_.load_record_input.payload);
        if(view_.elastic_output.fire) {
            next_->west_done=false;
            next_->east_done=false;
        } else {
            if(view_.distributor_route_output[0].fire) next_->east_done=true;
            if(view_.distributor_route_output[1].fire) next_->west_done=true;
        }
    }

    State state_{};
    std::optional<State> next_{};
    ProgramDistributionInputs inputs_{};
    ProgramDistributionHardwareInterfaceView view_{};
    bool driven_{false};
};

} // namespace ftlpu
