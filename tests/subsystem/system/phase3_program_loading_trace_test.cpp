#include "ftlpu/system/c2c_dma_system.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace ftlpu;

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

Ddr4Config ddr_config()
{
    Ddr4Config config{};
    config.read_latency_cycles=2;
    config.read_latency_jitter_cycles=0;
    config.peak_bandwidth_bytes_per_second=config.lpu_clock_hz*32;
    return config;
}

std::vector<std::uint8_t> mem_macro_bytes()
{
    const std::vector<std::uint32_t> lanes{
        0x25,0,0,0x02000021,0x180,0x00303200,0x400,0,0};
    std::vector<std::uint8_t> bytes(36);
    for(std::size_t b=0;b<bytes.size();++b)
        bytes[b]=static_cast<std::uint8_t>(lanes[b/4]>>(8*(b%4)));
    return bytes;
}

void stage(C2cDmaSystem& system,std::uint64_t address,
    const std::vector<std::uint8_t>& bytes)
{
    for(std::size_t offset=0;offset<bytes.size();offset+=32) {
        C2cVector vector{};
        for(std::size_t b=0;b<32;++b)
            vector.payload[b/hw::kLanesPerTile][b%hw::kLanesPerTile]=
                offset+b<bytes.size()?bytes[offset+b]:0xa5;
        system.ddr4().initialize_vector(address+offset,vector);
    }
}

unsigned bit(bool value) { return value?1u:0u; }

template<std::size_t N>
std::string hex_bytes(const std::array<std::uint8_t,N>& bytes)
{
    std::ostringstream stream;
    stream<<std::hex<<std::setfill('0');
    for(const auto byte:bytes)
        stream<<std::setw(2)<<static_cast<unsigned>(byte);
    return stream.str();
}

void channel_header(std::ostream& os,const std::string& prefix)
{
    os<<','<<prefix<<"_valid,"<<prefix<<"_ready,"<<prefix<<"_fire,"
      <<prefix<<"_control,"<<prefix<<"_data,"<<prefix<<"_strb,"
      <<prefix<<"_scope,"<<prefix<<"_destination,"
      <<prefix<<"_address,"<<prefix<<"_last";
}

void channel_row(std::ostream& os,
    const ProgramDistributionChannelView& channel)
{
    os<<','<<bit(channel.valid)<<','<<bit(channel.ready)
      <<','<<bit(channel.fire)<<','<<bit(channel.payload.control_prime)
      <<','<<hex_bytes(channel.payload.data)
      <<','<<static_cast<unsigned>(channel.payload.strb)
      <<','<<static_cast<unsigned>(channel.payload.scope)
      <<','<<channel.payload.destination<<','<<channel.payload.imem_address
      <<','<<bit(channel.payload.image_last);
}

void r3_header(std::ostream& os)
{
    channel_header(os,"r3_distributor_in");
    channel_header(os,"r3_elastic_out");
    os<<",r3_elastic_level,r3_east_done,r3_west_done";
    channel_header(os,"r3_east_route");
    channel_header(os,"r3_west_route");
    constexpr const char* hemispheres[]={"east","west"};
    for(std::size_t h=0;h<2;++h)
        for(std::size_t relay=0;
            relay<ProgramDistributionTransport::kRelays;++relay) {
            const auto base=std::string("r3_")+hemispheres[h]+"_relay"
                +std::to_string(relay);
            channel_header(os,base+"_in");
            channel_header(os,base+"_forward");
            channel_header(os,base+"_local");
            os<<','<<base<<"_level";
        }
    channel_header(os,"r3_east_outer");
    channel_header(os,"r3_west_outer");
    channel_header(os,"r3_vxm");
    os<<",r3_idle,r3_fault,r3_fault_code";
}

void r3_row(std::ostream& os,
    const ProgramDistributionHardwareInterfaceView& view)
{
    channel_row(os,view.load_record_input);
    channel_row(os,view.elastic_output);
    os<<','<<view.elastic_level<<','<<bit(view.east_done)
      <<','<<bit(view.west_done);
    channel_row(os,view.distributor_route_output[0]);
    channel_row(os,view.distributor_route_output[1]);
    for(std::size_t h=0;h<2;++h)
        for(std::size_t relay=0;
            relay<ProgramDistributionTransport::kRelays;++relay) {
            channel_row(os,view.relay_input[h][relay]);
            channel_row(os,view.relay_forward_output[h][relay]);
            channel_row(os,view.mem_endpoint_input[h][relay]);
            os<<','<<view.relay_level[h][relay];
        }
    channel_row(os,view.outer_endpoint_input[0]);
    channel_row(os,view.outer_endpoint_input[1]);
    channel_row(os,view.vxm_endpoint_input);
    os<<','<<bit(view.idle)<<','<<bit(view.fault)
      <<','<<static_cast<unsigned>(view.fault_code);
}

void descriptor(std::ostream& os,
    const std::optional<ProgramLoadConfigDescriptor>& value)
{
    if(!value) {
        os<<",0,0,0,0,0,0,0,0,0,0";
        return;
    }
    os<<','<<value->source_address<<','<<value->byte_count
      <<','<<static_cast<unsigned>(value->scope)
      <<','<<value->destination
      <<','<<static_cast<unsigned>(value->format)
      <<','<<value->imem_base_address
      <<','<<bit(value->image_first)<<','<<bit(value->image_last)
      <<','<<value->expected_word_count
      <<','<<static_cast<unsigned>(value->dma_id);
}

} // namespace

int run_r4(const std::filesystem::path& output, bool mxm)
{
    std::filesystem::create_directories(output);
    const std::string target=mxm?"mxm0":"mem0";
    const auto trace_name="phase3_r4_"+target+"_trace.csv";
    std::ofstream trace(output/trace_name);
    require(static_cast<bool>(trace),"cannot open R4 trace");
    const auto prefix="r4_"+target;
    trace<<"cycle,phase";
    channel_header(trace,prefix+"_program");
    for(const auto* field:{"program_error","write_fire","prime_fire",
            "destination_legal","frontend_load_valid","frontend_load_ready",
            "frontend_load_fire","frontend_prime_valid","frontend_prime_ready",
            "frontend_prime_fire","imem_write_fire","imem_address",
            mxm?"imem_data128":"imem_data96","state_configured","state_last_seen",
            "state_base","state_next_address","state_word_count","state_prime_started",
            "state_fault_code","commit_valid"})
        trace<<','<<prefix<<'_'<<field;
    trace<<'\n';

    auto owner=std::make_unique<C2cDmaSystem>(ddr_config(),1);
    auto& system=*owner;
    auto& loader=system.program_loader();
    auto& icu=system.chip().icu();
    const auto state=[&] {
        return mxm?icu.mxm_load_iq(0).program_endpoint_state()
            :icu.mem_iq(0).program_endpoint_state();
    };
    auto bytes=mxm?std::vector<std::uint8_t>(32):mem_macro_bytes();
    if(mxm) {
        // CONTROL_ONE and MXM_LOAD_GOLDEN in lpu_mxm_macro_decoder_tb.sv.
        bytes[0]=0x17;
        bytes[16]=0x28;
        bytes[20]=0x08;
        bytes[23]=0x04;
    }
    const ProgramImage image{0x60000,
        mxm?IcuMacroQueueKind::MxmLoad:IcuMacroQueueKind::Mem,0,
        bytes.size(),IcuQueueMode::Macro};
    stage(system,image.source_address,bytes);
    loader.submit({image},system.cycle());
    if(!mxm) {
        loader.start();
        loader.set_dependencies_ready(true);
    }
    std::size_t rows=0,writes=0,primes=0;
    constexpr std::size_t mem_trace_rows=92;
    bool observed_prime_state=false;
    std::optional<ProgramDistributionRecord> held;
    while(mxm ? !observed_prime_state
              : (!loader.kernel_done() || rows < mem_trace_rows)) {
        require(rows<4000,"R4 endpoint test timed out");
        const auto cycle=system.cycle();
        const auto before=state();
        observed_prime_state=before.prime_started;
        system.tick();
        const auto after=state();
        const auto& view=loader.program_load_hardware_interface_view();
        const auto& transport=view.r3.distribution;
        const auto& port=mxm?transport.outer_endpoint_input[0]:
            transport.mem_endpoint_input[0][0];
        const auto expected_destination=mxm?0x100:0;
        const bool destination_legal=port.payload_known
            &&port.payload.destination==expected_destination;
        const bool write=port.fire&&!port.payload.control_prime;
        const bool prime=port.fire&&port.payload.control_prime;
        require(port.fire==(port.valid&&port.ready),"R4 handshake inconsistent");
        if(held) require(port.valid&&port.payload==*held,"R4 held payload changed");
        if(port.valid&&!port.ready) held=port.payload; else held.reset();
        require(after.word_count==before.word_count+write,"iMEM advanced without WRITE fire");
        require(after.prime_started==(before.prime_started||prime),
            "PRIME state diverged from Endpoint fire");
        require(!view.program_fault_o&&!after.fault_code,"R4 loading fault");
        if(write) {
            require(destination_legal&&port.payload.strb==(mxm?15:7),
                "invalid R4 write routing or strobe");
            const auto address=port.payload.imem_address;
            const auto actual=mxm?icu.mxm_load_iq(0).read_raw_imem(address).lanes[0]
                :icu.mem_iq(0).read_raw_imem(address).lanes[0];
            std::uint32_t expected=0;
            for(std::size_t byte=0;byte<4;++byte)
                expected|=std::uint32_t(bytes[address*image.word_bytes()+byte])<<(byte*8);
            require(actual==expected,"iMEM write not visible after accepting edge");
        }
        writes+=write;
        primes+=prime;
        trace<<cycle<<",PRE_POSEDGE";
        channel_row(trace,port);
        trace<<','<<bit(port.error)
            <<','<<bit(write)<<','<<bit(prime)<<','<<bit(destination_legal)
            <<','<<bit(write)<<','<<bit(before.ready())<<','<<bit(write)
            <<','<<bit(prime)<<','<<bit(before.ready())<<','<<bit(prime)
            <<','<<bit(write)<<','<<(write?port.payload.imem_address:0)<<',';
        auto effective=hex_bytes(port.payload.data);
        effective.resize(image.word_bytes()*2);
        trace<<(write?effective:std::string(image.word_bytes()*2,'0'))
            <<','<<bit(before.configured)<<','<<bit(before.last_seen)
            <<','<<before.base<<','<<before.next_address<<','<<before.word_count
            <<','<<bit(before.prime_started)<<','<<static_cast<unsigned>(before.fault_code)
            <<','<<bit(view.endpoint_write_commit_valid)<<'\n';
        ++rows;
    }
    require(!held&&writes==image.word_count()&&primes==1,"R4 transfers incomplete");
    require(icu.read_program_image(image)==bytes,"local iMEM bytes differ");
    std::size_t blocked_writes=0;
    if(mxm) {
        // Drive the already primed endpoint through the actual Transport in
        // a directed check. The production loader only submits legal images.
        ProgramDistributionTransport transport;
        ProgramDistributionInputs inputs{};
        inputs.endpoint_ready=[&](const auto&) {return state().ready();};
        inputs.load_record_valid=true;
        auto& payload=inputs.load_record;
        payload.destination=0x100;
        payload.strb=15;
        payload.data.fill(0xa5);
        for(std::size_t cycle=0;cycle<16&&blocked_writes<2;++cycle) {
            transport.drive(inputs);
            transport.low();
            const auto& view=transport.hardware_interface_view();
            if(view.load_record_input.fire) inputs.load_record_valid=false;
            const auto& endpoint=view.outer_endpoint_input[0];
            if(endpoint.valid) {
                require(!endpoint.ready&&!endpoint.fire&&endpoint.payload==payload,
                    "primed MXM accepted or changed a held WRITE");
                ++blocked_writes;
            }
            transport.posedge();
        }
        require(blocked_writes==2 && icu.read_program_image(image)==bytes,
            "MXM post-PRIME stall did not preserve local iMEM");
    } else {
        auto probe=std::make_unique<C2cDmaSystem>(ddr_config(),1);
        auto& queue=probe->chip().icu().mem_iq(0);
        queue.load_mem_program_word(0,{1,2,3,0xdeadbeefU},7,true);
        const auto word=queue.read_raw_imem(0);
        require(word.lanes.size()==3&&word.lanes[0]==1&&word.lanes[1]==2
                &&word.lanes[2]==3,"nonstrobed MEM lane entered local iMEM");
    }
    std::ofstream manifest(output/("phase3_r4_"+target+"_manifest.json"));
    require(static_cast<bool>(manifest),"cannot open R4 manifest");
    manifest<<"{\n  \"schema\": \"ftlpu.phase3.r4.manifest.v1\",\n"
        <<"  \"status\": \"PASS\",\n  \"phase\": \"PRE_POSEDGE\",\n"
        <<"  \"cycle_offset\": 0,\n  \"trace\": \""<<trace_name<<"\",\n"
        <<"  \"rows\": "<<rows<<",\n  \"write_fires\": "<<writes
        <<",\n  \"prime_fires\": "<<primes
        <<",\n  \"directed_post_prime_blocked_write_cycles\": "<<blocked_writes<<"\n}\n";
    return 0;
}

int run_r4_mxm_endpoint(const std::filesystem::path& output)
{
    // Same stimulus and observation window as lpu_program_endpoint_mxm_tb.
    constexpr std::size_t trace_rows=10;
    constexpr std::uint16_t destination=0x100;
    auto owner=std::make_unique<C2cDmaSystem>(ddr_config(),1);
    auto& icu=owner->chip().icu();
    const auto queue_state=[&] { return icu.mxm_load_iq(0).program_endpoint_state(); };
    icu.stage_program_loading(true);
    icu.stage_program_issue_enabled(false);
    const ProgramImage image{0,IcuMacroQueueKind::MxmLoad,0,32,IcuQueueMode::Macro};
    require(!queue_state().configured && !queue_state().prime_started,
        "MXM direct trace requires reset endpoint state");

    constexpr auto trace_name="phase3_r4_mxm_endpoint_direct_trace.csv";
    std::filesystem::create_directories(output);
    std::ofstream trace(output/trace_name);
    require(static_cast<bool>(trace),"cannot open MXM endpoint direct trace");
    trace<<"cycle,phase";
    for(const auto* field:{"endpoint_valid","endpoint_ready","endpoint_fire",
            "endpoint_control","endpoint_data","endpoint_strb","endpoint_destination",
            "endpoint_address","endpoint_last","endpoint_error","destination_legal",
            "write_fire","prime_fire","frontend_load_valid","frontend_load_ready",
            "frontend_load_fire","frontend_prime_valid","frontend_prime_ready",
            "frontend_prime_fire","imem_write_fire","imem_write_address","imem_write_data",
            "configured","last_seen","load_base","next_address","word_count",
            "prime_started","frontend_fault","frontend_fault_code"})
        trace<<",r4_mxm_"<<field;
    trace<<'\n';

    std::optional<ProgramDistributionRecord> held;
    std::size_t writes=0,primes=0,stalls=0;
    for(std::size_t cycle=0;cycle<trace_rows;++cycle) {
        ProgramDistributionRecord payload{};
        payload.destination=destination;
        const bool valid=cycle==1||cycle==2||cycle==3||cycle>=5;
        if(cycle==1||cycle==2||cycle>=5) payload.strb=15;
        if(cycle==1) payload.data[0]=0x17;
        if(cycle==2) {
            payload.imem_address=1;
            payload.data[0]=0x28;
            payload.data[4]=0x08;
            payload.data[7]=0x04;
            payload.image_last=true;
        }
        if(cycle==3) {
            payload.control_prime=true;
            payload.image_last=true;
        }
        if(cycle>=5) {
            payload.imem_address=2;
            payload.data[0]=0xed;
            payload.data[1]=0xfe;
            payload.image_last=true;
        }

        const auto before=queue_state();
        const bool ready=owner->program_loader().endpoint_ready(payload,true);
        const bool fire=valid&&ready;
        const bool legal=payload.destination==destination;
        const bool error=fire&&(!legal||(!payload.control_prime
            && (payload.strb!=15 || payload.imem_address>=icu.mxm_load_iq(0).imem_depth)));
        const bool write_fire=fire&&!payload.control_prime&&!error;
        const bool prime_fire=fire&&payload.control_prime&&!error;
        const bool load_ready=before.ready();
        const bool prime_ready=before.ready();
        const bool load_fire=write_fire&&load_ready;
        const bool local_prime_fire=prime_fire&&prime_ready;
        const bool imem_write=load_fire&&!before.last_seen
            && (!before.configured||payload.imem_address==before.next_address);
        if(held) require(valid&&payload==*held,"MXM direct held payload changed");
        if(valid&&!ready) {held=payload;++stalls;} else held.reset();
        require(!error&&(!load_fire||imem_write),"illegal MXM directed WRITE");

        // Use the production staged operations and common clock edge.
        if(imem_write) icu.stage_program_word(image,payload.imem_address,payload.data.data());
        if(local_prime_fire) icu.stage_program_seal(image);
        auto packed=payload.data;
        std::reverse(packed.begin(),packed.end());
        const auto packed_hex=hex_bytes(packed);
        trace<<cycle<<",PRE_POSEDGE,"<<bit(valid)<<','<<bit(ready)<<','<<bit(fire)
            <<','<<bit(payload.control_prime)<<','<<packed_hex<<','
            <<static_cast<unsigned>(payload.strb)<<','<<payload.destination<<','
            <<payload.imem_address<<','<<bit(payload.image_last)<<','<<bit(error)
            <<','<<bit(legal)<<','<<bit(write_fire)<<','<<bit(prime_fire)
            <<','<<bit(write_fire)<<','<<bit(load_ready)<<','<<bit(load_fire)
            <<','<<bit(prime_fire)<<','<<bit(prime_ready)<<','<<bit(local_prime_fire)
            <<','<<bit(imem_write)<<','<<payload.imem_address<<','<<packed_hex
            <<','<<bit(before.configured)<<','<<bit(before.last_seen)<<','<<before.base
            <<','<<before.next_address<<','<<before.word_count<<','<<bit(before.prime_started)
            <<','<<bit(before.fault_code!=0)<<','<<static_cast<unsigned>(before.fault_code)
            <<'\n';
        owner->tick();
        const auto after=queue_state();
        require(after.word_count==before.word_count+imem_write
            &&after.prime_started==(before.prime_started||local_prime_fire)
            &&!after.fault_code,"MXM direct commit differs from accepting handshake");
        writes+=imem_write;
        primes+=local_prime_fire;
    }
    require(writes==2&&primes==1&&stalls==5,"MXM direct coverage mismatch");
    const auto bytes=icu.read_program_image(image);
    std::vector<std::uint8_t> expected(32);
    expected[0]=0x17;expected[16]=0x28;expected[20]=0x08;expected[23]=0x04;
    require(bytes==expected,"MXM direct write/hold changed local iMEM image");
    std::ofstream manifest(output/"phase3_r4_mxm_endpoint_direct_manifest.json");
    require(static_cast<bool>(manifest),"cannot open MXM endpoint direct manifest");
    manifest<<"{\n  \"schema\": \"ftlpu.phase3.r4.mxm.endpoint_direct.v1\",\n"
        <<"  \"status\": \"PASS\",\n  \"phase\": \"PRE_POSEDGE\",\n"
        <<"  \"cycle_offset\": 0,\n  \"trace\": \""<<trace_name<<"\",\n"
        <<"  \"rows\": "<<trace_rows<<",\n  \"write_fires\": "<<writes
        <<",\n  \"prime_fires\": "<<primes<<",\n  \"blocked_write_cycles\": "<<stalls
        <<",\n  \"stimulus\": \"lpu_program_endpoint_mxm_tb: East MXM0 Load\"\n}\n";
    return 0;
}

int main(int argc,char** argv) try
{
    if(argc>2&&std::string(argv[2])=="r4_mxm_endpoint_direct")
        return run_r4_mxm_endpoint(argv[1]);
    if(argc>2&&(std::string(argv[2])=="r4_mem"||std::string(argv[2])=="r4_mxm"))
        return run_r4(argv[1],std::string(argv[2])=="r4_mxm");
    const bool r2_mode=argc>2&&std::string(argv[2])=="r2";
    const auto output_dir=argc>1?std::filesystem::path(argv[1])
        :std::filesystem::current_path()
            /(r2_mode?"phase3_r2_artifacts":"phase3_r3_artifacts");
    std::filesystem::create_directories(output_dir);
    const auto trace_name=r2_mode?"phase3_r2_program_loading_trace.csv":
        "phase3_r3_program_distribution_trace.csv";
    std::ofstream trace(output_dir/trace_name);
    if(!trace) throw std::runtime_error("cannot open Phase3 trace");
    trace<<"cycle,phase,cmd_write_valid,cmd_write_ready,cmd_write_fire,"
        "cmd_write_address,cmd_write_opcode,cmd_write_source,cmd_write_bytes,"
        "cmd_write_destination,cmd_write_format,cmd_write_encoding,"
        "cmd_write_imem_base,cmd_write_first,cmd_write_last,"
        "cmd_write_expected_words,cmd_write_jump,load_start_valid,"
        "load_start_ready,load_start_fire,load_start_count,load_txn_valid,"
        "load_txn_ready,load_txn_fire,load_txn_source,load_txn_bytes,"
        "load_txn_scope,load_txn_destination,load_txn_format,"
        "load_txn_imem_base,load_txn_first,load_txn_last,"
        "load_txn_expected_words,load_txn_dma_id,prime_valid,prime_ready,"
        "prime_fire,prime_scope,prime_destination,command_pc,command_busy,"
        "command_done,command_fault,command_fault_code,config_enq_valid,"
        "config_enq_ready,config_enq_fire,config_enq_source,config_enq_bytes,"
        "config_enq_scope,config_enq_destination,config_enq_format,"
        "config_enq_imem_base,config_enq_first,config_enq_last,"
        "config_enq_expected_words,config_enq_dma_id,config_deq_valid,"
        "config_deq_ready,config_deq_fire,config_deq_source,config_deq_bytes,"
        "config_deq_scope,config_deq_destination,config_deq_format,"
        "config_deq_imem_base,config_deq_first,config_deq_last,"
        "config_deq_expected_words,config_deq_dma_id,config_level,"
        "canonical_path,load_request_active,load_pipeline_idle,"
        "distribution_idle,load_complete,program_ready,program_running,"
        "program_done,program_fault,program_fault_code,dma_final_response,"
        "dma_response_id,ddr_rsp_valid,ddr_rsp_ready,ddr_rsp_fire,"
        "ddr_rsp_index,ddr_rsp_keep,ddr_rsp_last,ddr_rsp_id,ddr_rsp_error,"
        "ddr_rsp_data,dma_rsp_valid,dma_rsp_ready,dma_rsp_fire,dma_rsp_index,"
        "dma_rsp_keep,dma_rsp_last,dma_rsp_id,dma_rsp_error,dma_rsp_data,"
        "dma_response_fifo_level,reasm_word_valid,reasm_word_ready,"
        "reasm_word_fire,reasm_image_complete,reasm_word_address,"
        "reasm_word_strb,reasm_word_last,reasm_word_data,word_fifo_enq_valid,"
        "word_fifo_enq_ready,word_fifo_enq_fire,word_fifo_deq_valid,"
        "word_fifo_deq_ready,word_fifo_deq_fire,word_fifo_level,"
        "word_fifo_head_address,word_fifo_head_strb,word_fifo_head_last,"
        "word_fifo_head_data,mem_program_valid,mem_program_ready,mem_program_fire,"
        "mem_program_address,mem_program_strb,mem_program_last,"
        "mem_program_error,mem_frontend_ready,mem_fetched_words,"
        "mem_decoded_contexts,mem_issued_count,mem_issue_delta";
    if(!r2_mode) r3_header(trace);
    trace<<'\n';

    ProgramLoadConfig program_load{};
    C2cDmaSystem system(ddr_config(),1,SystemHardwareConfiguration{},
        program_load);
    auto& loader=system.program_loader();
    auto& queue=system.chip().icu().mem_iq(0);
    const auto bytes=mem_macro_bytes();
    const ProgramImage image{0x60000,IcuMacroQueueKind::Mem,0,
        bytes.size(),IcuQueueMode::Macro};
    stage(system,image.source_address,bytes);
    loader.submit({image},system.cycle());
    loader.start();
    loader.set_dependencies_ready(true);

    std::size_t rows=0,command_writes=0,load_starts=0,load_fires=0;
    std::size_t prime_fires=0,config_enqueues=0,config_dequeues=0;
    std::size_t mem_program_fires=0;
    std::size_t ddr_response_fires=0,dma_response_fires=0;
    std::size_t reassembler_word_fires=0,word_fifo_enqueues=0;
    std::size_t word_fifo_dequeues=0;
    std::optional<std::size_t> ddr_final_cycle{},dma_final_cycle{};
    std::optional<std::size_t> config_dequeue_cycle{};
    std::vector<std::size_t> ddr_beat_indices{},dma_beat_indices{};
    std::size_t distributor_input_fires=0,elastic_output_fires=0;
    std::size_t distributor_write_fires=0,distributor_prime_fires=0;
    std::size_t east_route_fires=0,west_route_fires=0;
    std::size_t east_relay0_input_fires=0,east_relay0_local_fires=0;
    std::size_t endpoint_stall_cycles=0;
    std::optional<ProgramDistributionRecord> held_endpoint{};
    std::optional<ProgramLoadCommand> held_command{};
    std::optional<ProgramLoadConfigDescriptor> held_config{};
    std::size_t previous_issued=queue.issued_count();
    std::size_t mem_fetched_words=0;
    constexpr std::size_t r3_trace_rows=92;
    while(!loader.kernel_done() || (!r2_mode && rows<r3_trace_rows)) {
        require(rows<4000,"Phase3 production scenario timed out");
        const auto cycle=system.cycle();
        const bool mem_frontend_ready_before_edge=
            queue.local_frontend_ready(loader.started());
        system.tick();
        const auto& view=loader.program_load_hardware_interface_view();
        const auto& command=view.command;
        const auto& fifo=view.config_fifo;
        const auto& r2=view.r2;
        const auto& r3=view.r3.distribution;
        const auto& ddr_rsp=r2.dma.ddr_response_input;
        const auto& dma_rsp=r2.dma.reassembler_response_output;
        const auto& mem=loader.mem_program_port_signals();
        const auto& decoder=queue.macro_decoder_statistics();
        const auto& imem_read=queue.imem_read_port_signals();
        mem_fetched_words+=imem_read.imem_read_rsp_fire;
        const auto issued=queue.issued_count();
        const auto issue_delta=issued-previous_issued;
        previous_issued=issued;

        if(held_command) {
            require(command.ctrl_load_txn_valid_o
                    && command.active_command==*held_command,
                "LOAD payload changed while ready was low");
            if(command.ctrl_load_txn_ready_i) held_command.reset();
        }
        if(command.ctrl_load_txn_valid_o&&!command.ctrl_load_txn_ready_i)
            held_command=command.active_command;
        if(held_config) {
            require(fifo.dequeue_valid_o
                    && fifo.dequeue_data_o==held_config,
                "Config FIFO head changed while ready was low");
            if(fifo.dequeue_ready_i) held_config.reset();
        }
        if(fifo.dequeue_valid_o&&!fifo.dequeue_ready_i)
            held_config=fifo.dequeue_data_o;

        command_writes+=command.program_cmd_write_fire;
        load_starts+=command.program_load_start_fire;
        load_fires+=command.ctrl_load_txn_fire;
        prime_fires+=command.ctrl_prime_fire;
        config_enqueues+=fifo.enqueue_fire;
        config_dequeues+=fifo.dequeue_fire;
        mem_program_fires+=mem.mem_program_fire;
        ddr_response_fires+=ddr_rsp.fire;
        dma_response_fires+=dma_rsp.fire;
        reassembler_word_fires+=r2.reassembler_word_fire;
        word_fifo_enqueues+=r2.word_fifo_enqueue_fire;
        word_fifo_dequeues+=r2.word_fifo_dequeue_fire;
        if(ddr_rsp.fire) {
            ddr_beat_indices.push_back(ddr_rsp.beat_index);
            if(ddr_rsp.last) ddr_final_cycle=cycle;
        }
        if(dma_rsp.fire) {
            dma_beat_indices.push_back(dma_rsp.beat_index);
            if(dma_rsp.last) dma_final_cycle=cycle;
        }
        if(fifo.dequeue_fire) config_dequeue_cycle=cycle;
        distributor_input_fires+=r3.load_record_input.fire;
        if(r3.load_record_input.fire) {
            distributor_prime_fires+=r3.load_record_input.payload.control_prime;
            distributor_write_fires+=!r3.load_record_input.payload.control_prime;
        }
        elastic_output_fires+=r3.elastic_output.fire;
        east_route_fires+=r3.distributor_route_output[0].fire;
        west_route_fires+=r3.distributor_route_output[1].fire;
        east_relay0_input_fires+=r3.relay_input[0][0].fire;
        east_relay0_local_fires+=r3.mem_endpoint_input[0][0].fire;
        const auto& endpoint=r3.mem_endpoint_input[0][0];
        if(held_endpoint) {
            require(endpoint.valid&&endpoint.payload==*held_endpoint,
                "Program Endpoint payload changed during backpressure");
            if(endpoint.ready) held_endpoint.reset();
        }
        if(endpoint.valid&&!endpoint.ready) {
            held_endpoint=endpoint.payload;
            ++endpoint_stall_cycles;
        }

        trace<<cycle<<",PRE_POSEDGE,"<<bit(command.program_cmd_write_valid_i)
            <<','<<bit(command.program_cmd_write_ready_o)
            <<','<<bit(command.program_cmd_write_fire)
            <<','<<static_cast<unsigned>(command.program_cmd_write_address_i)
            <<','<<static_cast<unsigned>(command.program_cmd_write_opcode_i)
            <<','<<command.program_cmd_write_source_address_i
            <<','<<command.program_cmd_write_byte_count_i
            <<','<<command.program_cmd_write_destination_i
            <<','<<static_cast<unsigned>(command.program_cmd_write_format_i)
            <<','<<static_cast<unsigned>(command.program_cmd_write_encoding_mode_i)
            <<','<<command.program_cmd_write_imem_base_address_i
            <<','<<bit(command.program_cmd_write_image_first_i)
            <<','<<bit(command.program_cmd_write_image_last_i)
            <<','<<command.program_cmd_write_expected_word_count_i
            <<','<<static_cast<unsigned>(command.program_cmd_write_jump_address_i)
            <<','<<bit(command.program_load_start_valid_i)
            <<','<<bit(command.program_load_start_ready_o)
            <<','<<bit(command.program_load_start_fire)
            <<','<<command.program_load_start_command_count_i
            <<','<<bit(command.ctrl_load_txn_valid_o)
            <<','<<bit(command.ctrl_load_txn_ready_i)
            <<','<<bit(command.ctrl_load_txn_fire)
            <<','<<command.ctrl_load_txn_source_address_o
            <<','<<command.ctrl_load_txn_byte_count_o
            <<','<<static_cast<unsigned>(command.ctrl_load_txn_scope_o)
            <<','<<command.ctrl_load_txn_destination_o
            <<','<<static_cast<unsigned>(command.ctrl_load_txn_format_o)
            <<','<<command.ctrl_load_txn_imem_base_address_o
            <<','<<bit(command.ctrl_load_txn_image_first_o)
            <<','<<bit(command.ctrl_load_txn_image_last_o)
            <<','<<command.ctrl_load_txn_expected_word_count_o
            <<','<<static_cast<unsigned>(command.ctrl_load_txn_dma_id_o)
            <<','<<bit(command.ctrl_prime_valid_o)
            <<','<<bit(command.ctrl_prime_ready_i)
            <<','<<bit(command.ctrl_prime_fire)
            <<','<<static_cast<unsigned>(command.ctrl_prime_scope_o)
            <<','<<command.ctrl_prime_destination_o
            <<','<<static_cast<unsigned>(command.command_pc_o)
            <<','<<bit(command.busy_o)<<','<<bit(command.done_o)
            <<','<<bit(command.fault_o)
            <<','<<static_cast<unsigned>(command.fault_code_o)
            <<','<<bit(fifo.enqueue_valid_i)<<','<<bit(fifo.enqueue_ready_o)
            <<','<<bit(fifo.enqueue_fire);
        descriptor(trace,fifo.enqueue_data_i);
        trace<<','<<bit(fifo.dequeue_valid_o)
            <<','<<bit(fifo.dequeue_ready_i)<<','<<bit(fifo.dequeue_fire);
        descriptor(trace,fifo.dequeue_data_o);
        trace<<','<<fifo.level_o<<','<<bit(view.canonical_path)
            <<','<<bit(view.load_request_active)
            <<','<<bit(view.load_pipeline_idle)
            <<','<<bit(view.distribution_idle)
            <<','<<bit(view.program_load_complete_o)
            <<','<<bit(view.program_ready_o)
            <<','<<bit(view.program_running_o)
            <<','<<bit(view.program_done_o)
            <<','<<bit(view.program_fault_o)
            <<','<<static_cast<unsigned>(view.program_fault_code_o)
            <<','<<bit(dma_rsp.fire&&dma_rsp.last)
            <<','<<static_cast<unsigned>(dma_rsp.id)
            <<','<<bit(ddr_rsp.valid)<<','<<bit(ddr_rsp.ready)
            <<','<<bit(ddr_rsp.fire)<<','<<ddr_rsp.beat_index
            <<','<<ddr_rsp.keep<<','<<bit(ddr_rsp.last)
            <<','<<static_cast<unsigned>(ddr_rsp.id)
            <<','<<bit(ddr_rsp.error)<<','<<hex_bytes(ddr_rsp.data)
            <<','<<bit(dma_rsp.valid)<<','<<bit(dma_rsp.ready)
            <<','<<bit(dma_rsp.fire)<<','<<dma_rsp.beat_index
            <<','<<dma_rsp.keep<<','<<bit(dma_rsp.last)
            <<','<<static_cast<unsigned>(dma_rsp.id)
            <<','<<bit(dma_rsp.error)<<','<<hex_bytes(dma_rsp.data)
            <<','<<r2.dma.response_fifo_level
            <<','<<bit(r2.reassembler_word_valid_o)
            <<','<<bit(r2.reassembler_word_ready_i)
            <<','<<bit(r2.reassembler_word_fire)
            <<','<<bit(r2.reassembler_image_complete)
            <<','<<r2.reassembler_word.imem_address
            <<','<<static_cast<unsigned>(r2.reassembler_word.strb)
            <<','<<bit(r2.reassembler_word.image_last)
            <<','<<hex_bytes(r2.reassembler_word.data)
            <<','<<bit(r2.word_fifo_enqueue_valid_i)
            <<','<<bit(r2.word_fifo_enqueue_ready_o)
            <<','<<bit(r2.word_fifo_enqueue_fire)
            <<','<<bit(r2.word_fifo_dequeue_valid_o)
            <<','<<bit(r2.word_fifo_dequeue_ready_i)
            <<','<<bit(r2.word_fifo_dequeue_fire)
            <<','<<r2.word_fifo_level_o
            <<','<<r2.word_fifo_dequeue.imem_address
            <<','<<static_cast<unsigned>(r2.word_fifo_dequeue.strb)
            <<','<<bit(r2.word_fifo_dequeue.image_last)
            <<','<<hex_bytes(r2.word_fifo_dequeue.data)
            <<','<<bit(mem.mem_program_valid_o)
            <<','<<bit(mem.mem_program_ready_i)
            <<','<<bit(mem.mem_program_fire)
            <<','<<mem.mem_program_imem_address_o
            <<','<<static_cast<unsigned>(mem.mem_program_strb_o)
            <<','<<bit(mem.mem_program_last_o)
            <<','<<bit(mem.mem_program_error_i)
            <<','<<bit(mem_frontend_ready_before_edge&&!loader.started())
            <<','<<mem_fetched_words<<','<<decoder.decoded_contexts
            <<','<<issued<<','<<issue_delta;
        if(!r2_mode) r3_row(trace,r3);
        trace<<'\n';
        ++rows;
        require(!view.program_fault_o,"canonical Program Load raised fault");
    }

    require(command_writes==4,"canonical command RAM write count mismatch");
    require(load_starts==1&&load_fires==1&&prime_fires==1,
        "canonical control fire count mismatch");
    require(config_enqueues==1&&config_dequeues==1,
        "Config FIFO enqueue/dequeue count mismatch");
    require(mem_program_fires==image.word_count(),
        "Program Distribution did not write every MEM word");
    require(ddr_response_fires==2&&dma_response_fires==2,
        "DMA response beat count mismatch");
    require(ddr_beat_indices==std::vector<std::size_t>{0,1}
            &&dma_beat_indices==std::vector<std::size_t>{0,1},
        "DMA response beat index order mismatch");
    require(ddr_final_cycle&&dma_final_cycle
            &&*dma_final_cycle==*ddr_final_cycle+1,
        "DMA response FIFO did not impose its registered one-cycle boundary");
    require(config_dequeue_cycle&&dma_final_cycle
            &&*config_dequeue_cycle==*dma_final_cycle,
        "Config FIFO did not retire on final Reassembler response fire");
    require(reassembler_word_fires==image.word_count()
            &&word_fifo_enqueues==image.word_count()
            &&word_fifo_dequeues==image.word_count(),
        "Reassembler/Word FIFO fire count mismatch");
    if(!r2_mode) {
        require(distributor_input_fires==image.word_count()+1
                &&distributor_write_fires==image.word_count()
                &&distributor_prime_fires==1
                &&elastic_output_fires==image.word_count()+1
                &&east_route_fires==image.word_count()+1
                &&west_route_fires==0
                &&east_relay0_input_fires==image.word_count()+1
                &&east_relay0_local_fires==image.word_count()+1,
            "Distributor/Elastic/Relay transfer count mismatch");
        require(!held_endpoint,
            "R3 trace ended with a held Program Endpoint payload");
    }
    require(loader.statistics().written_words==image.word_count(),
        "local i-MEM commit count mismatch");
    require(system.chip().icu().read_program_image(image)==bytes,
        "local i-MEM image differs from source bytes");
    require(queue.frontend_ready()&&queue.macro_fetch_complete()
            && queue.macro_decoder_statistics().fetched_words==image.word_count()
            && queue.macro_decoder_statistics().decoded_contexts!=0,
        "MEM Macro Frontend/Decoder did not consume production image");
    require(queue.issued_count()!=0,
        "MEM production scenario reached no FU-local Issue event");
    require(loader.config_fifo().empty()&&!loader.command_controller().fault(),
        "production scenario did not drain cleanly");

    const auto manifest_name=r2_mode?"phase3_r2_manifest.json":
        "phase3_r3_manifest.json";
    std::ofstream manifest(output_dir/manifest_name);
    if(!manifest) throw std::runtime_error("cannot open Phase3 manifest");
    manifest<<"{\n";
    if(r2_mode)
        manifest<<"  \"schema\": \"ftlpu.phase3.r2.manifest.v1\",\n"
            "  \"revision\": \"R2_DMA_REASSEMBLER_WORD_FIFO_V1\",\n";
    else
        manifest<<"  \"schema\": \"ftlpu.phase3.r3.manifest.v1\",\n"
            "  \"revision\": \"R3_DISTRIBUTOR_RELAY_ELASTIC_V1\",\n";
    manifest<<
        "  \"status\": \"PASS\",\n"
        "  \"phase\": \"PRE_POSEDGE\",\n"
        "  \"cycle_offset\": 0,\n"
        "  \"workload\": \"single_mem_raw_macro_96b_production\",\n"
        "  \"trace\": \""<<trace_name<<"\",\n"
        "  \"rows\": "<<rows<<",\n"
        "  \"physical_words\": "<<image.word_count()<<",\n"
        "  \"command_write_fires\": "<<command_writes<<",\n"
        "  \"load_start_fires\": "<<load_starts<<",\n"
        "  \"load_transaction_fires\": "<<load_fires<<",\n"
        "  \"config_enqueue_fires\": "<<config_enqueues<<",\n"
        "  \"config_dequeue_fires\": "<<config_dequeues<<",\n"
        "  \"ddr_response_fires\": "<<ddr_response_fires<<",\n"
        "  \"dma_response_fires\": "<<dma_response_fires<<",\n"
        "  \"ddr_final_cycle\": "<<*ddr_final_cycle<<",\n"
        "  \"dma_final_cycle\": "<<*dma_final_cycle<<",\n"
        "  \"config_dequeue_cycle\": "<<*config_dequeue_cycle<<",\n"
        "  \"reassembler_word_fires\": "<<reassembler_word_fires<<",\n"
        "  \"word_fifo_enqueue_fires\": "<<word_fifo_enqueues<<",\n"
        "  \"word_fifo_dequeue_fires\": "<<word_fifo_dequeues<<",\n";
    if(!r2_mode) manifest<<
        "  \"distributor_input_fires\": "<<distributor_input_fires<<",\n"
        "  \"distributor_write_fires\": "<<distributor_write_fires<<",\n"
        "  \"distributor_prime_fires\": "<<distributor_prime_fires<<",\n"
        "  \"elastic_output_fires\": "<<elastic_output_fires<<",\n"
        "  \"east_route_fires\": "<<east_route_fires<<",\n"
        "  \"west_route_fires\": "<<west_route_fires<<",\n"
        "  \"east_relay0_input_fires\": "<<east_relay0_input_fires<<",\n"
        "  \"east_relay0_local_fires\": "<<east_relay0_local_fires<<",\n"
        "  \"endpoint_stall_cycles\": "<<endpoint_stall_cycles<<",\n";
    manifest<<
        "  \"mem_program_fires\": "<<mem_program_fires<<",\n"
        "  \"mem_issue_count\": "<<queue.issued_count()<<"\n"
        "}\n";
    std::cout<<"phase3_program_loading_trace_test "
        <<(r2_mode?"R2":"R3")<<" passed: rows="<<rows
        <<" mem_issue_count="<<queue.issued_count()<<'\n';
    return 0;
} catch(const std::exception& error) {
    std::cerr<<"phase3_program_loading_trace_test failed: "
        <<error.what()<<'\n';
    return 1;
}
