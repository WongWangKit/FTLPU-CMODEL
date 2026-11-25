#include "ftlpu/core/stream_fabric.hpp"
#include "ftlpu/core/timing.hpp"
#include "ftlpu/mem/mem_array.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
using namespace ftlpu;
std::string hex64(std::uint64_t value){std::ostringstream s;s<<"0x"<<std::hex<<std::setw(16)<<std::setfill('0')<<value;return s.str();}
std::uint64_t pattern(std::size_t tile){std::uint64_t v=0;for(std::size_t lane=0;lane<8;++lane)v|=std::uint64_t(0x10+tile*8+lane)<<(lane*8);return v;}
std::uint64_t segment(const StreamRegisterFabric& f,std::size_t column,std::size_t tile){std::uint64_t v=0;for(std::size_t lane=0;lane<8;++lane)v|=std::uint64_t(f.cell(column,tile,lane,StreamId::East(0)).data)<<(lane*8);return v;}
void stage_boundary(StreamRegisterFabric& f,std::size_t tile,std::uint64_t value){StreamPayloadTileSegment b{};for(std::size_t lane=0;lane<8;++lane)b[lane]=static_cast<std::uint8_t>(value>>(lane*8));f.stage_payload_segment(0,tile,StreamId::East(0),b,0,"boundary input");}
class Trace{public:explicit Trace(const std::filesystem::path&p):o(p){if(!o)throw std::runtime_error("cannot create MEM/SR trace");o<<"schema,run_id,case_id,source,global_cycle,logical_cycle,transport_tick,phase,sample_point,event_sequence,module,instance,signal,width,value,event_type,mapping_status,source_ref\n";}void emit(std::size_t c,const char*phase,const char*sample,const std::string&module,const std::string&instance,const std::string&signal,unsigned width,const std::string&value){o<<"ftlpu.mem_sr_sxm.cycle.v1,s2_mem_sr,s2_mem_sr,CMODEL,"<<c<<','<<c<<','<<c<<','<<phase<<','<<sample<<','<<seq++<<','<<module<<','<<instance<<','<<signal<<','<<width<<','<<value<<",SIGNAL_SAMPLE,DIRECT,mem_array.hpp\n";}private:std::ofstream o;std::size_t seq{};};
const MemArrayModel::MemTransfer* find_transfer(const std::vector<MemArrayModel::MemTransfer>&v,std::size_t tile){for(const auto&x:v)if(x.mem_slice==0&&x.bank==0&&x.tile==tile)return&x;return nullptr;}
std::uint64_t bytes64(const StreamPayloadTileSegment&b){std::uint64_t v=0;for(std::size_t lane=0;lane<8;++lane)v|=std::uint64_t(b[lane])<<(lane*8);return v;}
void emit_state(Trace&tr,const StreamRegisterFabric&f,std::size_t c,const char*phase,const char*sample){for(std::size_t tile=0;tile<4;++tile)for(std::size_t column=0;column<2;++column){const auto inst="east.c"+std::to_string(column)+".t"+std::to_string(tile)+".s0";const bool valid=f.cell(column,tile,0,StreamId::East(0)).valid;tr.emit(c,phase,sample,"stream_register_fabric",inst,"state_valid_o",1,valid?"1":"0");if(valid)tr.emit(c,phase,sample,"stream_register_fabric",inst,"state_data_o",64,hex64(segment(f,column,tile)));}}
}
int main(int argc,char**argv)try{const std::filesystem::path out=argc==2?argv[1]:".";std::filesystem::create_directories(out);Trace tr(out/"cmodel_s2_mem_sr_cycle_trace.csv");StreamRegisterFabric fabric(hw::kMemBoundaryStreamRegisterColumns);MemArrayModel mem(MemStreamPortMap::BetweenBoundaries(),MemArrayModel::MissingStreamPolicy::Error,MemArrayModel::ReadTiming::RegisteredR1);timing::Clock clock;clock.register_module(fabric);clock.register_module(mem);
for(std::size_t c=0;c<13;++c){if(c==1)mem.enqueue_instruction(0,MemInstruction::Write(3,StreamId::East(0)));if(c==7)mem.enqueue_instruction(0,MemInstruction::Read(3,StreamId::East(0)));fabric.low();const bool boundary_valid=c<4;if(boundary_valid)stage_boundary(fabric,c,pattern(c));mem.evaluate(fabric,false);fabric.stage_linear_links();tr.emit(c,"LOW","PRE_POSEDGE","mem","slice0.bank0","bank_issue_valid_i",1,(c==1||c==7)?"1":"0");if(c==1||c==7)tr.emit(c,"LOW","PRE_POSEDGE","mem","slice0.bank0","bank_issue_i",32,c==1?"0x00018001":"0x00018000");
for(std::size_t tile=0;tile<4;++tile){const auto inst="east.t"+std::to_string(tile)+".s0";tr.emit(c,"LOW","PRE_POSEDGE","mem_sr_boundary",inst,"boundary_valid_i",1,(boundary_valid&&tile==c)?"1":"0");if(boundary_valid&&tile==c)tr.emit(c,"LOW","PRE_POSEDGE","mem_sr_boundary",inst,"boundary_data_i",64,hex64(pattern(tile)));const auto&v=mem.hardware_interface_view();const auto*consume=find_transfer(v.mem_boundary_consume_o,tile);const auto*producer=find_transfer(v.mem_producer_valid_o,tile);const auto mi="slice0.bank0.t"+std::to_string(tile);tr.emit(c,"LOW","PRE_POSEDGE","mem",mi,"mem_boundary_consume_o",1,consume?"1":"0");tr.emit(c,"LOW","PRE_POSEDGE","mem",mi,"mem_producer_valid_o",1,producer?"1":"0");if(producer){const auto index=static_cast<std::size_t>(producer-v.mem_producer_valid_o.data());tr.emit(c,"LOW","PRE_POSEDGE","mem",mi,"mem_producer_data_o",64,hex64(bytes64(v.mem_producer_data_o[index])));tr.emit(c,"LOW","PRE_POSEDGE","mem",mi,"mem_producer_direction_o",1,v.mem_producer_direction_o[index]==StreamDirection::West?"1":"0");tr.emit(c,"LOW","PRE_POSEDGE","mem",mi,"mem_producer_stream_o",5,std::to_string(v.mem_producer_stream_o[index].index()));tr.emit(c,"LOW","PRE_POSEDGE","mem",mi,"mem_producer_boundary_o",4,std::to_string(v.mem_producer_boundary_o[index]));}}
tr.emit(c,"LOW","PRE_POSEDGE","mem","east","mem_fault_valid_o",1,mem.hardware_interface_view().mem_fault_valid_o?"1":"0");tr.emit(c,"LOW","PRE_POSEDGE","mem","east","mem_busy_o",1,mem.hardware_interface_view().mem_busy_o?"1":"0");emit_state(tr,fabric,c,"LOW","PRE_POSEDGE");clock.complete_low();clock.validate_all();clock.commit_all();emit_state(tr,fabric,c,"HIGH","POST_POSEDGE");clock.complete_high();clock.negedge_all();}
std::cout<<"MEM_SR_CMODEL_TRACE_PASS cycles=13\n";return 0;}catch(const std::exception&e){std::cerr<<"mem_sr_cycle_trace_test: "<<e.what()<<'\n';return 1;}
