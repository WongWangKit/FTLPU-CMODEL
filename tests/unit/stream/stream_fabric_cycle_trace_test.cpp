#include "ftlpu/core/stream_fabric.hpp"
#include "ftlpu/core/timing.hpp"

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

struct Drive {
    bool east_p0{}; bool east_p1{};
    std::uint64_t east_p0_data{}; std::uint64_t east_p1_data{};
    bool east_consume{}; bool west_p0{}; std::uint64_t west_p0_data{};
};

std::string hex64(std::uint64_t value)
{
    std::ostringstream stream;
    stream << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
    return stream.str();
}

std::uint64_t segment_value(const StreamRegisterFabric& fabric,
    std::size_t column, std::size_t tile, StreamId stream)
{
    std::uint64_t value = 0;
    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
        value |= std::uint64_t(fabric.cell(column, tile, lane, stream).data)
            << (lane * 8);
    return value;
}

void stage_segment(StreamRegisterFabric& fabric, std::size_t column,
    StreamId stream, std::uint64_t value, const char* producer)
{
    StreamPayloadTileSegment bytes{};
    for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
        bytes[lane] = static_cast<std::uint8_t>(value >> (lane * 8));
    fabric.stage_payload_segment(column, 0, stream, bytes, 0, producer);
}

class TraceWriter {
public:
    explicit TraceWriter(const std::filesystem::path& path) : output_(path)
    {
        if (!output_) throw std::runtime_error("cannot create SR CModel trace");
        output_ << "schema,run_id,case_id,source,global_cycle,logical_cycle,"
                   "transport_tick,phase,sample_point,event_sequence,module,"
                   "instance,signal,width,value,event_type,mapping_status,source_ref\n";
    }
    void emit(std::size_t cycle, const char* phase, const char* sample,
        const char* instance, const char* signal, unsigned width,
        const std::string& value)
    {
        output_ << "ftlpu.mem_sr_sxm.cycle.v1,s1_sr_cycle,s1_sr,CMODEL,"
                << cycle << ',' << cycle << ',' << cycle << ',' << phase << ','
                << sample << ',' << sequence_++ << ",stream_register_fabric,"
                << instance << ',' << signal << ',' << width << ',' << value
                << ",SIGNAL_SAMPLE,DIRECT,stream_fabric.hpp\n";
    }
private:
    std::ofstream output_; std::size_t sequence_{};
};

Drive drive_for(std::size_t cycle)
{
    switch (cycle) {
    case 1: return {true,false,0x0807060504030201ull,0,false,true,0x1817161514131211ull};
    case 2: return {true,false,0x2827262524232221ull,0,true,true,0x3837363534333231ull};
    case 3: return {true,true,0x4847464544434241ull,0x5857565554535251ull,false,true,0x6867666564636261ull};
    case 4: return {false,false,0,0,true,true,0x7877767574737271ull};
    case 5: return {true,false,0x8887868584838281ull,0,false,false,0};
    case 6: return {true,false,0x9897969594939291ull,0,true,false,0};
    default: return {};
    }
}

void emit_inputs(TraceWriter& trace, std::size_t cycle, const Drive& drive)
{
    trace.emit(cycle,"LOW","PRE_POSEDGE","east.c0.t0.s0","inject_p0_valid_i",1,drive.east_p0?"1":"0");
    trace.emit(cycle,"LOW","PRE_POSEDGE","east.c0.t0.s0","inject_p0_data_i",64,hex64(drive.east_p0_data));
    trace.emit(cycle,"LOW","PRE_POSEDGE","east.c0.t0.s0","inject_p1_valid_i",1,drive.east_p1?"1":"0");
    trace.emit(cycle,"LOW","PRE_POSEDGE","east.c0.t0.s0","inject_p1_data_i",64,hex64(drive.east_p1_data));
    trace.emit(cycle,"LOW","PRE_POSEDGE","east.c0.t0.s0","consume_i",1,drive.east_consume?"1":"0");
    trace.emit(cycle,"LOW","PRE_POSEDGE","west.c1.t0.s1","inject_p0_valid_i",1,drive.west_p0?"1":"0");
    trace.emit(cycle,"LOW","PRE_POSEDGE","west.c1.t0.s1","inject_p0_data_i",64,hex64(drive.west_p0_data));
}

void emit_state(TraceWriter& trace,const StreamRegisterFabric& fabric,
    std::size_t cycle,const char* phase,const char* sample)
{
    const auto east=StreamId::East(0),west=StreamId::West(1);
    trace.emit(cycle,phase,sample,"east.c0.t0.s0","state_valid_o",1,fabric.cell(0,0,0,east).valid?"1":"0");
    trace.emit(cycle,phase,sample,"east.c0.t0.s0","state_data_o",64,hex64(segment_value(fabric,0,0,east)));
    trace.emit(cycle,phase,sample,"west.c1.t0.s1","state_valid_o",1,fabric.cell(1,0,0,west).valid?"1":"0");
    trace.emit(cycle,phase,sample,"west.c1.t0.s1","state_data_o",64,hex64(segment_value(fabric,1,0,west)));
}
}

int main(int argc,char** argv) try {
    const std::filesystem::path output_dir=argc==2?argv[1]:".";
    std::filesystem::create_directories(output_dir);
    TraceWriter trace(output_dir/"cmodel_s1_sr_cycle_trace.csv");
    StreamRegisterFabric fabric(2); timing::Clock clock; clock.register_module(fabric); fabric.reset();
    for(std::size_t cycle=0;cycle<7;++cycle){
        const auto drive=drive_for(cycle); fabric.low();
        if(drive.east_p0)stage_segment(fabric,0,StreamId::East(0),drive.east_p0_data,"producer0");
        if(drive.east_p1)stage_segment(fabric,0,StreamId::East(0),drive.east_p1_data,"producer1");
        if(drive.east_consume)fabric.consume_segment(0,0,StreamId::East(0),"consumer0");
        if(drive.west_p0)stage_segment(fabric,1,StreamId::West(1),drive.west_p0_data,"producer0");
        emit_inputs(trace,cycle,drive); emit_state(trace,fabric,cycle,"LOW","PRE_POSEDGE");
        const auto& view=fabric.hardware_interface_view();
        trace.emit(cycle,"LOW","PRE_POSEDGE","east.c0.t0","collision_o",1,view.collision_o[0]?"1":"0");
        trace.emit(cycle,"LOW","PRE_POSEDGE","east.c0.t0","invalid_consume_o",1,view.invalid_consume_o[0]?"1":"0");
        clock.complete_low();clock.validate_all();clock.commit_all();
        emit_state(trace,fabric,cycle,"HIGH","POST_POSEDGE");
        clock.complete_high();clock.negedge_all();
    }
    std::cout<<"SR_CMODEL_TRACE_PASS cycles=7\n";return 0;
} catch(const std::exception& e){std::cerr<<"stream_fabric_cycle_trace_test: "<<e.what()<<'\n';return 1;}
