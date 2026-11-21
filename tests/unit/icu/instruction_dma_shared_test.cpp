#include "ftlpu/icu/instruction_fetch_dma.hpp"
#include "ftlpu/c2c/dma.hpp"
#include <iostream>
using namespace ftlpu;
void require(bool p,const char*m){if(!p)throw std::runtime_error(m);}
std::uint8_t value(std::size_t a){return (a*17+(a>>5)*7)^0xa3;}
void initialize(Ddr4Model& d,std::size_t base,std::size_t n){
 for(std::size_t a=base;a<base+n;a+=hw::kPhysicalVectorBytes){C2cVector v;
  for(std::size_t b=0;b<hw::kPhysicalVectorBytes;++b)v.payload[b/hw::kLanesPerTile][b%hw::kLanesPerTile]=value(a+b);
  d.initialize_vector(a,v);
 }
}
int main()try{
 const auto sizing = hw::config::kC2cDmaSizing;
 Ddr4Model defaults;
 C2cDmaEngine default_data(defaults);
 InstructionFetchDmaConfig default_program;
 require(default_program.response_fifo_depth == sizing.p, "default P mismatch");
 require(default_data.fifo_depth_vectors() == sizing.rt, "default RT mismatch");
 require(defaults.config().request_queue_depth == sizing.q, "default Q mismatch");
 const auto& geometry = defaults.axi().config();
 const auto beats = (hw::kPhysicalVectorBytes + defaults.config().beat_bytes - 1)
     / defaults.config().beat_bytes;
 require(geometry.transactions == sizing.q && geometry.ids == 8
     && geometry.per_id == (sizing.q + 7) / 8
     && geometry.rx_beats == sizing.q * beats
     && geometry.tx_beats == sizing.q * beats, "default Q geometry mismatch");
 initialize(defaults, 0x8000, 16 * hw::kPhysicalVectorBytes);
 default_data.issue(C2cDmaInstruction::Load(0x8000, 16));
 std::size_t default_vectors = 0;
 for (std::size_t cycle = 0; cycle < 20000; ++cycle) {
  if (cycle >= 1000 && default_data.receive_ready()) {
   const auto vector = default_data.pop_received();
   for (std::size_t b = 0; b < hw::kPhysicalVectorBytes; ++b)
    require(vector.payload[b / hw::kLanesPerTile][b % hw::kLanesPerTile]
        == value(0x8000 + default_vectors * hw::kPhysicalVectorBytes + b),
        "default sizing data mismatch");
   ++default_vectors;
  }
  default_data.tick(); defaults.tick();
  if (default_data.drained() && defaults.drained()) break;
 }
 require(default_vectors == 16 && defaults.drained(), "default sizing did not drain");
 Ddr4Config c;c.beat_bytes=std::min<std::size_t>(32,hw::kPhysicalVectorBytes);c.request_queue_depth=16;
 c.read_latency_cycles=7;c.read_latency_jitter_cycles=13;
 Ddr4Model d(c);InstructionFetchDmaConfig cfg;cfg.response_fifo_depth=2;cfg.max_outstanding=2;
 InstructionFetchDma p(d,cfg);C2cDmaEngine data(d,2,1,0);
 p.enable_clocked_mode();data.enable_clocked_mode();d.enable_clocked_mode();
 auto tick=[&]{p.low();data.low();d.low();p.posedge();data.posedge();d.posedge();p.high();data.high();d.high();};
 initialize(d,0xf00,4096);initialize(d,0x4000,8*hw::kPhysicalVectorBytes);
 p.start({0xff8,130,PhysicalInstructionWordWidth::Bits96,7});data.issue(C2cDmaInstruction::Load(0x4000,8));
 std::size_t words=0,vectors=0;bool filled=false;
 for(std::size_t t=0;t<20000;++t){
  if(t>=1000&&p.word_ready()){const auto w=p.front_word();require(w.index==words,"program word order");for(std::size_t b=0;b<w.byte_count;++b)require(w.bytes[b]==value(0xff8+words*12+b),"program data mismatch");++words;p.pop_word();}
  if(t>=300&&data.receive_ready()){auto v=data.pop_received();for(std::size_t b=0;b<hw::kPhysicalVectorBytes;++b)require(v.payload[b/hw::kLanesPerTile][b%hw::kLanesPerTile]==value(0x4000+vectors*hw::kPhysicalVectorBytes+b),"data mismatch");++vectors;}
  tick();filled|=p.response_fifo_occupancy()==2;
  require(!p.fault()&&!data.error(),"unexpected transport failure");
  require(p.pending_reads()+p.response_fifo_occupancy()<=2,"unreserved program response");
  require(d.axi().eval().rready,"external R blocked by consumer");
  if(p.drained()&&data.drained()&&d.drained())break;
 }
 require(words==130&&vectors==8&&filled&&d.drained(),"shared stalled transfer incomplete");
 // Legal R error, accepted-transaction drain, fresh task and exact bytes.
 initialize(d,0x6000,2048);d.set_response_fault([](auto a,bool w)->std::uint8_t{return !w&&a>=0x6000?2:0;});
 p.start({0x6000,64,PhysicalInstructionWordWidth::Bits128,8});
 for(int n=0;n<2000&&!p.fault();++n)tick();require(p.fault(),"R fault absent");
 p.begin_abort();for(int n=0;n<2000&&(!p.drained()||!d.drained());++n)tick();
 require(p.drained()&&d.drained()&&p.pending_reads()==0,"cancel leaked returns");
 p.recover();d.set_response_fault({});p.start({0x6000,64,PhysicalInstructionWordWidth::Bits128,9});words=0;
 for(int n=0;n<5000;++n){if(p.word_ready()){auto w=p.front_word();for(std::size_t b=0;b<16;++b)require(w.bytes[b]==value(0x6000+words*16+b),"recovered data mismatch");++words;p.pop_word();}tick();if(p.drained()&&d.drained())break;}
 require(words==64&&!p.fault()&&d.drained(),"new generation did not recover");
 std::cout<<"shared instruction DMA PASS: cross4K, unaligned tail, 1000-cycle consumer hold, data contention, reserved RX, R fault/drain/restart\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
