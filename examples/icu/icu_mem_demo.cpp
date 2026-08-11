#include "ftlpu/mem/tile_array.hpp"
#include "ftlpu/icu/icu.hpp"
#include "ftlpu/core/instruction_pipeline.hpp"
#include <cassert>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
int main(){
ftlpu::IcuUnit<ftlpu::MemInstruction> icu;
icu.load({ftlpu::IcuInstruction::Dispatch(),ftlpu::IcuInstruction::Nop(2),ftlpu::IcuInstruction::Dispatch(),ftlpu::IcuInstruction::Repeat(1,3)},
         {ftlpu::MemInstruction::Read(0,1),ftlpu::MemInstruction::Write(320,2)});
icu.notify();
ftlpu::NorthboundInstructionPipeline pipe;
std::ostringstream log;
for(int c=0;c<30;++c){
  auto i=icu.tick();if(i)pipe.issue_south(*i);
  pipe.tick(log);
  if(icu.done())break; }
auto t=log.str();
assert(t.find("Read a=0 s=1")!=std::string::npos);
assert(t.find("Write a=320 s=2")!=std::string::npos);
assert(t.find("NOP")==std::string::npos);
std::cout<<"OK: ICU drives NorthboundInstructionPipeline\n";return 0;}