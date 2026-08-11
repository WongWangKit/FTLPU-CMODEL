#include "ftlpu/icu/icu.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
namespace {
struct TestFunc { int op{0}; std::size_t addr{0}; std::size_t stream{0}; 
bool operator==(const TestFunc& o) const { return op==o.op&&addr==o.addr&&stream==o.stream; } };
bool require(bool c, const char* m) { if(!c){std::cerr<<"FAIL: "<<m<<"\n";} return c; }
}
int main() {
using Icu=ftlpu::IcuUnit<TestFunc>;
using I=ftlpu::IcuInstruction;
{ Icu icu; icu.load({},{}); assert(icu.synced()); icu.notify(); assert(icu.done()); }
{ Icu icu; icu.load({I::Dispatch()},{TestFunc{1,100,5}}); icu.notify();
  auto i=icu.tick(); assert(i.has_value()&&i->op==1); assert(icu.done()); }
{ Icu icu; icu.load({I::Nop(5),I::Dispatch()},{TestFunc{2,200,6}}); icu.notify();
  for(int j=0;j<5;++j){auto i=icu.tick(); assert(!i.has_value());}
  auto i=icu.tick(); assert(i.has_value()&&i->op==2); assert(icu.done()); }
{ Icu icu; icu.load({I::Dispatch(),I::Nop(1),I::Dispatch()},{TestFunc{3,0,0},TestFunc{4,0,0}}); icu.notify();
  auto i=icu.tick(); assert(i.has_value()&&i->op==3);
  i=icu.tick(); assert(!i.has_value());
  i=icu.tick(); assert(i.has_value()&&i->op==4); assert(icu.done()); }
{ Icu icu; icu.load({I::Dispatch(),I::Repeat(3,1)},{TestFunc{5,400,8}}); icu.notify();
  auto i=icu.tick(); assert(i.has_value()&&i->op==5);
  i=icu.tick(); assert(i.has_value()&&i->op==5);
  i=icu.tick(); assert(i.has_value()&&i->op==5);
  i=icu.tick(); assert(i.has_value()&&i->op==5); assert(icu.done()); }
{ Icu icu; icu.load({I::Dispatch(),I::Repeat(2,3)},{TestFunc{6,500,9}}); icu.notify();
  auto i=icu.tick(); assert(i.has_value());
  i=icu.tick(); assert(i.has_value());
  i=icu.tick(); assert(!i.has_value());
  i=icu.tick(); assert(!i.has_value());
  i=icu.tick(); assert(i.has_value()); assert(icu.done()); }
{ Icu icu; icu.load({I::Dispatch(),I::Sync(),I::Dispatch()},{TestFunc{10,0,0},TestFunc{11,0,0}}); icu.notify();
  auto i=icu.tick(); assert(i.has_value()&&i->op==10);
  i=icu.tick(); assert(icu.synced()); icu.notify();
  i=icu.tick(); assert(i.has_value()&&i->op==11); assert(icu.done()); }
{ Icu icu; icu.load({I::Dispatch()},{TestFunc{30,0,0}});icu.notify();icu.tick();assert(icu.done());
  icu.load({I::Dispatch()},{TestFunc{31,0,0}});assert(icu.synced());icu.notify();
  auto i=icu.tick();assert(i.has_value()&&i->op==31); }
{ Icu icu; icu.load({I::Sync(),I::Notify(),I::Dispatch()},{TestFunc{14,0,0}});
  assert(icu.synced());icu.notify();icu.tick();assert(icu.synced());icu.notify();
  auto i=icu.tick();assert(!i.has_value());
  i=icu.tick();assert(i.has_value()&&i->op==14);assert(icu.done()); }
{ Icu icu; icu.load({I::Dispatch(),I::Nop(3),I::Sync()},{TestFunc{40,0,0}});icu.notify();
  assert(icu.ice_pc()==0&&icu.func_pc()==0);icu.tick();assert(icu.ice_pc()==1);
  icu.tick();assert(icu.ice_pc()==2);icu.tick();icu.tick();icu.tick();assert(icu.synced()); }
}