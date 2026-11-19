#include "ftlpu/cmodel.hpp"
#include "ftlpu/rtl/program.hpp"
#include <iostream>

int main() {
  using namespace ftlpu::rtl;
  ftlpu::CModel model(Config::profile(32));
  // A macro emits four primitive MEM words. COUNT is outside the 32-bit ISA.
  Macro<MemInstruction> store{0, 4, 1, {true, false, true, false, 0, 7}, 1};
  Macro<MemInstruction> read{16, 4, 1, {false, false, true, false, 0, 3}, 1};
  unsigned results = 0;
  for (unsigned cycle = 0; cycle < 40; ++cycle) {
    std::optional<std::uint32_t> word;
    if (auto i = store.at(cycle))
      word = i->encode(model.config().mem_depth);
    if (auto i = read.at(cycle))
      word = i->encode(model.config().mem_depth);
    std::vector<MemBankColumn::Stream> streams(4);
    for (unsigned row = 0; row < 4; ++row)
      streams[row] = {true, (std::uint64_t(row) << 32) | cycle};
    const auto o = model.mem().tick(word, streams);
    for (unsigned row = 0; row < 4; ++row) {
      if (o.fault[row])
        return 1;
      if (o.read[row].valid) {
        ++results;
        std::cout << "E" << cycle << " SL" << row
                  << " stream=" << unsigned(o.read[row].stream)
                  << " data=" << o.read[row].data << '\n';
      }
    }
  }
  return results == 16 ? 0 : 1;
}
