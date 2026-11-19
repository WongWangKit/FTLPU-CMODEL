#pragma once
#include "ftlpu/rtl/mem.hpp"
#include "ftlpu/rtl/mxm_core.hpp"
#include "ftlpu/rtl/program.hpp"
#include "ftlpu/rtl/vxm_tile.hpp"
namespace ftlpu::rtl {
// Public entry point for current replacement-block simulation. Coordinates
// and raw ports match the new RTL; this collection does not invent a missing
// whole-chip SRAM/SXM routing network or a new ICU binary format.
class Model {
public:
  explicit Model(Config config = {}, std::size_t memory_columns = 1,
                 std::size_t mxm_units = 1)
      : config_(config) {
    config_.validate();
    if (!memory_columns || !mxm_units)
      throw std::invalid_argument("unit count");
    for (std::size_t i = 0; i < memory_columns; ++i)
      memory_.emplace_back(config_);
    for (std::size_t i = 0; i < mxm_units; ++i)
      mxm_.emplace_back(config_);
    tiles_.resize(config_.rows);
  }
  MemBankColumn &mem(std::size_t column = 0) { return memory_.at(column); }
  MxmCore &mxm(std::size_t unit = 0) { return mxm_.at(unit); }
  VxmTile &vxm(std::size_t superlane = 0) { return tiles_.at(superlane); }
  const Config &config() const { return config_; }
  void reset() {
    for (auto &m : memory_)
      m.reset();
    for (auto &m : mxm_)
      m.reset();
    for (auto &v : tiles_)
      v.reset();
  }

private:
  Config config_;
  std::vector<MemBankColumn> memory_;
  std::vector<MxmCore> mxm_;
  std::vector<VxmTile> tiles_;
};
} // namespace ftlpu::rtl
