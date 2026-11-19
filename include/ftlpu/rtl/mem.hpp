#pragma once
#include "ftlpu/rtl/config.hpp"
#include <array>
#include <optional>
#include <vector>

namespace ftlpu::rtl {
// One independently launched single-port bank column, bottom row = 0.
// tick() samples pre-edge commands/streams and publishes post-edge read
// responses. A prior row's execution becomes the next row's execution after
// one or three edges. No FIFO, retry, arbitration or inferred repeat count.
class MemBankColumn {
public:
  struct Stream {
    bool valid = false;
    std::uint64_t data = 0;
  };
  struct Event {
    bool valid = false;
    MemInstruction instruction{};
  };
  struct Response {
    bool valid = false, west = false;
    std::uint8_t stream = 0;
    std::uint64_t data = 0;
  };
  struct Outputs {
    std::vector<Event> executed;
    std::vector<Response> read;
    std::vector<bool> consume, fault;
    bool busy = false;
  };
  explicit MemBankColumn(Config c = {})
      : c_(c), memory_(c.rows, std::vector<std::uint64_t>(c.mem_depth)),
        execution_(c.rows), d1_(c.rows), d2_(c.rows) {
    c_.validate();
    if (c.rows != 2 && c.rows != 4 && c.rows != 8)
      throw std::invalid_argument("MEM supports 2/4/8 rows");
  }
  void reset() {
    for (auto *p : {&execution_, &d1_, &d2_})
      for (auto &x : *p)
        x.reset();
  }
  void initialize(std::size_t row, std::size_t address, std::uint64_t value) {
    memory_.at(row).at(address) = value;
  }
  std::uint64_t memory(std::size_t row, std::size_t address) const {
    return memory_.at(row).at(address);
  }
  Outputs tick(std::optional<std::uint32_t> raw = {},
               const std::vector<Stream> &input = {}) {
    Outputs o;
    o.executed.resize(c_.rows);
    o.read.resize(c_.rows);
    o.consume.resize(c_.rows);
    o.fault.resize(c_.rows);
    auto current = execution_;
    current[0].reset();
    if (raw) {
      try {
        current[0] = MemInstruction::decode(*raw, c_.mem_depth);
      } catch (const std::invalid_argument &) {
        o.fault[0] = true;
      }
    }
    for (std::size_t r = 0; r < c_.rows; ++r)
      if (current[r]) {
        const auto q = *current[r];
        o.executed[r] = {true, q};
        if (!q.write)
          o.read[r] = {true, q.west, q.stream, memory_[r][q.row]};
        else if (r < input.size() && input[r].valid) {
          memory_[r][q.row] = input[r].data;
          o.consume[r] = !q.preserve;
        } else if (!q.preserve)
          o.fault[r] = true;
      }
    auto next = execution_, n1 = d1_, n2 = d2_;
    next[0].reset();
    for (std::size_t r = 1; r < c_.rows; ++r) {
      const auto &source = current[r - 1];
      const bool short_valid = source && !source->delay3;
      if (short_valid && (d1_[r] || d2_[r]))
        throw std::logic_error(
            "unsafe MEM 3-to-1 schedule: pending long instruction");
      next[r] = d2_[r];
      if (!next[r] && short_valid)
        next[r] = source;
      n2[r] = d1_[r];
      n1[r] = (source && source->delay3) ? source : std::nullopt;
    }
    // RTL busy is combinational at execution endpoints and relay slots.
    for (std::size_t r = 0; r < c_.rows; ++r)
      o.busy = o.busy || bool(current[r]) || bool(d1_[r]) || bool(d2_[r]);
    execution_ = std::move(next);
    d1_ = std::move(n1);
    d2_ = std::move(n2);
    return o;
  }

private:
  Config c_;
  std::vector<std::vector<std::uint64_t>> memory_;
  std::vector<std::optional<MemInstruction>> execution_, d1_, d2_;
};
} // namespace ftlpu::rtl
