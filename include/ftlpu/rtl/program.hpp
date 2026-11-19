#pragma once
#include "ftlpu/rtl/config.hpp"
#include "ftlpu/rtl/vxm_tile.hpp"
#include <limits>
namespace ftlpu::rtl {
// Repeat belongs to this macro layer. The primitive encoders in config.hpp
// contain no COUNT. at() has no implicit stalls and never postpones a beat.
template <class Instruction> struct Macro {
  std::uint64_t start = 0, count = 1, interval = 1;
  Instruction primitive{};
  std::int64_t address_step = 0;
  std::optional<Instruction> at(std::uint64_t cycle) const {
    if (!count || !interval)
      throw std::invalid_argument("macro count/interval must be positive");
    if (cycle < start || (cycle - start) % interval)
      return {};
    const auto i = (cycle - start) / interval;
    if (i >= count)
      return {};
    auto result = primitive;
    auto induced = [&](std::uint64_t base, std::uint64_t limit) {
      if (address_step >= 0) {
        const auto step = std::uint64_t(address_step);
        if (base > limit || (step && i > (limit - base) / step))
          throw std::out_of_range("macro address induction");
        return std::uint16_t(base + i * step);
      }
      const auto step = std::uint64_t(0) - std::uint64_t(address_step);
      if (base > limit || i > base / step)
        throw std::out_of_range("macro address induction");
      return std::uint16_t(base - i * step);
    };
    if constexpr (requires { result.address; }) {
      result.address = induced(primitive.address, 65535);
    } else if constexpr (requires { result.row; }) {
      result.row = induced(primitive.row, 32767);
    } else if (address_step)
      throw std::invalid_argument("primitive has no address induction field");
    return result;
  }
};
// Cycle model of lpu_vxm_instruction_controller. Tile load, repeated execute,
// and final-marker retirement are distinct handshakes, with one shared count.
class VxmController {
public:
  struct Command {
    VxmTile::Configuration config;
    std::uint32_t count = 1, interval = 1;
    bool accumulate = false, output = true;
  };
  struct Output {
    bool command_ready = true, done = false, fault = false;
    std::optional<VxmTile::Configuration> load;
    bool execute = false;
    VxmTile::Phase phase;
  };
  VxmController() { reset(); }
  void reset() {
    state_ = Idle;
    remaining_ = cooldown_ = 0;
    first_ = false;
    fault_ = false;
    command_ = {};
    command_.output = false;
  }
  Output tick(std::optional<Command> command, bool load_ready,
              bool execute_ready, bool config_done) {
    Output o;
    o.fault = fault_;
    if (state_ == Idle) {
      if (command) {
        if (command->count > 1023 || command->interval > 255)
          throw std::invalid_argument("VXM resident repeat field width");
        if (!command->count)
          o.fault = true;
        else {
          command_ = *command;
          remaining_ = command->count;
          first_ = true;
          state_ = Load;
        }
      }
    } else if (state_ == Load) {
      if (load_ready)
        state_ = Run;
    } else if (state_ == Run) {
      if (cooldown_)
        --cooldown_;
      else if (execute_ready) {
        if (remaining_ == 1) {
          remaining_ = 0;
          state_ = WaitDone;
        } else {
          --remaining_;
          first_ = false;
          cooldown_ = command_.interval > 1 ? command_.interval - 1 : 0;
        }
      }
    } else if (state_ == WaitDone && config_done) {
      o.done = true;
      state_ = Idle;
    }
    o.command_ready = state_ == Idle;
    o.fault = fault_ = o.fault;
    if (state_ == Load)
      o.load = command_.config;
    o.execute = state_ == Run && !cooldown_;
    o.phase = {first_, remaining_ == 1, command_.accumulate, command_.output};
    return o;
  }

private:
  enum State { Idle, Load, Run, WaitDone };
  State state_ = Idle;
  Command command_{};
  std::uint32_t remaining_ = 0, cooldown_ = 0;
  bool first_ = false, fault_ = false;
};
} // namespace ftlpu::rtl
