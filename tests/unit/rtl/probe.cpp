#include "ftlpu/rtl/mem.hpp"
#include "ftlpu/rtl/mxm_core.hpp"
#include "ftlpu/rtl/program.hpp"
#include "ftlpu/rtl/vxm.hpp"
#include "ftlpu/rtl/vxm_tile.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
using namespace ftlpu::rtl;
int main(int argc, char **argv) {
  if (argc == 4 && std::string(argv[1]) == "conversion") {
    std::ifstream input(argv[2]);
    std::ofstream output(argv[3]);
    unsigned valid, from, to, t = 0;
    std::uint32_t raw;
    while (input >> std::dec >> valid >> from >> to >> std::hex >> raw) {
      const auto c = vxmath::convert(valid, Format(from), Format(to), raw);
      output << std::dec << t++ << ' ' << c.valid << ' ' << c.active << ' '
             << c.fault << ' ' << std::hex << std::setfill('0') << std::setw(8)
             << c.bits << '\n';
    }
    return 0;
  }
  if (argc == 4 && std::string(argv[1]) == "controller") {
    VxmController controller;
    std::ifstream input(argv[2]);
    std::ofstream output(argv[3]);
    unsigned reset, valid, count, interval, acc, out, lr, er, done, t = 0;
    while (input >> reset >> valid >> count >> interval >> acc >> out >> lr >>
           er >> done) {
      if (reset)
        controller.reset();
      std::optional<VxmController::Command> command;
      if (valid && !reset)
        command =
            VxmController::Command{{}, count, interval, bool(acc), bool(out)};
      const auto o = controller.tick(command, lr, er, done);
      const auto p = o.phase;
      output << std::dec << t++ << ' ' << o.fault << ' ' << o.command_ready
             << ' ' << bool(o.load) << ' ' << o.execute << ' ' << std::hex
             << unsigned(p.last | (p.first << 1) | (p.accumulate << 2) |
                         (p.output << 3))
             << ' ' << std::dec << o.done << ' ' << !o.command_ready << '\n';
    }
    return 0;
  }
  if (argc == 4 && std::string(argv[1]) == "backend") {
    Config cfg;
    cfg.accumulator_depth = 32;
    Accumulator acc(cfg);
    std::ifstream input(argv[2]);
    std::ofstream output(argv[3]);
    unsigned run, reset, launch, op, bf, address, pv, tag, t = 0;
    std::vector<std::uint32_t> bits(8);
    while (input >> std::dec >> run >> reset >> launch >> op >> bf >> address >>
           pv >> tag) {
      for (auto &x : bits)
        input >> std::hex >> x;
      if (reset)
        acc.reset();
      else if (!run)
        acc.flush();
      std::optional<Accumulator::Request> r;
      std::optional<Batch> a;
      if (run && !reset) {
        if (launch)
          r = Accumulator::Request{
              {BackendOperation(op), std::uint16_t(address), bool(bf)}, {}};
        if (pv) {
          a = Batch{};
          for (auto x : bits)
            a->push_back({x, std::uint16_t(tag), false});
        }
      }
      const auto o = acc.tick(r, a);
      unsigned req = 0, wr = 0, ba = 0;
      for (unsigned b = 0; b < 2; ++b)
        if (o.bank[b].valid) {
          req |= 1 << b;
          wr |= unsigned(o.bank[b].write) << b;
          ba |= unsigned(o.bank[b].row) << (4 * b);
        }
      output << std::dec << t++ << ' ' << o.fault << ' ' << bool(o.values)
             << ' ' << (o.values && o.bf16) << ' ' << (o.values ? o.address : 0)
             << ' ' << (o.values ? o.tag : 0) << ' ' << std::hex
             << std::setfill('0');
      for (int n = 7; n >= 0; --n)
        output << std::setw(8) << (o.values ? (*o.values)[n] : 0);
      output << ' ' << req << ' ' << wr << ' ' << std::setw(2) << ba << ' ';
      for (int b = 1; b >= 0; --b)
        for (int n = 7; n >= 0; --n)
          output << std::setw(8)
                 << (o.bank[b].valid && o.bank[b].write
                         ? acc.memory(2 * o.bank[b].row + b)[n]
                         : 0);
      output << '\n';
    }
    return 0;
  }
  if (argc == 4 && std::string(argv[1]) == "dequant") {
    WeightDequantizer dq;
    std::ifstream input(argv[2]);
    std::ofstream output(argv[3]);
    unsigned reset, run, request, wb, load, param_word, dq_word, t = 0;
    std::uint64_t mask;
    while (input >> std::dec >> reset >> run >> request >> wb >> load >>
           param_word >> dq_word >> std::hex >> mask) {
      WeightDequantizer::Streams streams;
      for (unsigned stream = 0; stream < 6; ++stream)
        for (unsigned lane = 0; lane < 8; ++lane) {
          unsigned byte;
          input >> std::hex >> byte;
          const bool valid = (mask >> (stream * 8 + lane)) & 1;
          streams[stream].push_back(valid ? std::optional(std::uint8_t(byte))
                                          : std::nullopt);
        }
      if (reset)
        dq.reset();
      else if (!run)
        dq.flush();
      WeightDequantizer::Input i;
      if (run && !reset) {
        if (request)
          i.dequant = DequantInstruction::decode(dq_word);
        if (load)
          i.load = ParamLoadInstruction::decode(param_word);
        i.weight_buffer = wb;
      }
      const auto &o = dq.tick(i, streams);
      std::uint64_t consume = 0;
      for (unsigned stream = 0; stream < 6; ++stream)
        consume |= std::uint64_t(o.consume[stream]) << (stream * 8);
      output << std::dec << t++ << ' ' << o.fault << ' ' << o.accepted << ' '
             << o.parameter_accepted << ' ' << std::hex << std::setfill('0')
             << std::setw(12) << consume << ' ' << std::dec << bool(o.weight)
             << ' ' << (o.weight && o.weight->buffer) << ' ' << std::hex;
      for (int l = 7; l >= 0; --l)
        output << std::setw(4) << (o.weight ? o.weight->values[l] : 0);
      output << ' ' << std::dec << (dq.parameter_ready(false) | (dq.parameter_ready(true) << 1)) << '\n';
    }
    return 0;
  }
  if ((argc == 6 || argc == 7) && std::string(argv[1]) == "tile") {
    const unsigned chain = std::stoi(argv[2]), fmt = std::stoi(argv[3]),
                   op = std::stoi(argv[4]);
    const unsigned mode = argc == 7 ? std::stoi(argv[5]) : 0;
    VxmTile tile;
    VxmTile::Configuration conf;
    conf.global.chain = chain;
    conf.global.compute = conf.global.lhs = conf.global.rhs = Format(fmt);
    conf.global.lhs_bits = conf.global.rhs_bits = fmt == 2 ? 1 : 0;
    const unsigned length = 2u << chain;
    for (unsigned q = 0; q < length; ++q)
      conf.active[q] = true;
    conf.words[length - 1] = op | (mode == 4 ? 24 : 0);
    std::ofstream output(argv[argc - 1]);
    for (unsigned t = 0; t < (mode == 4 ? 100u : 50u); ++t) {
      VxmTile::Input input;
      if (t == 0)
        input.configure = conf;
      input.execute = t == 1;
      if (mode == 1 && t < 28)
        input.output_ready.fill(false);
      if (mode == 3) {
        input.phase.output = t >= 25;
        if (t == 25) {
          auto c = conf;
          c.words[0] = 16;
          input.configure = c;
        }
        if (t == 26)
          input.execute = true;
      }
      if (mode == 4) {
        input.phase = {t < 33, t >= 65, true, true};
        input.execute = t == 1 || t == 33 || t == 65;
      }
      if (mode == 5) {
        if (t == 10)
          tile.reset();
        if (t == 12)
          input.configure = conf;
        if (t == 13)
          input.execute = true;
      }
      const bool high =
          fmt == 2 &&
          (t == 3 || (mode == 2 && t == 6) || (mode == 3 && t == 28) ||
           (mode == 4 && (t == 35 || t == 67)) || (mode == 5 && t == 15));
      for (unsigned s = 0; s < 32; ++s)
        for (unsigned l = 0; l < 8; ++l) {
          const std::uint32_t value = fmt == 0   ? 0x3c00 + l
                                      : fmt == 1 ? 0x3f80 + l
                                                 : 0x3f801234 + l;
          const auto half = high ? value >> 16 : value & 65535;
          if (mode != 2 || t >= 5)
            input.stream[s][l] = std::uint8_t(half >> ((s % 2) * 8));
        }
      const auto &o = tile.tick(input);
      std::uint32_t consumed = 0, valid = 0, tail = 0;
      for (unsigned s = 0; s < 32; ++s)
        if (o.consume[s])
          consumed |= 1u << s;
      for (unsigned s = 0; s < 16; ++s) {
        if (o.valid[s])
          valid |= 1u << s;
        if (o.tail[s])
          tail |= 1u << s;
      }
      output << std::dec << t << ' ' << o.fault << ' ' << o.configure_ready
             << ' ' << o.execute_ready << ' ' << o.done << ' ' << std::hex
             << std::setfill('0') << std::setw(8) << consumed << ' '
             << std::setw(8) << valid << ' ' << std::setw(4) << tail << ' ';
      for (int s = 15; s >= 0; --s)
        for (int l = 7; l >= 0; --l)
          output << std::setw(2) << unsigned(o.bytes[s][l]);
      output << '\n';
    }
    return 0;
  }
  if (argc == 4 && std::string(argv[1]) == "cell") {
    Supercell cell;
    std::ifstream input(argv[2]);
    std::ofstream output(argv[3]);
    unsigned av, ab, wv, wb, tag, t = 0;
    std::vector<std::uint16_t> a(8), w(8);
    while (input >> std::dec >> av >> ab >> wv >> wb >> tag) {
      for (auto &x : a)
        input >> std::hex >> x;
      for (auto &x : w)
        input >> std::hex >> x;
      Supercell::Input i;
      if (av)
        i.activation = Activation{a, bool(ab), std::uint16_t(tag)};
      if (wv)
        i.weight = Weight{w, bool(wb)};
      const auto &o = cell.tick(i);
      unsigned dv = 0, pv = 0, lost = 0;
      for (unsigned n = 0; n < 8; ++n) {
        if (o.dot[n]) {
          dv |= 1 << n;
          if (o.dot[n]->alignment_lost)
            lost |= 1 << n;
        }
        if (o.partial[n])
          pv |= 1 << n;
      }
      output << std::dec << t++ << ' ' << o.fault << ' ' << o.weight_accepted
             << ' ' << std::hex << std::setw(2) << std::setfill('0') << dv
             << ' ';
      for (int n = 7; n >= 0; --n)
        output << std::setw(8) << (o.dot[n] ? o.dot[n]->bits : 0);
      output << ' ' << std::setw(2) << lost << ' ' << std::setw(2) << pv << ' ';
      for (int n = 7; n >= 0; --n)
        output << std::setw(8) << (o.partial[n] ? o.partial[n]->bits : 0);
      output << std::dec << ' ' << bool(o.weight) << ' '
             << (o.weight && o.weight->buffer) << ' ' << std::hex;
      for (int n = 7; n >= 0; --n)
        output << std::setw(4) << (o.weight ? o.weight->values[n] : 0);
      output << std::dec << ' ' << bool(o.activation) << ' '
             << (o.activation && o.activation->buffer) << ' ' << std::hex;
      for (int n = 7; n >= 0; --n)
        output << std::setw(4) << (o.activation ? o.activation->values[n] : 0);
      output << std::dec << ' ' << (o.activation ? o.activation->tag : 0) << ' '
             << (o.ready[0] | (o.ready[1] << 1)) << '\n';
    }
    return 0;
  }
  if (argc == 4 && std::string(argv[1]) == "mem") {
    const auto rows = unsigned(std::stoi(argv[2]));
    auto cfg = Config::profile(rows * 8);
    cfg.mem_depth = 64;
    MemBankColumn mem(cfg);
    std::ofstream output(argv[3]);
    const unsigned s = 32 + 3 * (rows - 1) + 2,
                   short_start = s + 32 + 2 * (rows - 1);
    for (unsigned t = 0; t < 260; ++t) {
      std::optional<std::uint32_t> q;
      if (t < 32)
        q = 0x40000001 | (t << 15) | (7 << 3);
      if (t >= s && t < s + 32)
        q = 0x40000100 | ((t - s) << 15) | (3 << 3);
      if (t >= short_start && t < short_start + 32)
        q = 0x80000001 | ((t - short_start) << 15) | (5 << 3);
      if (t >= short_start + 32 && t < short_start + 64)
        q = ((t - short_start - 32) << 15) | (5 << 3);
      if (t >= short_start + 64 && t < short_start + 80)
        q = 0x40000000 | ((t - short_start - 64) << 15) | (6 << 3);
      if (t == 245)
        q = 0;
      std::vector<MemBankColumn::Stream> stream(rows);
      for (unsigned r = 0; r < rows; ++r)
        stream[r] = {((t + r) % 13) != 0,
                     0x1020304050607080ull ^
                         (0x0101010101010101ull * (t + 17 * r))};
      if (t == 240)
        mem.reset();
      const auto o = mem.tick(q, stream);
      for (unsigned r = 0; r < rows; ++r)
        output << std::dec << t << ' ' << r << ' ' << o.executed[r].valid << ' '
               << o.read[r].valid << ' ' << std::hex << std::setw(16)
               << std::setfill('0') << (o.read[r].valid ? o.read[r].data : 0)
               << ' ' << std::dec << o.consume[r] << ' ' << o.fault[r] << ' '
               << (o.read[r].valid && o.read[r].west) << ' '
               << (o.read[r].valid ? unsigned(o.read[r].stream) : 0) << '\n';
    }
    return 0;
  }
  if (argc == 3 && std::string(argv[1]) == "core") {
    MxmCore core;
    std::ofstream output(argv[2]);
    for (unsigned t = 0; t < 175; ++t) {
      MxmCore::Input i;
      i.activation.assign(4, std::vector<std::uint16_t>(8, 0x3f80));
      i.weight.resize(4);
      for (unsigned r = 0; r < 4; ++r) {
        auto &s = i.weight[r];
        for (unsigned l = 0; l < 8; ++l) {
          s[0].push_back(
              std::uint8_t(((t - 3 * r) * 5 + l * 7 + r * 11) % 17 - 8));
          s[1].push_back(0x80);
          s[2].push_back(0x3e + ((r + l) % 3));
          s[3].push_back(std::uint8_t(int(l % 3) - 1));
          (*i.activation[r])[l] = std::uint16_t(
              0x3f00 + (((t - 3 * r) * 13 + r * 19 + l * 5) % 192));
        }
      }
      if (t == 0)
        i.parameter_enqueue = ParamLoadInstruction::decode(0b01000);
      i.parameter_launch = t == 1;
      if (t == 2)
        i.load_enqueue = LoadInstruction{};
      i.load_launch = t == 3;
      if (t >= 2 && t < 34)
        i.dequant_enqueue = DequantInstruction{};
      i.dequant_launch = t >= 3 && t < 35;
      if (t >= 39 && t < 135)
        i.compute_enqueue = ComputeInstruction{};
      i.compute_launch = t >= 40 && t < 136;
      if (t >= 62 && t < 158) {
        const unsigned n = t - 62;
        i.backend_enqueue =
            n < 32   ? BackendInstruction{}
            : n < 64 ? BackendInstruction{BackendOperation::Store,
                                          std::uint16_t(n - 32), false}
                     : BackendInstruction{BackendOperation::AccOut,
                                          std::uint16_t(n - 64), false};
      }
      i.backend_launch = t >= 63 && t < 159;
      const auto &o = core.tick(i);
      for (unsigned c = 0; c < 4; ++c) {
        const auto &top = o.top[c];
        const auto &ret = o.returned[c];
        const auto &v = o.result[c];
        output << std::dec << t << ' ' << c << ' ' << o.fault << ' '
               << bool(top) << ' ' << std::hex << std::setfill('0');
        for (int l = 7; l >= 0; --l)
          output << std::setw(8) << (top ? (*top)[l].bits : 0);
        output << ' ' << std::dec << (top ? top->front().tag : 0) << ' '
               << bool(ret) << ' ' << std::hex;
        for (int l = 7; l >= 0; --l)
          output << std::setw(8) << (ret ? (*ret)[l].bits : 0);
        output << ' ' << std::dec << (ret ? ret->front().tag : 0) << ' '
               << bool(v.values) << ' ' << std::hex;
        for (int l = 7; l >= 0; --l)
          output << std::setw(8) << (v.values ? (*v.values)[l] : 0);
        output << ' ' << std::dec << (v.values ? v.tag : 0) << ' '
               << (v.values && v.bf16) << '\n';
      }
    }
    return 0;
  }
  if (argc != 6) {
    std::cerr << "probe alu KIND INPUT OUTPUT LUT\n";
    return 2;
  }
  if (std::string(argv[1]) != "alu")
    return 2;
  std::ifstream lut(argv[5]);
  LaneLuts banks;
  for (unsigned f = 0; f < 4; ++f) {
    banks[f].configured = true;
    banks[f].minimum = f ? 0x3c00 : 0xb800;
    banks[f].width = f == 2 ? 0x2800 : 0x2400;
    for (auto &r : banks[f].row)
      lut >> std::hex >> r;
  }
  VxmAlu alu(std::stoi(argv[2]) == 1 ? 1 : 3);
  std::ifstream input(argv[3]);
  std::ofstream output(argv[4]);
  unsigned valid, format, op, t = 0;
  std::uint32_t a, b;
  while (input >> std::dec >> valid >> format >> op >> std::hex >> a >> b) {
    std::optional<VxmAlu::Request> q;
    if (valid)
      q = VxmAlu::Request{VxmOperation(op), Format(format), a, b, 0};
    const auto r = alu.tick(q, banks);
    output << std::dec << t++ << ' ' << bool(r) << ' ' << std::hex
           << std::setw(8) << std::setfill('0') << (r ? r->bits : 0) << ' '
           << std::dec << (r ? r->fault : false) << '\n';
  }
}
