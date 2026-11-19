#include "ftlpu/cmodel.hpp"
#include "ftlpu/rtl/mem.hpp"
#include "ftlpu/rtl/mxm.hpp"
#include "ftlpu/rtl/mxm_core.hpp"
#include "ftlpu/rtl/vxm.hpp"
#include "ftlpu/rtl/vxm_tile.hpp"
#include <bit>
#include <cassert>
#include <cfenv>
#include <iostream>
#include <random>
using namespace ftlpu::rtl;
int main() {
  std::fesetround(FE_TONEAREST);
  ftlpu::CModel native_model(Config::profile(32));
  assert(native_model.config().mem_depth == 2048 &&
         native_model.config().accumulator_depth == 1024);
  Macro<BackendInstruction> macro{
      3, 32, 1, {BackendOperation::Store, 0, false}, 1};
  assert(!macro.at(2));
  assert(macro.at(34)->address == 31);
  assert(!macro.at(35));
  assert((BackendInstruction{BackendOperation::AccOut, 1023, true}.encode(
              Config::profile(32)) == 0x1fff));
  assert(fp::mxm_bf16(0xff812345) == 0xffc1);
  assert(fp::mxm_bf16(0x7f800001) == 0x7fc0);
  assert(vxmath::convert(true, Format::Fp16, Format::Fp32, 0xffff).bits ==
         0xffc00000);
  assert(vxmath::convert(true, Format::Bf16, Format::Fp32, 0x8001).bits ==
         0x80000000);
  std::mt19937 rng(0x4c505531);
  for (unsigned i = 0; i < 50000; ++i) {
    const auto a = std::uint32_t(rng()), b = std::uint32_t(rng());
    if (fp::nan(a) || fp::nan(b) || fp::inf(a) || fp::inf(b))
      continue;
    const volatile float x = std::bit_cast<float>(a),
                         y = std::bit_cast<float>(b);
    const float sum = x + y, product = x * y;
    assert(fp::add(a, b) == std::bit_cast<std::uint32_t>(sum));
    assert(fp::multiply(a, b) == std::bit_cast<std::uint32_t>(product));
    if ((b & 0x7fffffff) != 0) {
      const float quotient = x / y;
      assert(fp::divide(a, b) == std::bit_cast<std::uint32_t>(quotient));
    }
  }
  const std::vector<std::uint16_t> a{0x4b80, 0x3f80, 0xcb80, 0, 0, 0, 0, 0},
      b(8, 0x3f80);
  assert(fp::dot(a, b).bits == 0x3f800000); // retain cancellation residual
  assert(fp::dequant(0x7f, 0x80, 0x3f80) ==
         0x437f); // 255, no integer->BF16 pre-round
  for (unsigned rows : {2, 4, 8}) {
    auto cfg = Config::profile(rows * 8);
    MemBankColumn m(cfg);
    for (unsigned r = 0; r < rows; ++r)
      m.initialize(r, 0, r + 17);
    MemInstruction instruction;
    instruction.delay3 = true;
    for (unsigned t = 0; t < 3 * (rows - 1) + 5; ++t) {
      const auto o =
          m.tick(t == 0 ? std::optional(instruction.encode(cfg.mem_depth))
                        : std::nullopt);
      for (unsigned r = 0; r < rows; ++r) {
        assert(o.read[r].valid == (t == 3 * r));
        if (o.read[r].valid)
          assert(o.read[r].data == r + 17);
      }
    }
    m.reset();
    const unsigned next = 1 + 2 * (rows - 1);
    for (unsigned t = 0; t < next + rows + 3; ++t) {
      instruction.delay3 = t == 0;
      const auto o =
          m.tick((t == 0 || t == next)
                     ? std::optional(instruction.encode(cfg.mem_depth))
                     : std::nullopt);
      for (unsigned r = 0; r < rows; ++r)
        assert(o.executed[r].valid == (t == 3 * r || t == next + r));
    }
    m.reset();
    assert(m.memory(0, 0) == 17);
    instruction.delay3 = true;
    m.tick(instruction.encode(cfg.mem_depth));
    instruction.delay3 = false;
    bool rejected = false;
    try {
      m.tick(instruction.encode(cfg.mem_depth));
    } catch (const std::logic_error &) {
      rejected = true;
    }
    assert(rejected);
  }
  Config cfg = Config::profile(32);
  Supercell cell(cfg);
  for (unsigned i = 0; i < 8; ++i)
    assert(cell.tick({{}, Weight{b, false}, {}}).weight_accepted);
  assert(cell.output().ready[0]);
  for (unsigned t = 0; t < 15; ++t) {
    Supercell::Input in;
    if (t == 0)
      in.activation = Activation{a, false, 91};
    const auto &o = cell.tick(in);
    for (unsigned n = 0; n < 8; ++n) {
      assert(bool(o.dot[n]) == (t == 8));
      assert(bool(o.partial[n]) == (t == 11));
      if (o.partial[n])
        assert(o.partial[n]->bits == 0x3f800000 && o.partial[n]->tag == 91);
    }
    assert(!o.fault);
  }
  for (unsigned word = 0; word < (1u << ParamLoadInstruction::bits); ++word)
    assert(ParamLoadInstruction::decode(word).encode() == word);
  for (unsigned word = 0; word < (1u << DequantInstruction::bits); ++word)
    assert(DequantInstruction::decode(word).encode() == word);

  WeightDequantizer dq(cfg);
  WeightDequantizer::Streams streams;
  streams[0].assign(8, 3);
  streams[1].assign(8, 0x80);
  streams[2].assign(8, 0x3f);
  streams[3].assign(8, 1);
  const auto explicit_zero = ParamLoadInstruction::decode(0b01000);
  const auto &prepared = dq.tick({explicit_zero}, streams);
  assert(prepared.parameter_accepted && prepared.consume[3] == 255);
  assert(dq.tick({{}, DequantInstruction{}}, streams).accepted);
  streams[3].assign(8, 2);
  const auto &updated = dq.tick({explicit_zero}, streams);
  assert(updated.parameter_accepted && !updated.weight);
  assert(!dq.tick({}).weight);
  const auto &out = dq.tick({});
  assert(out.weight && out.weight->values[0] == 0x4000); // D1 used the old zero.
  assert(!dq.tick({}).weight);

  auto odd_config = cfg;
  odd_config.terms = 3;
  odd_config.weight_zero_point_enable = false;
  odd_config.weight_parameter_streams = {5, 4, 3, 2, 1};
  WeightDequantizer odd(odd_config);
  streams[0] = {0x21, 0xf3}; // FP4 0.5, 1, 1.5. The last high nibble is unused.
  streams[5].assign(3, 127);
  const ParamLoadInstruction ue8m0{WeightScaleMode::Ue8m0,
                                   WeightZeroMode::Implicit, false};
  const auto &loaded = odd.tick({ue8m0}, streams);
  assert(loaded.parameter_accepted && loaded.consume[5] == 7 &&
         loaded.consume[4] == 0);
  const DequantInstruction fp4{WeightFormat::Fp4E2M1, false};
  const auto &packed = odd.tick({{}, fp4, true}, streams);
  assert(packed.accepted && packed.consume[0] == 3);
  auto disabled_zero = ue8m0;
  disabled_zero.zero_mode = WeightZeroMode::Int8;
  const auto &rejected = odd.tick({disabled_zero}, streams);
  assert(rejected.fault && !rejected.parameter_accepted &&
         rejected.consume[4] == 0 && odd.parameter_ready(false));
  assert(!odd.tick({}).weight);
  const auto &fp4_out = odd.tick({});
  assert(fp4_out.weight && fp4_out.weight->buffer);
  assert(fp4_out.weight->values ==
         std::vector<std::uint16_t>({0x3f00, 0x3f80, 0x3fc0}));
  Accumulator acc(cfg);
  std::vector<std::uint32_t> initial(8, 0x3f800000);
  acc.initialize(0, initial);
  acc.initialize(1, initial);
  for (unsigned t = 0; t < 8; ++t) {
    std::optional<Accumulator::Request> launch;
    if (t < 2)
      launch = Accumulator::Request{
          {BackendOperation::AccStore, std::uint16_t(t), false}, 7};
    std::optional<std::vector<Value>> v;
    if (t == 3 || t == 4)
      v = std::vector<Value>(8, Value{0x40000000, 7});
    const auto o = acc.tick(launch, v);
    assert(!o.fault);
    if (t == 6 || t == 7)
      assert(o.bank[(t - 6) % 2].valid && o.bank[(t - 6) % 2].write);
  }
  assert(acc.memory(0)[0] == 0x40400000 && acc.memory(1)[0] == 0x40400000);
  {
    MxmCore paused(cfg);
    MxmCore::Input i;
    i.run = false;
    i.compute_enqueue = ComputeInstruction{};
    assert(!paused.tick(i).launch_ready[0]);
    i = {};
    i.compute_launch = true;
    const auto &o = paused.tick(i);
    assert(!o.fault && o.compute[0]);
  }
  MxmCore core(cfg);
  std::array<unsigned, 4> result_count{};
  for (unsigned t = 0; t < 110; ++t) {
    MxmCore::Input ci;
    ci.activation.assign(4, std::vector<std::uint16_t>(8, 0x3f80));
    ci.weight.resize(4);
    for (auto &s : ci.weight) {
      s[0].assign(8, 1);
      s[1].assign(8, 0x80);
      s[2].assign(8, 0x3f);
    }
    if (t == 0)
      ci.parameter_enqueue = ParamLoadInstruction{};
    if (t == 1)
      ci.parameter_launch = true;
    if (t == 2)
      ci.load_enqueue = LoadInstruction{false, false};
    if (t == 3)
      ci.load_launch = true;
    if (t >= 2 && t < 34)
      ci.dequant_enqueue = DequantInstruction{};
    if (t >= 3 && t < 35)
      ci.dequant_launch = true;
    if (t >= 39 && t < 71)
      ci.compute_enqueue = ComputeInstruction{};
    if (t >= 40 && t < 72)
      ci.compute_launch = true;
    if (t >= 62 && t < 94)
      ci.backend_enqueue = BackendInstruction{};
    if (t >= 63 && t < 95)
      ci.backend_launch = true;
    const auto &co = core.tick(ci);
    if (co.fault) {
      std::cerr << "Core fault E" << t << '\n';
      return 1;
    }
    for (unsigned c = 0; c < 4; ++c) {
      assert(bool(co.result[c].values) == (t >= 69 + c && t < 101 + c));
      if (co.result[c].values) {
        ++result_count[c];
        for (auto v : *co.result[c].values)
          assert(v == 0x42000000);
      }
    }
  }
  for (auto n : result_count)
    assert(n == 32);
  assert(VxmLocal::decode(3, 0, 0x26).operation == VxmOperation::Log);
  bool invalid = false;
  try {
    VxmLocal::decode(3, 0, 0x27);
  } catch (const std::invalid_argument &) {
    invalid = true;
  }
  assert(invalid);
  VxmInput vi;
  assert(vi.tick(0x1234, 1));
  assert(!vi.ready());
  assert(vi.tick(0x3f80, 1));
  assert(vi.value() == 0x3f801234);
  vi.consume();
  assert(!vi.ready());
  assert(vxmath::basic(VxmOperation::Bypass, Format::Fp32, 0x7fa12345, 0) ==
         0x7fa12345);
  assert(vxmath::basic(VxmOperation::Negate, Format::Bf16, 0x7f81, 0) ==
         0xff81);
  assert(vxmath::basic(VxmOperation::Max, Format::Fp32, 0x80000000, 0) == 0);
  LaneLuts luts;
  VxmAlu alu(3);
  assert(!alu.tick(
      VxmAlu::Request{VxmOperation::Log, Format::Fp32, 0x3f800000, 0, 11},
      luts));
  for (unsigned t = 1; t < 8; ++t) {
    const auto result = alu.tick({}, luts);
    assert(bool(result) == (t == 7));
    if (result)
      assert(result->bits == 0 && result->tag == 11 && !result->fault);
  }
  for (unsigned chain = 0; chain < 3; ++chain)
    for (unsigned fmt = 0; fmt < 3; ++fmt) {
      VxmTile tile;
      VxmTile::Configuration conf;
      conf.global.chain = chain;
      conf.global.compute = conf.global.lhs = conf.global.rhs = Format(fmt);
      conf.global.lhs_bits = conf.global.rhs_bits = fmt == 2 ? 1 : 0;
      const unsigned length = 2u << chain;
      for (unsigned q = 0; q < length; ++q)
        conf.active[q] = true;
      unsigned done = 0, output_beats = 0;
      for (unsigned t = 0; t < 50; ++t) {
        VxmTile::Input vi;
        if (t == 0)
          vi.configure = conf;
        if (t == 1)
          vi.execute = true;
        for (unsigned stream = 0; stream < 32; ++stream)
          for (unsigned l = 0; l < 8; ++l) {
            const std::uint32_t value = fmt == 0   ? 0x3c00 + l
                                        : fmt == 1 ? 0x3f80 + l
                                                   : 0x3f801234 + l;
            const auto half = fmt == 2 && t == 3 ? value >> 16 : value & 65535;
            vi.stream[stream][l] = std::uint8_t(half >> ((stream % 2) * 8));
          }
        const auto &vo = tile.tick(vi);
        assert(!vo.fault);
        done += vo.done;
        for (unsigned stream = 0; stream < 16; stream += 2)
          if (vo.valid[stream]) {
            ++output_beats;
            for (unsigned l = 0; l < 8; ++l) {
              const auto half = unsigned(vo.bytes[stream][l]) |
                                (unsigned(vo.bytes[stream + 1][l]) << 8);
              const auto value = fmt == 0   ? 0x3c00 + l
                                 : fmt == 1 ? 0x3f80 + l
                                            : 0x3f801234 + l;
              assert(half == (fmt == 2 && t == 2 * length + 4 ? value >> 16
                                                              : value & 65535));
            }
          }
      }
      assert(done == 1);
      assert(output_beats == (fmt == 2 ? 4 : 2));
      assert(tile.idle());
    }
  std::cout << "RTL_NATIVE_UNIT PASS float_pairs=50000 profiles=3\n";
}
