#pragma once
#include "ftlpu/rtl/config.hpp"
#include "ftlpu/rtl/float_bits.hpp"
#include "ftlpu/rtl/pipeline.hpp"
#include <algorithm>
#include <array>
#include <deque>
#include <optional>
#include <vector>

namespace ftlpu::rtl {
struct Value {
  std::uint32_t bits = 0;
  std::uint16_t tag = 0;
  bool alignment_lost = false;
};
struct Activation {
  std::vector<std::uint16_t> values;
  bool buffer = false;
  std::uint16_t tag = 0;
};
struct Weight {
  std::vector<std::uint16_t> values;
  bool buffer = false;
};
class Supercell {
public:
  struct Input {
    std::optional<Activation> activation;
    std::optional<Weight> weight;
    std::vector<std::optional<Value>> below;
  };
  struct Output {
    std::optional<Activation> activation;
    std::optional<Weight> weight;
    std::vector<std::optional<Value>> dot, partial;
    std::array<bool, 2> ready{};
    bool weight_accepted = false, fault = false;
  };
  explicit Supercell(Config c = {}, bool bottom = true, bool last = false)
      : c_(c), bottom_(bottom), last_(last) {
    c_.validate();
    for (auto &b : weights_)
      b.assign(c.outputs, std::vector<std::uint16_t>(c.terms));
    for (auto &b : valid_)
      b.assign(c.outputs, false);
    for (std::size_t n = 0; n < c.outputs; ++n) {
      dot_.emplace_back(9);
      add_.emplace_back(3);
    }
    out_.dot.resize(c.outputs);
    out_.partial.resize(c.outputs);
  }
  const Output &output() const { return out_; }
  void reset() {
    for (auto &b : valid_)
      std::fill(b.begin(), b.end(), false);
    for (auto &p : dot_)
      p.reset();
    for (auto &p : add_)
      p.reset();
    out_ = Output{};
    out_.dot.resize(c_.outputs);
    out_.partial.resize(c_.outputs);
  }
  const Output &tick(const Input &in) {
    Output next;
    next.fault = out_.fault;
    next.dot.resize(c_.outputs);
    next.partial.resize(c_.outputs);
    if (in.activation && in.activation->values.size() != c_.terms)
      throw std::invalid_argument("activation shape");
    if (in.weight && in.weight->values.size() != c_.terms)
      throw std::invalid_argument("weight shape");
    next.activation = in.activation;
    for (std::size_t n = 0; n < c_.outputs; ++n) {
      const auto d = dot_[n].output();
      const auto b = n < in.below.size() ? in.below[n] : std::optional<Value>{};
      const bool mismatch =
          !bottom_ && (bool(d) != bool(b) || (d && b && d->tag != b->tag));
      next.fault = next.fault || mismatch;
      std::optional<Value> merged;
      if (d && !mismatch)
        merged = Value{fp::add(d->bits, bottom_ ? 0 : b->bits), d->tag, false};
      next.partial[n] = add_[n].tick(merged);
      std::optional<Value> local;
      if (in.activation) {
        const auto &a = *in.activation;
        if (!valid_[a.buffer][n])
          next.fault = true;
        else {
          const auto v = fp::dot(a.values, weights_[a.buffer][n]);
          local = Value{v.bits, a.tag, v.alignment_lost};
        }
      }
      next.dot[n] = dot_[n].tick(local);
    }
    if (in.weight) {
      const auto &w = *in.weight;
      next.weight_accepted =
          !in.activation || in.activation->buffer != w.buffer;
      if (!next.weight_accepted)
        next.fault = true;
      else {
        if (!last_ && valid_[w.buffer].back())
          next.weight = Weight{weights_[w.buffer].back(), w.buffer};
        for (std::size_t n = c_.outputs - 1; n; --n) {
          weights_[w.buffer][n] = weights_[w.buffer][n - 1];
          valid_[w.buffer][n] = valid_[w.buffer][n - 1];
        }
        weights_[w.buffer][0] = w.values;
        valid_[w.buffer][0] = true;
      }
    }
    for (unsigned b = 0; b < 2; ++b)
      next.ready[b] = std::all_of(valid_[b].begin(), valid_[b].end(),
                                  [](bool v) { return v; });
    out_ = std::move(next);
    return out_;
  }

private:
  Config c_;
  bool bottom_, last_;
  std::array<std::vector<std::vector<std::uint16_t>>, 2> weights_;
  std::array<std::vector<bool>, 2> valid_;
  std::vector<Pipeline<Value>> dot_, add_;
  Output out_;
};

// D1 samples the selected bytes and snapshots the operands. CModel computes
// the exact payload at D1 and delays it through D4; the receiving Supercell
// uses that output on the next edge (D5).
class WeightDequantizer {
public:
  using Bytes = std::vector<std::optional<std::uint8_t>>;
  using Streams = std::array<Bytes, Config::streams>;
  using Consume = std::array<std::uint32_t, Config::streams>;

  struct Input {
    std::optional<ParamLoadInstruction> load;
    std::optional<DequantInstruction> dequant;
    bool weight_buffer = false;
  };
  struct Output {
    bool accepted = false, parameter_accepted = false, fault = false;
    std::optional<Weight> weight;
    Consume consume{};
  };
  explicit WeightDequantizer(Config c = {})
      : c_(c), pipe_(Config::dequant_stages - 1) {
    c_.validate();
  }
  void reset() {
    parameters_ = {};
    flush();
  }
  void flush() {
    pipe_.reset();
    out_ = {};
  } // run=0 retains parameter buffers
  const Output &output() const { return out_; }
  bool parameter_ready(bool buffer) const { return bool(parameters_[buffer]); }

  const Output &tick(const Input &in, const Streams &streams = {}) {
    Output o;
    o.fault = out_.fault;
    const bool collision = in.dequant && in.load &&
                           in.load->parameter_buffer == in.dequant->parameter_buffer;
    std::optional<Weight> w;
    if (in.dequant) {
      const auto format = in.dequant->format;
      const auto &p = parameters_[in.dequant->parameter_buffer];
      const bool packed = format == WeightFormat::Fp4E2M1;
      const auto required = packed ? (c_.terms + 1) / 2 : c_.terms;
      const auto &q = streams[c_.weight_q_stream];
      o.accepted = !collision && p && format_matches(format, *p) &&
                   available(q, required);
      if (o.accepted) {
        o.consume[c_.weight_q_stream] = lane_mask(required);
        Weight x;
        x.buffer = in.weight_buffer;
        x.values.reserve(c_.terms);
        for (std::size_t lane = 0; lane < c_.terms; ++lane) {
          const auto raw = packed ? (*q[lane / 2] >> ((lane % 2) * 4)) & 15
                                  : *q[lane];
          const auto &entry = p->lanes[lane];
          x.values.push_back(fp::dequant(format, raw, entry.zero, p->scale_mode,
                                         entry.scale, entry.block_scale));
        }
        w = std::move(x);
      }
    }
    if (in.load && !collision)
      o.parameter_accepted = load(*in.load, streams, o.consume);
    o.fault |= (in.dequant && !o.accepted) || (in.load && !o.parameter_accepted);
    o.weight = pipe_.tick(std::move(w));
    out_ = std::move(o);
    return out_;
  }

private:
  struct LaneParameters {
    std::uint32_t scale = 0;
    std::uint8_t block_scale = 0, zero = 0;
  };
  struct Parameters {
    WeightScaleMode scale_mode;
    bool has_zero;
    std::vector<LaneParameters> lanes;
  };
  static bool available(const Bytes &bytes, std::size_t count) {
    return bytes.size() >= count &&
           std::all_of(bytes.begin(), bytes.begin() + count,
                       [](const auto &byte) { return byte.has_value(); });
  }
  static std::uint32_t lane_mask(std::size_t count) {
    return std::uint32_t((std::uint64_t{1} << count) - 1);
  }
  static bool format_matches(WeightFormat format, const Parameters &p) {
    switch (format) {
    case WeightFormat::Int8:
      return p.scale_mode != WeightScaleMode::NvFp4;
    case WeightFormat::Fp4E2M1:
      return !p.has_zero;
    case WeightFormat::Fp8E4M3Fn:
      return !p.has_zero && p.scale_mode != WeightScaleMode::NvFp4;
    default:
      return false;
    }
  }
  bool load(ParamLoadInstruction command, const Streams &streams,
            Consume &consume) {
    const bool explicit_zero = command.zero_mode == WeightZeroMode::Int8;
    const bool block = command.scale_mode == WeightScaleMode::NvFp4;
    if (command.zero_mode != WeightZeroMode::Implicit &&
        (!explicit_zero || !c_.weight_zero_point_enable || block))
      return false;
    unsigned scale_bytes;
    switch (command.scale_mode) {
    case WeightScaleMode::Bf16: scale_bytes = 2; break;
    case WeightScaleMode::Ue8m0: scale_bytes = 1; break;
    case WeightScaleMode::Fp32:
    case WeightScaleMode::NvFp4: scale_bytes = 4; break;
    default: return false;
    }
    const auto count = scale_bytes + (explicit_zero || block);
    const auto &bindings = c_.weight_parameter_streams;
    for (unsigned byte = 0; byte < count; ++byte)
      if (!available(streams[bindings[byte]], c_.terms))
        return false;
    Parameters p{command.scale_mode, explicit_zero,
                 std::vector<LaneParameters>(c_.terms)};
    for (std::size_t lane = 0; lane < c_.terms; ++lane) {
      auto &entry = p.lanes[lane];
      for (unsigned byte = 0; byte < scale_bytes; ++byte)
        entry.scale |= std::uint32_t(*streams[bindings[byte]][lane]) << (8 * byte);
      if (command.scale_mode == WeightScaleMode::Bf16)
        entry.scale <<= 16;
      if (explicit_zero)
        entry.zero = *streams[bindings[scale_bytes]][lane];
      if (block)
        entry.block_scale = *streams[bindings[scale_bytes]][lane];
    }
    for (unsigned byte = 0; byte < count; ++byte)
      consume[bindings[byte]] = lane_mask(c_.terms);
    parameters_[command.parameter_buffer] = std::move(p);
    return true;
  }

  Config c_;
  std::array<std::optional<Parameters>, 2> parameters_{};
  Pipeline<Weight> pipe_;
  Output out_;
};

// An explicit command is launched per edge. Queued commands and launches are
// separate in the core wrapper; the backend never delays a command to await A.
class Accumulator {
public:
  struct Request {
    BackendInstruction instruction;
    std::optional<std::uint16_t> tag;
  };
  struct BankEvent {
    bool valid = false, write = false;
    std::size_t row = 0;
  };
  struct Output {
    std::optional<std::vector<std::uint32_t>> values;
    bool bf16 = false, fault = false;
    std::uint16_t address = 0, tag = 0;
    std::array<BankEvent, 2> bank{};
  };
  explicit Accumulator(Config c = {})
      : c_(c),
        memory_(c.accumulator_depth, std::vector<std::uint32_t>(c.outputs)),
        initialized_(c.accumulator_depth, false) {
    c_.validate();
  }
  void flush() {
    pending_.clear();
    cycle_ = 0;
    fault_ = false;
  }
  void reset() {
    flush();
    std::fill(initialized_.begin(), initialized_.end(), false);
  }
  void initialize(std::size_t a, const std::vector<std::uint32_t> &v) {
    check(v);
    memory_.at(a) = v;
    initialized_.at(a) = true;
  }
  const auto &memory(std::size_t a) const { return memory_.at(a); }
  Output tick(std::optional<Request> launch = {},
              std::optional<std::vector<Value>> partial = {}) {
    Output o;
    o.fault = fault_;
    if (launch) {
      launch->instruction.encode(c_);
      pending_.push_back({cycle_, *launch});
    }
    Pending *first = nullptr, *writeback = nullptr, *capture = nullptr;
    for (auto &p : pending_) {
      const auto age = cycle_ - p.edge;
      if (age == 1)
        first = &p;
      if (age == 3)
        capture = &p;
      if (age == 6 && p.valid && writes(p.request.instruction.operation))
        writeback = &p;
    }
    const bool collision = first &&
                           accumulates(first->request.instruction.operation) &&
                           writeback &&
                           first->request.instruction.address % 2 ==
                               writeback->request.instruction.address % 2;
    bool hazard = false;
    if (first && (accumulates(first->request.instruction.operation) ||
                  writes(first->request.instruction.operation)))
      for (const auto &p : pending_)
        if (&p != first && p.reserved && cycle_ - p.edge <= 6 &&
            p.request.instruction.address == first->request.instruction.address)
          hazard = true;
    o.fault =
        o.fault || collision || hazard || (bool(capture) != bool(partial));
    for (auto &p : pending_) {
      const auto age = cycle_ - p.edge;
      const auto cmd = p.request.instruction;
      const bool acc = accumulates(cmd.operation);
      if (age == 1) {
        p.ok = !collision && !hazard;
        if (acc && p.ok) {
          access(o, cmd.address, false);
          if (!initialized_[cmd.address]) {
            o.fault = true;
            p.ok = false;
          } else
            p.b = memory_[cmd.address];
        }
        p.reserved = p.ok && writes(cmd.operation);
      }
      // A is available after L2 and is sampled at L3. The C++ tick
      // represents that sampling edge, exactly like the SV adder A1.
      if (age == 3) {
        if (!partial || partial->size() != c_.outputs) {
          o.fault = true;
          continue;
        }
        if (!p.ok)
          continue;
        p.result.resize(c_.outputs);
        p.valid = true;
        if (!p.request.tag)
          p.request.tag = partial->front().tag;
        for (std::size_t n = 0; n < c_.outputs; ++n) {
          if ((*partial)[n].tag != *p.request.tag) {
            o.fault = true;
            p.valid = false;
          }
          p.result[n] =
              acc ? fp::add((*partial)[n].bits, p.b[n]) : (*partial)[n].bits;
        }
      }
      if (age == 6 && p.valid) {
        if (cmd.operation == BackendOperation::Store ||
            cmd.operation == BackendOperation::AccStore) {
          if (!collision && access(o, cmd.address, true)) {
            memory_[cmd.address] = p.result;
            initialized_[cmd.address] = true;
          }
        } else {
          if (o.values)
            o.fault = true;
          o.values = p.result;
          o.bf16 = cmd.bf16;
          o.address = cmd.address;
          o.tag = p.request.tag.value_or(0);
          if (cmd.bf16)
            for (auto &x : *o.values)
              x = fp::mxm_bf16(x);
        }
      }
    }
    while (!pending_.empty() && cycle_ - pending_.front().edge >= 6)
      pending_.pop_front();
    fault_ = o.fault;
    ++cycle_;
    return o;
  }

private:
  struct Pending {
    std::size_t edge;
    Request request;
    std::vector<std::uint32_t> b, result;
    bool valid = false, ok = false, reserved = false;
  };
  static bool accumulates(BackendOperation op) {
    return op == BackendOperation::AccStore || op == BackendOperation::AccOut;
  }
  static bool writes(BackendOperation op) {
    return op == BackendOperation::Store || op == BackendOperation::AccStore;
  }
  void check(const std::vector<std::uint32_t> &v) const {
    if (v.size() != c_.outputs)
      throw std::invalid_argument("accumulator batch shape");
  }
  bool access(Output &o, std::size_t address, bool write) {
    auto &b = o.bank[address % 2];
    if (b.valid) {
      o.fault = true;
      return false;
    }
    b = {true, write, address / 2};
    return true;
  }
  Config c_;
  std::vector<std::vector<std::uint32_t>> memory_;
  std::vector<bool> initialized_;
  std::deque<Pending> pending_;
  std::size_t cycle_ = 0;
  bool fault_ = false;
};
} // namespace ftlpu::rtl
