#pragma once
#include "ftlpu/rtl/vxm.hpp"
#include <bitset>
namespace ftlpu::rtl {
class VxmTile {
public:
  struct Configuration {
    VxmGlobalConfig global;
    std::array<bool, 8> active{};
    std::array<std::uint8_t, 8> words{};
  };
  struct Phase {
    bool first = true, last = true, accumulate = false, output = true;
  };
  using Vector = std::array<std::uint32_t, 8>;
  struct Input {
    std::optional<Configuration> configure;
    bool execute = false;
    Phase phase;
    std::array<std::optional<std::uint32_t>, 8> immediate{};
    std::array<std::array<std::optional<std::uint8_t>, 8>, 32> stream{};
    std::array<bool, 16> output_ready;
    Input() { output_ready.fill(true); }
  };
  struct Output {
    std::array<bool, 32> consume{};
    std::array<bool, 16> valid{};
    std::array<std::array<std::uint8_t, 8>, 16> bytes{};
    std::array<std::optional<Vector>, 16> tail{};
    bool execute_ready = false, configure_ready = true, done = false,
         fault = false;
  };
  VxmTile() {
    for (unsigned s = 0; s < 16; ++s)
      for (unsigned l = 0; l < 8; ++l)
        alu_.emplace_back(s);
  }
  void reset() {
    for (auto &a : alu_)
      a.reset();
    for (auto &a : lhs_)
      for (auto &b : a)
        b.reset();
    for (auto &a : rhs_)
      for (auto &b : a)
        b.reset();
    pending_ = {};
    metadata_ = {};
    completed_ = {};
    token_ = {};
    feedback_ = {};
    accumulator_ = {};
    output_ = {};
    output_high_ = {};
    phase_ = {};
    out_ = {};
    configured_ = false;
    fault_ = false;
    for (auto &lane : luts_)
      for (auto &bank : lane) {
        bank.configured = false;
        bank.minimum = 0;
        bank.width = 0x3c00;
      }
  }
  bool idle() const {
    for (unsigned s = 0; s < 16; ++s)
      if (pending_[s] || metadata_[s] || completed_[s] || token_[s])
        return false;
    for (const auto &o : output_)
      if (o)
        return false;
    return true;
  }
  void configure_lut(unsigned bank, unsigned row, std::uint64_t value,
                     std::uint16_t minimum, std::uint16_t width) {
    if (bank >= 4 || row >= 64)
      throw std::out_of_range("LUT coordinate");
    if (!idle())
      throw std::logic_error("LUT configuration while execution is active");
    if ((bank == 0 && value > 0x3ffffff) || (bank == 3 && value >> 50))
      throw std::invalid_argument("LUT row width");
    for (auto &lane : luts_) {
      lane[bank].configured = true;
      lane[bank].row[row] = value;
      lane[bank].minimum = minimum;
      lane[bank].width = width;
    }
  }
  const LaneLuts &lane_luts(unsigned lane) const { return luts_.at(lane); }
  const Output &output() const { return out_; }
  const Output &tick(const Input &in) {
    Output o;
    o.fault = fault_;
    const auto before_token = token_, before_feedback = feedback_;
    auto next_token = token_, next_feedback = feedback_;
    auto next_accumulator = accumulator_;
    auto next_pending = pending_;
    auto next_metadata = metadata_;
    auto next_completed = completed_;
    std::array<std::optional<Token>, 16> forwarding{}, feedback_write{};
    std::array<std::array<bool, 8>, 4> lut_port{};
    const unsigned length = 2u << configuration_.global.chain;
    int tail_stage = -1;
    for (unsigned s = 0; s < 16; ++s)
      if (configuration_.active[s % 8] && s % length == length - 1)
        tail_stage = int(s);
    const bool launch_ready = configured_ &&
                              std::none_of(pending_.begin(), pending_.end(),
                                           [](bool v) { return v; }) &&
                              tail_stage >= 0 && !in.configure;
    const bool config_ready = idle();
    // Consume the PRE-edge output beat atomically. FP32 uses the same
    // pair for the high half on the next edge, under independent ready.
    for (unsigned block = 0; block < 8; ++block)
      if (output_[block] && in.output_ready[2 * block] &&
          in.output_ready[2 * block + 1]) {
        if (configuration_.global.compute == Format::Fp32 &&
            !output_high_[block])
          output_high_[block] = true;
        else {
          output_[block].reset();
          output_high_[block] = false;
        }
      }
    for (unsigned s = 0; s < 16; ++s) {
      std::optional<VxmLocal> decoded;
      if (configured_ && configuration_.active[s % 8]) {
        try {
          decoded = VxmLocal::decode(s, configuration_.global.chain,
                                     configuration_.words[s % 8]);
        } catch (const std::invalid_argument &) {
          o.fault = true;
        }
      }
      const auto old_complete = completed_[s];
      if (old_complete) {
        const auto &c = *old_complete;
        next_metadata[s].reset();
        next_completed[s].reset();
        if (c.acc_write)
          next_accumulator[s] = c.token.value;
        const bool emit = !c.acc_write || c.phase.last;
        if (decoded && emit) {
          if (!decoded->tail) {
            forwarding[s + 1] = c.token;
          } else if (c.phase.output && c.phase.last) {
            if (output_[s / 2])
              o.fault = true;
            output_[s / 2] = c.token.value;
            output_high_[s / 2] = false;
          } else
            feedback_write[s] = c.token;
        }
      }
      std::array<std::optional<VxmAlu::Request>, 8> requests;
      std::optional<Metadata> accepted;
      if (decoded && pending_[s] && (!metadata_[s] || old_complete)) {
        Token t;
        Vector rhs{};
        bool available = true;
        const auto &d = *decoded;
        const unsigned block = s / 2;
        const unsigned feedback_stage = s + length - 1;
        for (unsigned l = 0; l < 8; ++l) {
          const auto immediate = in.immediate[s % 8];
          auto get_immediate = [&]() {
            if (!immediate)
              available = false;
            return immediate.value_or(0);
          };
          if (d.head) {
            if (d.lhs == OperandSource::Feedback) {
              if (!before_feedback[feedback_stage])
                available = false;
              else {
                t.value[l] = before_feedback[feedback_stage]->value[l];
                t.original[l] = before_feedback[feedback_stage]->original[l];
                t.auxiliary[l] = before_feedback[feedback_stage]->auxiliary[l];
              }
            } else if (d.lhs == OperandSource::Stream) {
              if (!lhs_[block][l].ready())
                available = false;
              else
                t.value[l] =
                    convert(lhs_[block][l].value(), configuration_.global.lhs,
                            configuration_.global.compute, o.fault, available);
            } else
              t.value[l] = get_immediate();
            if (d.rhs == OperandSource::Stream) {
              if (!rhs_[block][l].ready())
                available = false;
              else
                rhs[l] =
                    convert(rhs_[block][l].value(), configuration_.global.rhs,
                            configuration_.global.compute, o.fault, available);
            } else
              rhs[l] = get_immediate();
            if (d.lhs != OperandSource::Feedback) {
              t.original[l] = t.value[l];
              t.auxiliary[l] = rhs[l];
            }
          } else {
            if (!before_token[s]) {
              available = false;
              continue;
            }
            t = *before_token[s];
            if (d.rhs == OperandSource::Original)
              rhs[l] = t.original[l];
            else if (d.rhs == OperandSource::Auxiliary)
              rhs[l] = t.auxiliary[l];
            else if (d.rhs == OperandSource::Immediate)
              rhs[l] = get_immediate();
            else if (acc_write(s, d, phase_[s]) && phase_[s].first)
              rhs[l] = d.operation == VxmOperation::Max
                           ? (configuration_.global.compute == Format::Fp32
                                  ? 0xff800000
                              : configuration_.global.compute == Format::Bf16
                                  ? 0xff80
                                  : 0xfc00)
                           : 0;
            else {
              if (!accumulator_[s])
                available = false;
              else
                rhs[l] = (*accumulator_[s])[l];
            }
          }
          if (!alu_[s * 8 + l].ready())
            available = false;
        }
        if (available) {
          if (unsigned(d.operation) > 5) {
            const unsigned function =
                s % 4 == 1                              ? 0
                : d.operation == VxmOperation::Special0 ? 1
                : d.operation == VxmOperation::Special1 ? 2
                                                        : 3;
            for (unsigned l = 0; l < 8; ++l)
              if (luts_[l][function].configured &&
                  VxmArithmetic::requires_lut(
                      function, configuration_.global.compute, t.value[l])) {
                if (lut_port[function][l])
                  throw std::logic_error(
                      "two stages request the same function/lane LUT port; "
                      "compiler schedule is illegal");
                lut_port[function][l] = true;
              }
          }
          t.end = d.head ? (s == unsigned(tail_stage + 1 - length) &&
                            phase_[s].last)
                         : t.end;
          accepted = Metadata{t, phase_[s], acc_write(s, d, phase_[s])};
          for (unsigned l = 0; l < 8; ++l)
            requests[l] =
                VxmAlu::Request{d.operation, configuration_.global.compute,
                                t.value[l], rhs[l], 0};
          next_pending[s] = false;
          if (!d.head)
            next_token[s].reset();
          else {
            if (d.lhs == OperandSource::Feedback)
              next_feedback[feedback_stage].reset();
            for (auto &x : lhs_[block])
              x.reset();
            for (auto &x : rhs_[block])
              x.reset();
          }
        }
      }
      if (accepted)
        next_metadata[s] = accepted;
      std::optional<Metadata> completed;
      for (unsigned l = 0; l < 8; ++l) {
        const auto result = alu_[s * 8 + l].tick(requests[l], luts_[l]);
        if (result) {
          o.fault = o.fault || result->fault;
          const auto meta = accepted ? accepted : metadata_[s];
          if (!meta) {
            o.fault = true;
            continue;
          }
          if (!completed)
            completed = meta;
          completed->token.value[l] = result->bits;
        }
      }
      if (completed) {
        next_completed[s] = completed;
        if (decoded && decoded->tail &&
            (!completed->acc_write || completed->phase.last))
          o.tail[s] = completed->token.value;
        if (int(s) == tail_stage && completed->token.end)
          o.done = true;
      }
      // Capture only when a pending chain head needs that source. The
      // ready bit becomes visible after this edge, so it cannot execute
      // against the same-edge byte capture.
      if (decoded && decoded->head && pending_[s] && !accepted) {
        const unsigned block = s / 2;
        if (decoded->lhs == OperandSource::Stream)
          capture(in, block * 4, configuration_.global.lhs_bits, lhs_[block],
                  o);
        if (decoded->rhs == OperandSource::Stream)
          capture(in, block * 4 + 2, configuration_.global.rhs_bits,
                  rhs_[block], o);
      }
    }
    for (unsigned s = 0; s < 16; ++s) {
      if (forwarding[s]) {
        if (next_token[s])
          o.fault = true;
        next_token[s] = forwarding[s];
      }
      if (feedback_write[s]) {
        if (next_feedback[s])
          o.fault = true;
        next_feedback[s] = feedback_write[s];
      }
    }
    token_ = std::move(next_token);
    feedback_ = std::move(next_feedback);
    accumulator_ = std::move(next_accumulator);
    pending_ = next_pending;
    metadata_ = std::move(next_metadata);
    completed_ = std::move(next_completed);
    if (in.configure) {
      if (!config_ready)
        o.fault = true;
      else {
        configuration_ = *in.configure;
        configuration_.global.encode();
        configured_ = true;
        const auto &g = configuration_.global;
        if (g.lhs_bits != (g.lhs == Format::Fp32 ? 1 : 0) ||
            g.rhs_bits != (g.rhs == Format::Fp32 ? 1 : 0))
          o.fault = true;
        for (auto &block : lhs_)
          for (auto &l : block)
            l.reset();
        for (auto &block : rhs_)
          for (auto &l : block)
            l.reset();
      }
    }
    if (in.execute) {
      if (!launch_ready)
        o.fault = true;
      else
        for (unsigned s = 0; s < 16; ++s) {
          pending_[s] = configuration_.active[s % 8];
          phase_[s] = in.phase;
        }
    }
    for (unsigned block = 0; block < 8; ++block)
      if (output_[block]) {
        o.valid[block * 2] = o.valid[block * 2 + 1] = true;
        for (unsigned l = 0; l < 8; ++l) {
          const auto value =
              (*output_[block])[l] >> (output_high_[block] ? 16 : 0);
          o.bytes[block * 2][l] = std::uint8_t(value);
          o.bytes[block * 2 + 1][l] = std::uint8_t(value >> 8);
        }
      }
    bool has_tail = false;
    const auto new_length = 2u << configuration_.global.chain;
    for (unsigned s = 0; s < 16; ++s)
      has_tail = has_tail || (configuration_.active[s % 8] &&
                              s % new_length == new_length - 1);
    fault_ = o.fault;
    o.configure_ready = idle();
    o.execute_ready = configured_ && !in.configure &&
                      std::none_of(pending_.begin(), pending_.end(),
                                   [](bool v) { return v; }) &&
                      has_tail;
    out_ = std::move(o);
    return out_;
  }

private:
  struct Token {
    Vector value{}, original{}, auxiliary{};
    bool end = false;
  };
  struct Metadata {
    Token token;
    Phase phase;
    bool acc_write = false;
  };
  static bool acc_write(unsigned s, const VxmLocal &d, Phase phase) {
    return phase.accumulate && (s % 2) &&
           (d.operation == VxmOperation::Add ||
            d.operation == VxmOperation::Max) &&
           d.rhs == OperandSource::Accumulator;
  }
  static std::uint32_t convert(std::uint32_t value, Format from, Format to,
                               bool &fault, bool &available) {
    const auto c = vxmath::convert(true, from, to, value);
    fault = fault || c.fault;
    available = available && c.valid;
    return c.bits;
  }
  static void capture(const Input &in, unsigned stream, unsigned bits,
                      std::array<VxmInput, 8> &values, Output &o) {
    bool available = true;
    for (unsigned l = 0; l < 8; ++l)
      available = available && in.stream[stream][l] && in.stream[stream + 1][l];
    if (!available)
      return;
    bool consumed = false;
    for (unsigned l = 0; l < 8; ++l)
      consumed =
          values[l].tick(std::uint16_t(*in.stream[stream][l]) |
                             (std::uint16_t(*in.stream[stream + 1][l]) << 8),
                         bits) ||
          consumed;
    if (consumed)
      o.consume[stream] = o.consume[stream + 1] = true;
  }
  Configuration configuration_{};
  bool configured_ = false, fault_ = false;
  std::array<LaneLuts, 8> luts_{};
  std::vector<VxmAlu> alu_;
  std::array<std::array<VxmInput, 8>, 8> lhs_, rhs_;
  std::array<bool, 16> pending_{};
  std::array<Phase, 16> phase_{};
  std::array<std::optional<Token>, 16> token_, feedback_;
  std::array<std::optional<Metadata>, 16> metadata_, completed_;
  std::array<std::optional<Vector>, 16> accumulator_;
  std::array<std::optional<Vector>, 8> output_{};
  std::array<bool, 8> output_high_{};
  Output out_;
};
} // namespace ftlpu::rtl
