#pragma once
#include "ftlpu/rtl/mxm.hpp"
namespace ftlpu::rtl {
using Batch = std::vector<Value>;
class ResultReturn {
public:
  explicit ResultReturn(Config c = {})
      : c_(c), down_(c.rows * c.columns), left_(c.rows * c.columns) {
    c_.validate();
  }
  void reset() {
    for (auto *a : {&down_, &left_})
      for (auto &x : *a)
        x.reset();
  }
  std::vector<std::optional<Batch>> output() const {
    std::vector<std::optional<Batch>> out(c_.rows);
    for (std::size_t r = 0; r < c_.rows; ++r)
      out[r] = left_[r * c_.columns];
    return out;
  }
  void tick(const std::vector<std::optional<Batch>> &top) {
    auto down = down_, left = left_;
    for (std::size_t r = 0; r < c_.rows; ++r)
      for (std::size_t c = 0; c < c_.columns; ++c) {
        const auto p = r * c_.columns + c;
        if (r > c)
          down[p] = r == c_.rows - 1 ? top.at(c) : down_[p + c_.columns];
        if (r == c)
          left[p] = r == c_.rows - 1 ? top.at(c) : down_[p + c_.columns];
        else if (r > c)
          left[p] = left_[p + 1];
      }
    down_ = std::move(down);
    left_ = std::move(left);
  }
  const auto &down_links() const { return down_; }
  const auto &left_links() const { return left_; }

private:
  Config c_;
  std::vector<std::optional<Batch>> down_, left_;
};
template <class T> class CommandWave {
public:
  CommandWave(std::size_t rows, std::size_t interval)
      : rows_(rows), interval_(interval), stages_(1 + (rows - 1) * interval) {}
  const std::optional<T> &row(std::size_t r) const {
    return stages_.at(r * interval_);
  }
  void tick(std::optional<T> v = {}) {
    for (std::size_t s = stages_.size() - 1; s; --s)
      stages_[s] = std::move(stages_[s - 1]);
    stages_[0] = std::move(v);
  }
  void reset() {
    for (auto &s : stages_)
      s.reset();
  }

private:
  std::size_t rows_, interval_;
  std::vector<std::optional<T>> stages_;
};
// Full native array -> downward/left return -> independent accumulator banks.
// Inputs are the same pre-edge ports as lpu_mxm_core. Enqueue is not launch;
// enqueue+launch on an empty queue cannot execute the new word on that edge.
class MxmCore {
public:
  using Streams = WeightDequantizer::Streams;
  struct Input {
    bool run = true;
    std::optional<ComputeInstruction> compute_enqueue;
    std::optional<LoadInstruction> load_enqueue;
    std::optional<ParamLoadInstruction> parameter_enqueue;
    std::optional<DequantInstruction> dequant_enqueue;
    std::optional<BackendInstruction> backend_enqueue;
    bool compute_launch = false, load_launch = false, parameter_launch = false,
         dequant_launch = false, backend_launch = false;
    std::vector<std::optional<std::vector<std::uint16_t>>> activation;
    std::vector<Streams> weight;
  };
  struct Output {
    std::vector<std::optional<ComputeInstruction>> compute;
    std::vector<std::optional<LoadInstruction>> load;
    std::vector<std::optional<ParamLoadInstruction>> parameter;
    std::vector<std::optional<DequantInstruction>> dequant;
    // Per row, per stream, one bit for each consumed lane on this edge.
    std::vector<WeightDequantizer::Consume> weight_consume;
    std::vector<std::optional<Batch>> top, returned;
    std::vector<Accumulator::Output> result;
    // Ordering: compute, weight load, parameter load, dequant, backend.
    std::array<bool, 5> enqueue_ready{}, launch_ready{};
    bool fault = false;
  };
  explicit MxmCore(Config c = {})
      : c_(c), compute_(c.rows, Config::compute_hop), load_(c.rows, Config::compute_hop),
        parameter_(c.rows, Config::compute_hop), dequant_(c.rows, Config::compute_hop),
        backend_(c.columns, Config::result_hop), returns_(c),
        remaining_(c.rows), load_word_(c.rows) {
    c_.validate();
    for (std::size_t r = 0; r < c.rows; ++r) {
      dequantizers_.emplace_back(c);
      for (std::size_t col = 0; col < c.columns; ++col)
        cells_.emplace_back(c, r == 0, col == c.columns - 1);
    }
    for (std::size_t col = 0; col < c.columns; ++col)
      accumulators_.emplace_back(c);
    reset();
  }
  void reset() { flush(true); }
  void flush(bool reset_storage = false) {
    if (reset_storage) {
      parameter_queue_.clear();
      dequant_queue_.clear();
      compute_queue_.clear();
      load_queue_.clear();
      backend_queue_.clear();
    }
    compute_.reset();
    load_.reset();
    parameter_.reset();
    dequant_.reset();
    backend_.reset();
    returns_.reset();
    for (auto &c : cells_)
      c.reset();
    for (auto &d : dequantizers_) {
      if (reset_storage)
        d.reset();
      else
        d.flush();
    }
    for (auto &a : accumulators_) {
      if (reset_storage)
        a.reset();
      else
        a.flush();
    }
    std::fill(remaining_.begin(), remaining_.end(), 0);
    tag_ = 0;
    out_ = {};
    out_.top.resize(c_.columns);
    out_.returned.resize(c_.rows);
    out_.result.resize(c_.columns);
    out_.parameter.resize(c_.rows);
    out_.dequant.resize(c_.rows);
    out_.weight_consume.resize(c_.rows);
    readiness(out_, false);
  }
  const Output &output() const { return out_; }
  Accumulator &accumulator(std::size_t c) { return accumulators_.at(c); }
  const ResultReturn &return_fabric() const { return returns_; }
  const Output &tick(const Input &in) {
    if (!in.run) {
      flush();
      enqueue_all(in, out_.fault);
      readiness(out_, false);
      return out_;
    }
    Output next;
    next.fault = out_.fault;
    next.top.resize(c_.columns);
    next.result.resize(c_.columns);
    auto compute_pop = pop(compute_queue_, in.compute_launch, next.fault);
    const bool load_busy =
        load_.row(0) ? (!load_.row(0)->column && c_.columns * c_.outputs > 1)
                     : remaining_[0] > 1;
    auto load_pop = pop(load_queue_, in.load_launch && !load_busy, next.fault);
    if (in.load_launch && load_busy)
      next.fault = true;
    auto parameter_pop = pop(parameter_queue_, in.parameter_launch, next.fault);
    auto dequant_pop = pop(dequant_queue_, in.dequant_launch, next.fault);
    auto backend_pop = pop(backend_queue_, in.backend_launch, next.fault);
    const auto old_top = out_.top;
    const auto old_return = returns_.output();
    std::vector<Supercell::Output> before;
    before.reserve(cells_.size());
    const Streams absent;
    for (const auto &cell : cells_)
      before.push_back(cell.output());
    for (std::size_t r = 0; r < c_.rows; ++r) {
      const auto arrival = load_.row(r);
      if (arrival)
        load_word_[r] = *arrival;
      const bool load_active = bool(arrival) || remaining_[r] != 0;
      const auto &dq = dequant_.row(r);
      if (load_active != bool(dq))
        next.fault = true;
      WeightDequantizer::Input di;
      di.load = parameter_.row(r);
      if (load_active)
        di.dequant = dq;
      di.weight_buffer = load_word_[r].weight_buffer;
      const auto old_weight = dequantizers_[r].output().weight;
      const auto &d = dequantizers_[r].tick(di, r < in.weight.size() ? in.weight[r] : absent);
      next.weight_consume.push_back(d.consume);
      next.fault = next.fault || d.fault;
      for (std::size_t col = 0; col < c_.columns; ++col) {
        Supercell::Input ci;
        const auto p = r * c_.columns + col;
        if (col) {
          ci.activation = before[p - 1].activation;
          ci.weight = before[p - 1].weight;
        } else {
          ci.weight = old_weight;
          if (compute_.row(r)) {
            if (r < in.activation.size() && in.activation[r])
              ci.activation = Activation{*in.activation[r],
                                         compute_.row(r)->command.weight_buffer,
                                         compute_.row(r)->tag};
            else
              next.fault = true;
          }
        }
        if (r)
          ci.below = before[p - c_.columns].partial;
        const auto &co = cells_[p].tick(ci);
        next.fault = next.fault || co.fault;
        if (r == c_.rows - 1) {
          Batch b;
          for (const auto &value : co.partial)
            if (value)
              b.push_back(*value);
          if (b.size() == c_.outputs)
            next.top[col] = std::move(b);
        }
      }
      if (arrival)
        remaining_[r] = arrival->column ? 0 : c_.columns * c_.outputs - 1;
      else if (remaining_[r])
        --remaining_[r];
    }
    returns_.tick(old_top);
    next.returned = returns_.output();
    std::optional<TaggedCompute> tc;
    if (compute_pop)
      tc = TaggedCompute{*compute_pop, tag_++};
    compute_.tick(tc);
    load_.tick(load_pop);
    parameter_.tick(parameter_pop);
    dequant_.tick(dequant_pop);
    backend_.tick(backend_pop);
    for (std::size_t col = 0; col < c_.columns; ++col) {
      std::optional<Accumulator::Request> request;
      if (backend_.row(col))
        request = Accumulator::Request{*backend_.row(col), {}};
      next.result[col] = accumulators_[col].tick(request, old_return[col]);
      next.fault = next.fault || next.result[col].fault;
    }
    for (std::size_t r = 0; r < c_.rows; ++r) {
      next.compute.push_back(compute_.row(r)
                                 ? std::optional(compute_.row(r)->command)
                                 : std::nullopt);
      next.load.push_back(load_.row(r));
      next.parameter.push_back(parameter_.row(r));
      next.dequant.push_back(dequant_.row(r));
    }
    enqueue_all(in, next.fault);
    readiness(next, true);
    out_ = std::move(next);
    return out_;
  }

private:
  struct TaggedCompute {
    ComputeInstruction command;
    std::uint16_t tag;
  };
  template <class T>
  std::optional<T> pop(std::deque<T> &q, bool launch, bool &fault) {
    if (!launch)
      return {};
    if (q.empty()) {
      fault = true;
      return {};
    }
    auto t = q.front();
    q.pop_front();
    return t;
  }
  template <class T>
  void enqueue(std::deque<T> &q, const std::optional<T> &value, bool &fault) {
    if (!value || q.size() >= c_.queue_depth)
      return;
    q.push_back(*value);
  }
  void enqueue_all(const Input &in, bool &fault) {
    if (in.compute_enqueue)
      in.compute_enqueue->encode();
    if (in.load_enqueue)
      in.load_enqueue->encode();
    if (in.parameter_enqueue)
      in.parameter_enqueue->encode();
    if (in.dequant_enqueue)
      in.dequant_enqueue->encode();
    if (in.backend_enqueue)
      in.backend_enqueue->encode(c_);
    enqueue(compute_queue_, in.compute_enqueue, fault);
    enqueue(load_queue_, in.load_enqueue, fault);
    enqueue(parameter_queue_, in.parameter_enqueue, fault);
    enqueue(dequant_queue_, in.dequant_enqueue, fault);
    enqueue(backend_queue_, in.backend_enqueue, fault);
  }
  void readiness(Output &o, bool run) const {
    o.enqueue_ready = {compute_queue_.size() < c_.queue_depth,
                       load_queue_.size() < c_.queue_depth,
                       parameter_queue_.size() < c_.queue_depth,
                       dequant_queue_.size() < c_.queue_depth,
                       backend_queue_.size() < c_.queue_depth};
    const bool busy =
        load_.row(0) ? (!load_.row(0)->column && c_.columns * c_.outputs > 1)
                     : remaining_[0] > 1;
    o.launch_ready = {
        run && !compute_queue_.empty(), run && !load_queue_.empty() && !busy,
        run && !parameter_queue_.empty(), run && !dequant_queue_.empty(),
        run && !backend_queue_.empty()};
  }
  Config c_;
  std::vector<Supercell> cells_;
  std::vector<WeightDequantizer> dequantizers_;
  std::vector<Accumulator> accumulators_;
  CommandWave<TaggedCompute> compute_;
  CommandWave<LoadInstruction> load_;
  CommandWave<ParamLoadInstruction> parameter_;
  CommandWave<DequantInstruction> dequant_;
  CommandWave<BackendInstruction> backend_;
  ResultReturn returns_;
  std::vector<std::size_t> remaining_;
  std::vector<LoadInstruction> load_word_;
  std::deque<ComputeInstruction> compute_queue_;
  std::deque<LoadInstruction> load_queue_;
  std::deque<ParamLoadInstruction> parameter_queue_;
  std::deque<DequantInstruction> dequant_queue_;
  std::deque<BackendInstruction> backend_queue_;
  std::uint16_t tag_ = 0;
  Output out_;
};
} // namespace ftlpu::rtl
