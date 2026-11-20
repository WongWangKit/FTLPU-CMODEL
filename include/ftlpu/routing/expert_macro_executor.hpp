#pragma once
#include "ftlpu/routing/expert_task_controller.hpp"
#include "ftlpu/system/icu.hpp"

namespace ftlpu {
struct ExpertMacroTaskTrace {
  ExpertTaskContext task;
  std::uint64_t start_cycle{}, finish_cycle{}, output_segments{};
  std::size_t weight_buffer{};
};
class ExpertMacroExecutor final : public ExpertTaskConsumer {
public:
  bool idle() const { return state_ == State::Idle && !done_; }
  const std::deque<ExpertMacroTaskTrace> &trace() const { return trace_; }
  std::uint64_t gate_weight_hits() const { return gate_weight_hits_; }
  void reset() { *this = ExpertMacroExecutor{}; }
  void configure(const ExpertMacroTemplateDescriptor &t,
                 const RouteAddressDescriptor &a,
                 const ExpertWeightTableDescriptor *w) {
    if (!idle())
      throw std::logic_error("configure busy expert macro executor");
    template_ = &t;
    address_ = &a;
    weights_ = w;
    gate_weight_reuse_ = std::ranges::any_of(t.queues, [](const auto &q) {
      return std::ranges::any_of(q.bindings, [](const auto &b) {
        return b.parameter == MacroTaskParameter::GateLoadCount;
      });
    });
    early_compute_queue_.reset();
    if (t.gate_hit_cycle_advance) {
      const auto compute = std::ranges::find_if(t.queues, [](const auto &q) {
        return q.kind == static_cast<unsigned>(IcuMacroQueueKind::MxmCompute);
      });
      if (compute != t.queues.end())
        early_compute_queue_ = std::size_t(compute - t.queues.begin());
    }
    gate_weight_owner_.reset();
  }
  bool accept_task(const ExpertTaskContext &task) override {
    if (!idle())
      return false;
    if (!template_ || !task.valid_rows || task.valid_rows > 32 ||
        std::uint64_t(task.assignment_base) + task.valid_rows >
            template_->output.rows)
      throw std::invalid_argument("expert macro task output capacity");
    task_ = task;
    input_ = false;
    weight_ready_ = weights_ == nullptr;
    state_ = State::Wait;
    return true;
  }
  void input_ready(const ExpertTaskContext &task) override {
    check(task);
    if (state_ != State::Wait || input_)
      throw std::logic_error("unexpected macro INPUT_READY");
    input_ = true;
  }
  void weights_ready(const ExpertTaskContext &task, std::size_t buffer) override {
    check(task);
    if (state_ != State::Wait || weight_ready_)
      throw std::logic_error("unexpected macro WEIGHT_READY");
    weight_ready_ = true;
    weight_buffer_ = buffer;
  }
  void validate_double_buffer_layout(InstructionControlUnit &icu) const {
    if (!weights_ || icu.transport_overlap_enabled())
      throw std::invalid_argument("expert double buffering requires dedicated weight transport");
    std::set<std::size_t> primary;
    for (std::size_t i = 0; i < weights_->segments; ++i) {
      const auto s = expert_weight_segment(*weights_, i);
      if (s.fabric_stream != 30)
        throw std::invalid_argument("expert double buffering reserves W30 for weights");
      primary.insert(InstructionControlUnit::mem_queue(
          static_cast<Hemisphere>(s.hemisphere), s.slice, s.bank));
    }
    for (const auto bank : primary) {
      const auto alternate = bank ^ 1;
      if (primary.contains(alternate) || !icu.mem_iq(alternate).done())
        throw std::invalid_argument("alternate expert weight bank is already scheduled");
      for (const auto *view : {&address_->source, &address_->destination, &template_->output})
        for (std::size_t plane = 0; plane < view->bytes(); ++plane) {
          const auto &p = view->planes[plane];
          if (alternate == InstructionControlUnit::mem_queue(
                               static_cast<Hemisphere>(p.hemisphere), p.slice, p.bank))
            throw std::invalid_argument("alternate expert weight bank aliases a tensor view");
        }
      for (const auto &queue : template_->queues) {
        if (queue.kind != 0) continue;
        if (queue.index == alternate ||
            (queue.index == bank && !std::ranges::any_of(queue.bindings, [](const auto &binding) {
               return binding.parameter == MacroTaskParameter::WeightBase;
             })))
          throw std::invalid_argument("expert double buffering requires dedicated weight read queues");
      }
    }
  }
  std::optional<ExpertTaskIdentity> poll_done() override {
    auto result = done_;
    if (result) {
      done_.reset();
      state_ = State::Idle;
    }
    return result;
  }
  void tick(InstructionControlUnit &icu,
            std::array<MemArrayModel, hw::kHemispheres> &mems,
            bool datapaths_drained, std::uint64_t cycle) {
    if (state_ == State::Wait && input_ && weight_ready_) {
      for (auto &m : mems)
        m.begin_expert_output(task_.identity.generation,
                              task_.identity.task_id);
      queue_index_ = 0;
      // The MoE template reserves one MXM buffer for Gate across tasks.
      // Only a completed task from this plan can establish its contents.
      gate_weight_hit_ = gate_weight_reuse_ && gate_weight_owner_ &&
          gate_weight_owner_->identity.generation == task_.identity.generation &&
          gate_weight_owner_->identity.plan_epoch == task_.identity.plan_epoch &&
          gate_weight_owner_->expert_id == task_.expert_id;
      gate_weight_hits_ += gate_weight_hit_;
      state_ = State::Bind;
      return;
    }
    if (state_ == State::Bind) {
      // The already-bound Compute decoder can fetch while the controller
      // binds the last three queues. Keep one bind per physical cycle.
      constexpr std::size_t compute_decode_overlap = 3;
      if (early_compute_queue_ && *early_compute_queue_ < queue_index_ &&
          template_->queues.size() - queue_index_ <= compute_decode_overlap) {
        const auto &compute = template_->queues[*early_compute_queue_];
        icu.mxm_compute_iq(compute.index).prepare_task_cycle();
      }
      const auto &q = template_->queues.at(queue_index_++);
      const auto p = parameters(q);
      icu.visit_program_queue(
          static_cast<IcuMacroQueueKind>(q.kind), q.index, false,
          [&](auto &queue) { queue.begin_task(p, q.bindings); });
      if (queue_index_ == template_->queues.size())
        state_ = State::Prepare;
      return;
    }
    if (state_ == State::Prepare) {
      bool ready = true;
      for (const auto &q : template_->queues)
        icu.visit_program_queue(static_cast<IcuMacroQueueKind>(q.kind), q.index,
                                false, [&](auto &queue) {
                                  queue.prepare_task_cycle();
                                  ready &= queue.task_ready();
                                });
      if (ready) {
        const auto advance = gate_weight_hit_ ? template_->gate_hit_cycle_advance : 0;
        for (const auto &q : template_->queues)
          icu.visit_program_queue(static_cast<IcuMacroQueueKind>(q.kind),
                                  q.index, false,
                                  [&](auto &queue) { queue.launch_task(advance); });
        if (trace_.size() == 256)
          trace_.pop_front();
        trace_.push_back({task_, cycle + 1, 0, 0, weight_buffer_});
        state_ = State::Run;
      }
      return;
    }
    if (state_ != State::Run)
      return;
    std::uint64_t committed = 0;
    for (const auto &m : mems)
      committed += m.expert_output_segments(task_.identity.generation,
                                            task_.identity.task_id);
    const auto expected =
        std::uint64_t(task_.valid_rows) *
        (template_->output.columns / hw::kPhysicalVectorBytes) *
        template_->output.bytes() * hw::kTileRows;
    if (committed > expected)
      throw std::logic_error("duplicate expert output SRAM commit");
    bool done = true;
    for (const auto &q : template_->queues)
      icu.visit_program_queue(static_cast<IcuMacroQueueKind>(q.kind), q.index,
                              false,
                              [&](auto &queue) { done &= queue.done(); });
    if (done && datapaths_drained) {
      if (committed != expected)
        throw std::logic_error("expert macro output commit count mismatch");
      for (const auto &q : template_->queues)
        icu.visit_program_queue(static_cast<IcuMacroQueueKind>(q.kind), q.index,
                                false,
                                [](auto &queue) { queue.finish_task(); });
      trace_.back().finish_cycle = cycle;
      trace_.back().output_segments = committed;
      if (gate_weight_reuse_)
        gate_weight_owner_ = task_;
      done_ = task_.identity;
      state_ = State::Done;
    }
  }

private:
  void check(const ExpertTaskContext &task) const {
    if (task != task_)
      throw std::logic_error("stale expert macro task");
  }
  MacroTaskParameters parameters(const ExpertTemplateQueue &q) const {
    MacroTaskParameters p;
    p.generation = task_.identity.generation;
    p.task_id = task_.identity.task_id;
    p.values[0] = task_.valid_rows;
    p.values[4] = template_->workspace_base;
    p.values[5] = template_->scale_bf16;
    p.values[6] = task_.assignment_base;
    p.values[static_cast<unsigned>(MacroTaskParameter::GateLoadCount)] =
        gate_weight_hit_ ? 0 : kMoeGateWeightLoadCount;
    for (const auto &binding : q.bindings) {
      if (binding.parameter != MacroTaskParameter::InputBase &&
          binding.parameter != MacroTaskParameter::OutputBase &&
          binding.parameter != MacroTaskParameter::WeightBase)
        continue;
      if (q.kind != 0)
        throw std::invalid_argument(
            "physical SRAM base binding on non-MEM queue");
      const auto local =
          q.index % InstructionControlUnit::kMemQueuesPerHemisphere;
      const auto h = q.index / InstructionControlUnit::kMemQueuesPerHemisphere,
                 s = local / hw::kMemBanksPerSlice,
                 b = local % hw::kMemBanksPerSlice;
      bool found = false;
      if (binding.parameter == MacroTaskParameter::WeightBase) {
        if (!weights_)
          throw std::invalid_argument(
              "weight base binding without weight table");
        p.values[2] = q.weight_base;
        p.alternate_weight_bank = weight_buffer_ == 1;
        found = true;
      } else {
        const bool output = binding.parameter == MacroTaskParameter::OutputBase;
        const auto &view = output ? template_->output : address_->destination;
        for (std::size_t plane = 0; plane < view.bytes(); ++plane) {
          const auto place = view.planes[plane];
          if (place.hemisphere == h && place.slice == s && place.bank == b) {
            const auto base = std::uint64_t(place.base_row) +
                              (output ? std::uint64_t(task_.assignment_base) *
                                            view.token_stride
                                      : 0);
            if (base > UINT32_MAX)
              throw std::overflow_error("macro task base address overflow");
            p.values[output ? 3 : 1] = static_cast<std::uint32_t>(base);
            found = true;
          }
        }
      }
      if (!found)
        throw std::invalid_argument(
            "macro base binding requires a fixed matching queue layout");
    }
    return p;
  }
  enum class State { Idle, Wait, Bind, Prepare, Run, Done };
  State state_{State::Idle};
  const ExpertMacroTemplateDescriptor *template_{};
  const RouteAddressDescriptor *address_{};
  const ExpertWeightTableDescriptor *weights_{};
  ExpertTaskContext task_{};
  std::optional<ExpertTaskIdentity> done_;
  std::deque<ExpertMacroTaskTrace> trace_;
  bool input_{}, weight_ready_{};
  bool gate_weight_reuse_{}, gate_weight_hit_{};
  std::optional<std::size_t> early_compute_queue_;
  std::optional<ExpertTaskContext> gate_weight_owner_;
  std::uint64_t gate_weight_hits_{};
  std::size_t queue_index_{}, weight_buffer_{};
};
} // namespace ftlpu
