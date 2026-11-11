#pragma once
#include "ftlpu/routing/control_unit.hpp"
#include <limits>

namespace ftlpu {
// A clocked device endpoint. Runtime must not implement these handshakes by
// reading routing results or acknowledging individual tasks on the host.
class ExpertTaskConsumer {
public:
  virtual ~ExpertTaskConsumer() = default;
  virtual bool accept_task(const ExpertTaskContext &) = 0;
  virtual void input_ready(const ExpertTaskContext &) = 0;
  virtual void weights_ready(const ExpertTaskContext &) {}
  virtual void weights_ready(const ExpertTaskContext &task, std::size_t) { weights_ready(task); }
  virtual std::optional<ExpertTaskIdentity> poll_done() = 0;
};
class ExpertWeightService {
public:
  virtual ~ExpertWeightService() = default;
  virtual void start(const ExpertTaskContext &) = 0;
  virtual bool ready(ExpertTaskIdentity) const = 0;
  virtual bool transfer_started(ExpertTaskIdentity task) const { return ready(task); }
  virtual bool supports_prefetch() const { return false; }
  virtual void prefetch(const ExpertTaskContext &) {}
  virtual std::size_t buffer_for(ExpertTaskIdentity) const { return 0; }
  virtual void release_task(ExpertTaskIdentity) = 0;
};

struct ExpertTaskStatistics {
  std::uint64_t tasks{}, completed{}, wait_cycles{};
};

class ExpertTaskController {
public:
  bool idle() const { return state_ == State::Idle && !completion_; }
  const ExpertTaskContext &task() const { return task_; }
  const ExpertTaskStatistics &statistics() const { return statistics_; }
  void set_input_weight_overlap(bool enabled) {
    if (!idle())
      throw std::logic_error("configure overlap during task execution");
    overlap_input_weights_ = enabled;
  }
  void connect_weights(ExpertWeightService *weights) {
    if (!idle())
      throw std::logic_error("connect weights during task execution");
    weights_ = weights;
  }
  void connect(ExpertTaskConsumer *consumer) {
    if (!idle())
      throw std::logic_error("connect busy task controller");
    consumer_ = consumer;
  }
  void disconnect_if(ExpertTaskConsumer *consumer) {
    if(!idle())throw std::logic_error("disconnect busy task controller");
    if(consumer_==consumer)consumer_=nullptr;
  }
  void reset_program() {
    if (!idle())
      throw std::logic_error("reset busy task controller");
    reset();
  }
  void reset() {
    auto *consumer = consumer_;
    auto *weights = weights_;
    const auto overlap = overlap_input_weights_;
    *this = ExpertTaskController{};
    consumer_ = consumer;
    weights_ = weights;
    overlap_input_weights_ = overlap;
  }
  void start(RoutingInstruction instruction,
             const std::vector<RoutingDescriptor> &descriptors,
             std::uint64_t generation, RoutingControlUnit &rcu) {
    if (!idle() || !consumer_ || instruction.opcode != RoutingOpcode::Dispatch)
      throw std::logic_error("task controller busy or device consumer missing");
    (void)encode_routing_instruction(instruction);
    const auto &d = descriptors.at(instruction.descriptor);
    RoutingControlUnit::validate_descriptor(d);
    if (d.opcode != RoutingOpcode::Dispatch)
      throw std::invalid_argument("dispatch opcode mismatch");
    const auto &a = descriptors.at(d.dispatch.address_template_descriptor_id);
    RoutingControlUnit::validate_descriptor(a);
    if (a.opcode != RoutingOpcode::Address ||
        a.address.context_id != d.dispatch.context_id ||
        a.address.max_rows < d.dispatch.block_rows)
      throw std::invalid_argument("invalid dispatch address template");
    const auto epoch = rcu.pin_plan(generation, d.dispatch.context_id);
    instruction_ = instruction;
    descriptor_ = d.dispatch;
    address_ = a;
    task_ = {};
    task_.identity = {generation, epoch, 0};
    task_.route_context_id = d.dispatch.context_id;
    state_ = State::ReadActiveCount;
    pending_read_ = false;
    lookahead_issued_ = false;
  }
  void input_committed(const RoutingCompletion &completion) {
    if (state_ != State::WaitGather ||
        completion.generation != task_.identity.generation ||
        completion.task_id != task_.identity.task_id ||
        completion.opcode != RoutingOpcode::Address)
      throw std::logic_error("unexpected task input completion");
    state_ = State::PublishInput;
  }
  void notify_done(ExpertTaskIdentity identity) {
    if (identity.generation < task_.identity.generation)
      return;
    if (identity.generation == task_.identity.generation &&
        (identity.plan_epoch < task_.identity.plan_epoch ||
         (identity.plan_epoch == task_.identity.plan_epoch &&
          identity.task_id && identity.task_id <= retired_task_)))
      return;
    if (identity != task_.identity || !identity.task_id ||
        state_ != State::WaitDone)
      throw std::logic_error("unexpected task consumer completion");
    done_ = true;
  }
  std::optional<RoutingCompletion> take_completion() {
    auto result = completion_;
    completion_.reset();
    return result;
  }
  void tick(RoutingControlUnit &rcu) {
    if (state_ == State::Idle)
      return;
    if (auto done = consumer_->poll_done())
      notify_done(*done);
    switch (state_) {
    case State::ReadActiveCount:
      if (auto value = read(rcu, RouteMetadataKind::ActiveCount, 0)) {
        active_count_ = *value;
        state_ = active_count_ ? State::ReadExpert : State::Release;
      }
      break;
    case State::ReadExpert:
      if (auto value =
              read(rcu, RouteMetadataKind::Active, task_.active_index)) {
        task_.expert_id = *value;
        state_ = State::ReadCount;
      }
      break;
    case State::ReadCount:
      if (auto value = read(rcu, RouteMetadataKind::Count, task_.expert_id)) {
        if (!*value)
          throw std::logic_error("zero-count active expert");
        task_.expert_token_count = *value;
        state_ = State::ReadOffset;
      }
      break;
    case State::ReadOffset:
      if (auto value = read(rcu, RouteMetadataKind::Offset, task_.expert_id)) {
        expert_base_ = *value;
        state_ = State::Build;
      }
      break;
    case State::Build:
      if (next_task_ == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("task identity exhausted");
      task_.identity.task_id = ++next_task_;
      task_.valid_rows = std::min(descriptor_.block_rows,
                                  task_.expert_token_count - task_.row_offset);
      task_.assignment_base = expert_base_ + task_.row_offset;
      task_.more =
          task_.row_offset + task_.valid_rows < task_.expert_token_count;
      ++statistics_.tasks;
      state_ = State::Offer;
      break;
    case State::Offer:
      if (!consumer_->accept_task(task_)) {
        ++statistics_.wait_cycles;
        break;
      }
      if (weights_ && overlap_input_weights_) {
        weights_->start(task_);
        state_ = State::WaitWeightLaunch;
      } else {
        state_ = State::StartGather;
      }
      break;
    case State::WaitWeightLaunch:
      // Descriptor scanning can outlast a small Gather. Start the Gather
      // when the transfer is actually underway (or weights already reside).
      if (weights_->transfer_started(task_.identity))
        state_ = State::StartGather;
      else
        ++statistics_.wait_cycles;
      break;
    case State::StartGather:
      address_.address.active_index = task_.active_index;
      address_.address.row_offset = task_.row_offset;
      address_.address.max_rows = descriptor_.block_rows;
      rcu.start_task_address(address_, task_.identity);
      state_ = State::WaitGather;
      break;
    case State::WaitGather:
      ++statistics_.wait_cycles;
      break;
    case State::PublishInput:
      consumer_->input_ready(task_);
      if (weights_) {
        if (!overlap_input_weights_)
          weights_->start(task_);
        state_ = State::WaitWeights;
      } else
        state_ = State::WaitDone;
      break;
    case State::WaitWeights:
      if (weights_->ready(task_.identity)) {
        consumer_->weights_ready(task_, weights_->buffer_for(task_.identity));
        state_ = State::WaitDone;
      } else
        ++statistics_.wait_cycles;
      break;
    case State::WaitDone:
      if (!lookahead_issued_ && weights_ && weights_->supports_prefetch() &&
          task_.active_index + 1 < active_count_) {
        const auto expert = read(rcu, RouteMetadataKind::Active, task_.active_index + 1);
        if (!expert) break;
        auto next = task_;
        ++next.active_index;
        next.expert_id = *expert;
        const auto rows_left = task_.expert_token_count - task_.row_offset;
        const auto blocks_left = (rows_left + descriptor_.block_rows - 1) / descriptor_.block_rows;
        if (task_.identity.task_id > std::numeric_limits<std::uint64_t>::max() - blocks_left)
          throw std::overflow_error("expert lookahead task identity exhausted");
        next.identity.task_id += blocks_left;
        weights_->prefetch(next);
        lookahead_issued_ = true;
      }
      if (done_) {
        if (weights_)
          weights_->release_task(task_.identity);
        done_ = false;
        retired_task_ = task_.identity.task_id;
        ++statistics_.completed;
        state_ = State::Advance;
      } else
        ++statistics_.wait_cycles;
      break;
    case State::Advance:
      if (task_.more) {
        task_.row_offset += task_.valid_rows;
        state_ = State::Build;
      } else {
        task_.row_offset = 0;
        lookahead_issued_ = false;
        state_ = ++task_.active_index < active_count_ ? State::ReadExpert
                                                      : State::Release;
      }
      break;
    case State::Release:
      rcu.release_plan();
      completion_ =
          RoutingCompletion{task_.identity.generation, instruction_.completion,
                            RoutingOpcode::Dispatch};
      state_ = State::Idle;
      break;
    case State::Idle:
      break;
    }
  }

private:
  enum class State {
    Idle,
    ReadActiveCount,
    ReadExpert,
    ReadCount,
    ReadOffset,
    Build,
    Offer,
    WaitWeightLaunch,
    StartGather,
    WaitGather,
    PublishInput,
    WaitWeights,
    WaitDone,
    Advance,
    Release
  };
  std::optional<std::uint32_t>
  read(RoutingControlUnit &rcu, RouteMetadataKind kind, std::uint32_t index) {
    if (pending_read_) {
      auto value = rcu.take_metadata_response();
      if (value)
        pending_read_ = false;
      return value;
    }
    pending_read_ = rcu.request_metadata({kind, index});
    return {};
  }
  State state_{State::Idle};
  ExpertTaskConsumer *consumer_{};
  ExpertWeightService *weights_{};
  RoutingInstruction instruction_{};
  RouteTaskDispatchDescriptor descriptor_{};
  RoutingDescriptor address_{};
  ExpertTaskContext task_{};
  ExpertTaskStatistics statistics_{};
  std::optional<RoutingCompletion> completion_;
  std::uint64_t next_task_{}, retired_task_{};
  std::uint32_t active_count_{}, expert_base_{};
  bool pending_read_{}, done_{}, overlap_input_weights_{}, lookahead_issued_{};
};
} // namespace ftlpu
