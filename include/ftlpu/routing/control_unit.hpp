#pragma once

#include "ftlpu/mem/mem_array.hpp"
#include "ftlpu/routing/instruction.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <optional>
#include <tuple>

namespace ftlpu {

struct RoutingStatistics {
  std::uint64_t descriptor_words{}, descriptor_wait_cycles{}, cycles{},
      comparisons{}, metadata_accesses{}, control_requests{},
      bank_wait_cycles{}, address_wait_cycles{}, stream_wait_cycles{},
      gathered_vectors{}, commit_segments{};
  std::size_t peak_addresses{};
};
struct RoutingAssignment {
  std::uint32_t token{}, slot{}, weight_bits{};
};
struct RoutingTraceEntry {
  std::uint64_t generation{}, begin_cycle{}, end_cycle{};
  std::uint32_t instance{};
  RoutingOpcode opcode{};
};
struct RoutingCompletion {
  std::uint64_t generation{};
  std::uint32_t instance{};
  RoutingOpcode opcode{};
  std::uint64_t task_id{}; // zero denotes a public instruction completion
};
enum class RouteMetadataKind { ActiveCount, Active, Count, Offset, Tokens, K, Inverse, AssignmentWeight };
struct RouteMetadataRequest {
  RouteMetadataKind kind;
  std::uint32_t index;
};

// No host algorithm runs inside this model: each switch transition is one
// physical clock and performs at most one comparator or metadata RAM access.
class RoutingControlUnit {
public:
  struct CycleTrace {
    bool valid{}, candidate_valid{}, exchanged{};
    const char *step{"Idle"};
    RoutingOpcode opcode{RoutingOpcode::Select};
    std::uint64_t sequence{};
    std::uint32_t descriptor{}, completion{}, token{}, expert{}, slot{}, position{}, assignment{},
      value{}, raw_bits{}, incoming_id{}, choice_bits{}, candidate_id{}, candidate_bits{}, winner_id{};
  };
  void enable_cycle_trace(bool enabled=true){cycle_trace_enabled_=enabled;}
  const CycleTrace &last_cycle_trace() const {return cycle_trace_;}
  void configure(RoutingConfig config) {
    if (!idle() || plan_in_use())
      throw std::logic_error("configure busy RCU");
    config.validate();
    config_ = config;
  }
  bool idle() const { return state_ == State::Idle && !completion_; }
  bool busy() const { return !idle(); }
  std::uint64_t plan_epoch() const { return plan_epoch_; }
  bool plan_in_use() const {
    return plan_pinned_ || metadata_request_ || metadata_response_;
  }
  std::uint64_t pin_plan(std::uint64_t generation, std::uint32_t context) {
    if (!idle() || plan_in_use() || !planned_ || context != context_ ||
        generation != generation_)
      throw std::logic_error("routing plan unavailable");
    plan_pinned_ = true;
    return plan_epoch_;
  }
  void release_plan() {
    if (!plan_pinned_ || !idle() || metadata_request_ || metadata_response_)
      throw std::logic_error("routing plan still in use");
    plan_pinned_ = false;
  }
  bool request_metadata(RouteMetadataRequest request) {
    if (!plan_pinned_)
      throw std::logic_error("metadata read without plan lease");
    if (!idle() || metadata_request_ || metadata_response_)
      return false;
    metadata_request_ = request;
    return true;
  }
  std::optional<std::uint32_t> take_metadata_response() {
    auto result = metadata_response_;
    metadata_response_.reset();
    return result;
  }
  const RoutingStatistics &statistics() const { return stats_; }
  const std::deque<RoutingTraceEntry> &trace() const { return trace_; }
  void observe_descriptor_fetch(std::size_t words) {
    stats_.descriptor_words += words;
    stats_.descriptor_wait_cycles += words;
  }
  const std::vector<std::uint32_t> &counts() const { return counts_; }
  const std::vector<std::uint32_t> &offsets() const { return offsets_; }
  const std::vector<std::uint32_t> &active_experts() const { return active_; }
  const std::vector<std::uint32_t> &selected_experts() const { return ids_; }
  const std::vector<std::uint32_t> &selected_score_bits() const { return raw_; }
  const std::vector<RoutingAssignment> &assignments() const {
    return assignments_;
  }
  const RouteBlockInfo &block_info() const { return block_; }
  std::optional<RoutingCompletion> take_completion() {
    auto r = completion_;
    completion_.reset();
    return r;
  }
  void reset_program() {
    if (!idle() || plan_in_use())
      throw std::logic_error("reset busy RCU");
    auto c = config_;
    *this = RoutingControlUnit{};
    config_ = c;
  }
  void reset() {
    auto c = config_;
    *this = RoutingControlUnit{};
    config_ = c;
  }

  static void validate_descriptor(const RoutingDescriptor &d,
                                  std::size_t depth = hw::kSramDepthRows) {
    const auto shape = [&](const RoutingTensorView &v, std::uint32_t rows,
                           std::uint32_t cols, RoutingElement e) {
      v.validate(depth);
      if (v.rows != rows || v.columns != cols || v.element != e)
        throw std::invalid_argument("routing view shape/type mismatch");
    };
    switch (d.opcode) {
    case RoutingOpcode::Select: {
      const auto &s = d.select;
      if (!s.context_id || !s.tokens || s.tokens > RoutingConfig::max_tokens ||
          !s.experts || s.experts > RoutingConfig::max_experts || !s.k ||
          s.k > s.experts || s.k > RoutingConfig::max_k)
        throw std::invalid_argument("routing SELECT capacity/context");
      shape(s.choice_scores, s.tokens, s.experts, RoutingElement::F32);
      if (s.raw_scores.present())
        shape(s.raw_scores, s.tokens, s.experts, RoutingElement::F32);
      if (s.eligibility.present())
        shape(s.eligibility, s.tokens, s.experts, RoutingElement::U32);
      if (s.selected_ids.present())
        shape(s.selected_ids, s.tokens, s.k, RoutingElement::U32);
      if (s.selected_scores.present())
        shape(s.selected_scores, s.tokens, s.k, RoutingElement::F32);
      std::vector<RoutingTensorView> inputs{s.choice_scores};
      if (s.raw_scores.present())
        inputs.push_back(s.raw_scores);
      if (s.eligibility.present())
        inputs.push_back(s.eligibility);
      for (auto *out : {&s.selected_ids, &s.selected_scores})
        if (out->present())
          for (const auto &in : inputs)
            if (overlap(*out, in))
              throw std::invalid_argument(
                  "routing SELECT output aliases input");
      if (s.selected_ids.present() && s.selected_scores.present() &&
          overlap(s.selected_ids, s.selected_scores))
        throw std::invalid_argument("routing SELECT outputs overlap");
      break;
    }
    case RoutingOpcode::Plan:
      if (!d.plan.context_id)
        throw std::invalid_argument("routing PLAN context");
      if (d.plan.weights.present()) {
        d.plan.weights.validate(depth);
        if (d.plan.weights.element != RoutingElement::F32)
          throw std::invalid_argument("routing weights must be FP32");
      }
      break;
    case RoutingOpcode::Address: {
      const auto &a = d.address;
      if (!a.context_id || !a.max_rows ||
          a.max_rows > RoutingConfig::block_rows)
        throw std::invalid_argument("routing block capacity/context");
      a.source.validate(depth);
      a.destination.validate(depth);
      if ((a.source.element != RoutingElement::F16 &&
           a.source.element != RoutingElement::BF16) ||
          a.source.element != a.destination.element ||
          a.source.columns != a.destination.columns ||
          a.source.columns % hw::kPhysicalVectorBytes ||
          a.destination.rows < a.max_rows)
        throw std::invalid_argument("routing Gather requires matching "
                                    "whole-vector 16-bit byte-planar views");
      if (overlap(a.source, a.destination))
        throw std::invalid_argument(
            "routing Gather source/destination overlap");
      break;
    }
    case RoutingOpcode::Dispatch:
      if (!d.dispatch.context_id || !d.dispatch.block_rows ||
          d.dispatch.block_rows > RoutingConfig::block_rows ||
          d.dispatch.address_template_descriptor_id >=
              RoutingConfig::descriptor_depth)
        throw std::invalid_argument("routing dispatch descriptor");
      break;
    case RoutingOpcode::WeightTable:
      validate_expert_weight_table(d.weight_table, depth);
      break;
    case RoutingOpcode::ExpertTemplate:
      validate_expert_macro_template(d.expert_template,depth);break;
    case RoutingOpcode::Combine:
      validate_route_combine(d.combine,depth);break;
    default:
      throw std::invalid_argument("routing descriptor opcode");
    }
  }
  void start(RoutingInstruction i, const RoutingDescriptor &d,
             std::uint64_t generation) {
    if (plan_in_use())
      throw std::logic_error("routing plan is leased");
    begin(i, d, generation, 0);
  }
  void start_task_address(const RoutingDescriptor &d,
                          ExpertTaskIdentity identity) {
    if (!plan_pinned_ || identity.generation != generation_ ||
        identity.plan_epoch != plan_epoch_ || !identity.task_id ||
        d.opcode != RoutingOpcode::Address)
      throw std::logic_error("invalid internal routing task");
    begin({RoutingOpcode::Address, 0, 0}, d, identity.generation,
          identity.task_id);
  }

private:
  void begin(RoutingInstruction i, const RoutingDescriptor &d,
             std::uint64_t generation, std::uint64_t task_id) {
    if (!config_.enabled || !idle() || !generation || i.opcode != d.opcode)
      throw std::logic_error("RCU disabled/busy or descriptor mismatch");
    if (i.opcode == RoutingOpcode::Dispatch ||
        i.opcode == RoutingOpcode::WeightTable || i.opcode==RoutingOpcode::ExpertTemplate || i.opcode==RoutingOpcode::Combine)
      throw std::logic_error("dispatch requires task controller");
    if (!task_id)
      (void)encode_routing_instruction(i);
    validate_descriptor(d);
    if (d.opcode != RoutingOpcode::Select &&
        (!selected_ ||
         context_ != (d.opcode == RoutingOpcode::Plan ? d.plan.context_id
                                                      : d.address.context_id)))
      throw std::logic_error("routing context not selected");
    if (d.opcode == RoutingOpcode::Plan && d.plan.weights.present() &&
        (d.plan.weights.rows != t_ || d.plan.weights.columns != k_))
      throw std::invalid_argument("routing weight shape mismatch");
    if (d.opcode == RoutingOpcode::Address &&
        (!planned_ || d.address.active_index >= active_.size() ||
         d.address.source.rows < t_))
      throw std::invalid_argument(
          "routing block requires a valid planned active expert");
    instruction_ = i;
    internal_task_id_ = task_id;
    descriptor_ = d;
    generation_ = generation;
    owner_ = ++next_owner_;
    if (!owner_)
      throw std::invalid_argument("routing owner overflow");
    banks_.clear();
    if (trace_.size() == RoutingConfig::imem_depth)
      trace_.pop_front();
    trace_.push_back({generation, clock_, 0, i.completion, i.opcode});
    const auto add = [&](const RoutingTensorView &v) {
      if (!v.present())
        return;
      for (std::size_t p = 0; p < v.bytes(); ++p) {
        const auto &q = v.planes[p];
        banks_.push_back({q.hemisphere, q.slice, q.bank});
      }
    };
    if (d.opcode == RoutingOpcode::Select) {
      add(d.select.choice_scores);
      add(d.select.raw_scores);
      add(d.select.eligibility);
      add(d.select.selected_ids);
      add(d.select.selected_scores);
      context_ = d.select.context_id;
      t_ = d.select.tokens;
      e_ = d.select.experts;
      k_ = d.select.k;
      ids_.resize(t_ * k_);
      raw_.resize(t_ * k_);
      selected_ = planned_ = false;
      token_ = expert_ = slot_ = 0;
      next_ = State::SelectClear;
    } else if (d.opcode == RoutingOpcode::Plan) {
      add(d.plan.weights);
      counts_.resize(e_);
      offsets_.resize(e_ + 1);
      cursors_.resize(e_);
      active_.clear();
      assignments_.resize(t_ * k_);
      inverse_.resize(t_ * k_);
      pos_ = 0;
      planned_ = false;
      next_ = State::ClearCounts;
    } else {
      add(d.address.source);
      add(d.address.destination);
      block_ = {};
      row_ = feature_ = plane_ = 0;
      transfer_ = Transfer::Idle;
      address_fifo_.clear();
      next_ = State::AddressExpert;
    }
    std::sort(banks_.begin(), banks_.end());
    banks_.erase(std::unique(banks_.begin(), banks_.end()), banks_.end());
    state_ = State::Acquire;
  }

public:
  void tick(std::array<MemArrayModel, hw::kHemispheres> &mems,
            std::array<StreamRegisterFabric, hw::kHemispheres> &fabrics) {
    ++clock_;
    cycle_trace_={};
    if(cycle_trace_enabled_) {
      auto &tr=cycle_trace_;tr.valid=state_!=State::Idle || metadata_request_.has_value();tr.sequence=clock_;
      tr.step=state_name(state_);tr.opcode=instruction_.opcode;tr.descriptor=instruction_.descriptor;tr.completion=instruction_.completion;
      tr.token=static_cast<std::uint32_t>(instruction_.opcode==RoutingOpcode::Plan && k_?pos_/k_:token_);
      tr.expert=instruction_.opcode==RoutingOpcode::Plan?current_expert_:static_cast<std::uint32_t>(expert_);
      tr.slot=static_cast<std::uint32_t>(instruction_.opcode==RoutingOpcode::Plan && k_?pos_%k_:slot_);
      tr.position=static_cast<std::uint32_t>(pos_);tr.assignment=static_cast<std::uint32_t>(dest_);
      tr.value=io_value_;tr.raw_bits=incoming_.raw;
      if(state_==State::Idle && metadata_request_) {tr.step="MetadataRead";tr.position=metadata_request_->index;tr.value=static_cast<std::uint32_t>(metadata_request_->kind);}
      if(state_==State::ChoiceRead)tr.choice_bits=io_value_;
      if(state_==State::Compare) {
        const auto &candidate=candidates_[compare_];tr.slot=static_cast<std::uint32_t>(compare_);tr.incoming_id=incoming_.id;
        tr.choice_bits=std::bit_cast<std::uint32_t>(incoming_.choice);tr.candidate_valid=candidate.valid;
        tr.candidate_id=candidate.id;tr.candidate_bits=std::bit_cast<std::uint32_t>(candidate.choice);
      }
      if(state_==State::StoreId || state_==State::StoreRaw) {tr.value=candidates_[slot_].id;tr.raw_bits=candidates_[slot_].raw;}
      if(state_==State::CountWrite)tr.value=temp_+1;
      if(state_==State::PrefixOffset){tr.expert=static_cast<std::uint32_t>(pos_);tr.value=sum_;}
      if(state_==State::AssignBits)tr.raw_bits=weight_;
      if(state_==State::AssignInverse)tr.value=static_cast<std::uint32_t>(dest_);
    }
    if (state_ == State::Idle) {
      if (metadata_request_ && !metadata_response_) {
        const auto q = *metadata_request_;
        switch (q.kind) {
        case RouteMetadataKind::ActiveCount:
          metadata_response_ = static_cast<std::uint32_t>(active_.size());
          break;
        case RouteMetadataKind::Active:
          metadata_response_ = active_.at(q.index);
          break;
        case RouteMetadataKind::Count:
          metadata_response_ = counts_.at(q.index);
          break;
        case RouteMetadataKind::Offset:
          metadata_response_ = offsets_.at(q.index);
          break;
        case RouteMetadataKind::Tokens: metadata_response_=t_;break;
        case RouteMetadataKind::K: metadata_response_=k_;break;
        case RouteMetadataKind::Inverse: metadata_response_=inverse_.at(q.index);break;
        case RouteMetadataKind::AssignmentWeight: metadata_response_=assignments_.at(q.index).weight_bits;break;
        }
        metadata_request_.reset();
        ++stats_.metadata_accesses;
      }
      return;
    }
    ++stats_.cycles;
    if (instruction_.opcode == RoutingOpcode::Address &&
        state_ != State::Acquire)
      tick_transfer(mems, fabrics);
    switch (state_) {
    case State::Idle:
      break;
    case State::Acquire:
      // All requested banks are drained before granting any lease, so a
      // producer cannot be stranded behind a partially acquired bank set.
      for (auto [h, s, b] : banks_)
        if (!mems[h].routing_bank_available(s, b)) {
          ++stats_.bank_wait_cycles;
          return;
        }
      for (auto [h, s, b] : banks_)
        if (!mems[h].acquire_routing_bank(s, b, owner_))
          throw std::logic_error("routing bank lease race");
      state_ = next_;
      break;
    case State::Io:
      tick_io(mems);
      break;
    case State::SelectClear:
      candidates_[slot_++] = {};
      if (slot_ == k_) {
        slot_ = 0;
        state_ = State::Eligibility;
      }
      break;
    case State::Eligibility:
      if (expert_ == e_) {
        if (!candidates_[k_ - 1].valid)
          throw std::invalid_argument(
              "routing has fewer eligible experts than K");
        slot_ = 0;
        state_ = State::StoreId;
      } else if (descriptor_.select.eligibility.present())
        read(descriptor_.select.eligibility, token_, expert_, State::MaskRead);
      else
        state_ = State::ReadChoice;
      break;
    case State::MaskRead:
      if (io_value_ == 0) {
        ++expert_;
        state_ = State::Eligibility;
      } else
        state_ = State::ReadChoice;
      break;
    case State::ReadChoice:
      read(descriptor_.select.choice_scores, token_, expert_,
           State::ChoiceRead);
      break;
    case State::ChoiceRead:
      incoming_ = {std::bit_cast<float>(io_value_), io_value_,
                   static_cast<std::uint32_t>(expert_), true};
      if (std::isnan(incoming_.choice))
        throw std::invalid_argument("NaN routing choice score");
      if (!descriptor_.select.raw_scores.present() ||
          descriptor_.select.raw_scores == descriptor_.select.choice_scores) {
        compare_ = 0;
        state_ = State::Compare;
      } else
        read(descriptor_.select.raw_scores, token_, expert_, State::RawRead);
      break;
    case State::RawRead:
      if (std::isnan(std::bit_cast<float>(io_value_)))
        throw std::invalid_argument("NaN routing raw score");
      incoming_.raw = io_value_;
      compare_ = 0;
      state_ = State::Compare;
      break;
    case State::Compare: {
      auto &c = candidates_[compare_];
      ++stats_.comparisons;
      const bool exchange=incoming_.valid &&
          (!c.valid || incoming_.choice > c.choice ||
           (incoming_.choice == c.choice && incoming_.id < c.id));
      if(exchange)
        std::swap(c, incoming_);
      if(cycle_trace_enabled_){cycle_trace_.exchanged=exchange;cycle_trace_.winner_id=c.id;}
      if (++compare_ == k_) {
        ++expert_;
        state_ = State::Eligibility;
      }
      break;
    }
    case State::StoreId:
      ids_[token_ * k_ + slot_] = candidates_[slot_].id;
      ++stats_.metadata_accesses;
      state_ = State::StoreRaw;
      break;
    case State::StoreRaw:
      raw_[token_ * k_ + slot_] = candidates_[slot_].raw;
      ++stats_.metadata_accesses;
      state_ = State::ExportId;
      break;
    case State::ExportId:
      if (descriptor_.select.selected_ids.present())
        write(descriptor_.select.selected_ids, token_, slot_,
              ids_[token_ * k_ + slot_], State::ExportRaw);
      else
        state_ = State::ExportRaw;
      break;
    case State::ExportRaw:
      if (descriptor_.select.selected_scores.present())
        write(descriptor_.select.selected_scores, token_, slot_,
              raw_[token_ * k_ + slot_], State::NextSelected);
      else
        state_ = State::NextSelected;
      break;
    case State::NextSelected:
      if (++slot_ < k_)
        state_ = State::StoreId;
      else if (++token_ < t_) {
        slot_ = expert_ = 0;
        state_ = State::SelectClear;
      } else {
        selected_ = true;
        state_ = State::Finish;
      }
      break;
    case State::ClearCounts:
      counts_[pos_++] = 0;
      ++stats_.metadata_accesses;
      if (pos_ == e_) {
        pos_ = 0;
        state_ = State::CountId;
      }
      break;
    case State::CountId:
      current_expert_ = ids_[pos_];
      ++stats_.metadata_accesses;
      state_ = State::CountRead;
      break;
    case State::CountRead:
      temp_ = counts_[current_expert_];
      ++stats_.metadata_accesses;
      state_ = State::CountWrite;
      break;
    case State::CountWrite:
      counts_[current_expert_] = temp_ + 1;
      ++stats_.metadata_accesses;
      state_ = temp_ ? State::CountNext : State::ActiveWrite;
      break;
    case State::ActiveWrite:
      active_.push_back(current_expert_);
      ++stats_.metadata_accesses;
      state_ = State::CountNext;
      break;
    case State::CountNext:
      if (++pos_ == ids_.size()) {
        pos_ = sum_ = 0;
        state_ = State::PrefixCount;
      } else
        state_ = State::CountId;
      break;
    case State::PrefixCount:
      temp_ = counts_[pos_];
      ++stats_.metadata_accesses;
      state_ = State::PrefixOffset;
      break;
    case State::PrefixOffset:
      offsets_[pos_] = sum_;
      ++stats_.metadata_accesses;
      state_ = State::PrefixCursor;
      break;
    case State::PrefixCursor:
      cursors_[pos_] = sum_;
      sum_ += temp_;
      ++stats_.metadata_accesses;
      if (++pos_ == e_)
        state_ = State::PrefixEnd;
      else
        state_ = State::PrefixCount;
      break;
    case State::PrefixEnd:
      offsets_[e_] = sum_;
      ++stats_.metadata_accesses;
      if (sum_ != ids_.size())
        throw std::logic_error("routing count conservation");
      pos_ = 0;
      state_ = State::AssignId;
      break;
    case State::AssignId:
      current_expert_ = ids_[pos_];
      ++stats_.metadata_accesses;
      state_ = State::AssignCursor;
      break;
    case State::AssignCursor:
      dest_ = cursors_[current_expert_];
      ++stats_.metadata_accesses;
      state_ = State::AssignWeight;
      break;
    case State::AssignWeight:
      if (descriptor_.plan.weights.present())
        read(descriptor_.plan.weights, pos_ / k_, pos_ % k_, State::WeightRead);
      else {
        weight_ = raw_[pos_];
        ++stats_.metadata_accesses;
        state_ = State::AssignToken;
      }
      break;
    case State::WeightRead:
      weight_ = io_value_;
      state_ = State::AssignToken;
      break;
    case State::AssignToken:
      assignments_[dest_].token = static_cast<std::uint32_t>(pos_ / k_);
      ++stats_.metadata_accesses;
      state_ = State::AssignSlot;
      break;
    case State::AssignSlot:
      assignments_[dest_].slot = static_cast<std::uint32_t>(pos_ % k_);
      ++stats_.metadata_accesses;
      state_ = State::AssignBits;
      break;
    case State::AssignBits:
      assignments_[dest_].weight_bits = weight_;
      ++stats_.metadata_accesses;
      state_ = State::AssignInverse;
      break;
    case State::AssignInverse:
      inverse_[pos_]=static_cast<std::uint32_t>(dest_);
      ++stats_.metadata_accesses;
      state_ = State::AssignAdvance;
      break;
    case State::AssignAdvance:
      cursors_[current_expert_] = static_cast<std::uint32_t>(dest_ + 1);
      ++stats_.metadata_accesses;
      if (++pos_ == ids_.size()) {
        planned_ = true;
        state_ = State::Finish;
      } else
        state_ = State::AssignId;
      break;
    case State::AddressExpert:
      current_expert_ = active_[descriptor_.address.active_index];
      ++stats_.metadata_accesses;
      state_ = State::AddressCount;
      break;
    case State::AddressCount: {
      const auto &a = descriptor_.address;
      const auto count = counts_[current_expert_];
      ++stats_.metadata_accesses;
      const auto remaining = a.row_offset < count ? count - a.row_offset : 0;
      block_ = {context_,
                current_expert_,
                count,
                a.row_offset,
                std::min(a.max_rows, remaining),
                remaining > a.max_rows};
      state_ = State::AddressOffset;
      break;
    }
    case State::AddressOffset:
      base_assignment_ = offsets_[block_.expert_id];
      ++stats_.metadata_accesses;
      state_ = State::AddressRow;
      break;
    case State::AddressRow:
      if (row_ == block_.valid_rows) {
        state_ = State::AddressDrain;
        break;
      }
      source_token_ =
          assignments_[base_assignment_ + block_.row_offset + row_].token;
      ++stats_.metadata_accesses;
      state_ = State::AddressGenerate;
      break;
    case State::AddressGenerate:
      if (address_fifo_.size() == RoutingConfig::address_depth) {
        ++stats_.address_wait_cycles;
        break;
      }
      {
        const auto &a = descriptor_.address;
        address_fifo_.push_back({a.source.planes[plane_],
                                 a.destination.planes[plane_],
                                 a.source.row(source_token_, feature_, plane_),
                                 a.destination.row(row_, feature_, plane_)});
      }
      stats_.peak_addresses =
          std::max(stats_.peak_addresses, address_fifo_.size());
      if (++plane_ == 2) {
        plane_ = 0;
        feature_ += hw::kPhysicalVectorBytes;
        if (feature_ == descriptor_.address.source.columns) {
          feature_ = 0;
          ++row_;
          state_ = State::AddressRow;
        }
      }
      break;
    case State::AddressDrain:
      if (address_fifo_.empty() && transfer_ == Transfer::Idle)
        state_ = State::Finish;
      break;
    case State::Finish:
      for (auto [h, s, b] : banks_)
        if (!mems[h].routing_bank_idle(s, b)) {
          ++stats_.bank_wait_cycles;
          return;
        }
      for (auto [h, s, b] : banks_)
        mems[h].release_routing_bank(s, b, owner_);
      banks_.clear();
      trace_.back().end_cycle = clock_;
      if (instruction_.opcode == RoutingOpcode::Plan)
        ++plan_epoch_;
      completion_ = RoutingCompletion{generation_, instruction_.completion,
                                      instruction_.opcode, internal_task_id_};
      state_ = State::Idle;
      break;
    }
  }

private:
  enum class State {
    Idle,
    Acquire,
    Io,
    SelectClear,
    Eligibility,
    MaskRead,
    ReadChoice,
    ChoiceRead,
    RawRead,
    Compare,
    StoreId,
    StoreRaw,
    ExportId,
    ExportRaw,
    NextSelected,
    ClearCounts,
    CountId,
    CountRead,
    CountWrite,
    ActiveWrite,
    CountNext,
    PrefixCount,
    PrefixOffset,
    PrefixCursor,
    PrefixEnd,
    AssignId,
    AssignCursor,
    AssignWeight,
    WeightRead,
    AssignToken,
    AssignSlot,
    AssignBits,
      AssignInverse,
    AssignAdvance,
    AddressExpert,
    AddressCount,
    AddressOffset,
    AddressRow,
    AddressGenerate,
    AddressDrain,
    Finish
  };
  enum class Transfer { Idle, Collect, Emit, Commit };
  static const char *state_name(State state) {
#define RCU_STATE_NAME(name) case State::name:return #name;
    switch(state) {
      RCU_STATE_NAME(Idle) RCU_STATE_NAME(Acquire) RCU_STATE_NAME(Io)
      RCU_STATE_NAME(SelectClear) RCU_STATE_NAME(Eligibility) RCU_STATE_NAME(MaskRead)
      RCU_STATE_NAME(ReadChoice) RCU_STATE_NAME(ChoiceRead) RCU_STATE_NAME(RawRead)
      RCU_STATE_NAME(Compare) RCU_STATE_NAME(StoreId) RCU_STATE_NAME(StoreRaw)
      RCU_STATE_NAME(ExportId) RCU_STATE_NAME(ExportRaw) RCU_STATE_NAME(NextSelected)
      RCU_STATE_NAME(ClearCounts) RCU_STATE_NAME(CountId) RCU_STATE_NAME(CountRead)
      RCU_STATE_NAME(CountWrite) RCU_STATE_NAME(ActiveWrite) RCU_STATE_NAME(CountNext)
      RCU_STATE_NAME(PrefixCount) RCU_STATE_NAME(PrefixOffset) RCU_STATE_NAME(PrefixCursor)
      RCU_STATE_NAME(PrefixEnd) RCU_STATE_NAME(AssignId) RCU_STATE_NAME(AssignCursor)
      RCU_STATE_NAME(AssignWeight) RCU_STATE_NAME(WeightRead) RCU_STATE_NAME(AssignToken)
      RCU_STATE_NAME(AssignSlot) RCU_STATE_NAME(AssignBits) RCU_STATE_NAME(AssignInverse)
      RCU_STATE_NAME(AssignAdvance) RCU_STATE_NAME(AddressExpert) RCU_STATE_NAME(AddressCount)
      RCU_STATE_NAME(AddressOffset) RCU_STATE_NAME(AddressRow) RCU_STATE_NAME(AddressGenerate)
      RCU_STATE_NAME(AddressDrain) RCU_STATE_NAME(Finish)
    }
#undef RCU_STATE_NAME
    return "Unknown";
  }
  struct Candidate {
    float choice{};
    std::uint32_t raw{}, id{};
    bool valid{};
  };
  struct AddressPacket {
    RoutingPlane source, destination;
    std::size_t source_row, destination_row;
  };
  static bool overlap(const RoutingTensorView &a, const RoutingTensorView &b) {
    for (std::size_t i = 0; i < a.bytes(); ++i)
      for (std::size_t j = 0; j < b.bytes(); ++j) {
        auto p = a.planes[i], q = b.planes[j];
        if (p.hemisphere == q.hemisphere && p.slice == q.slice &&
            p.bank == q.bank &&
            p.base_row <= b.row(b.rows - 1, b.columns - 1, j) &&
            q.base_row <= a.row(a.rows - 1, a.columns - 1, i))
          return true;
      }
    return false;
  }
  void read(const RoutingTensorView &v, std::size_t r, std::size_t c,
            State next) {
    io_ = {v, r, c, 0, 0, false, false, false, {}};
    resume_ = next;
    state_ = State::Io;
  }
  void write(const RoutingTensorView &v, std::size_t r, std::size_t c,
             std::uint32_t bits, State next) {
    io_ = {v, r, c, 0, bits, true, false, false, {}};
    resume_ = next;
    state_ = State::Io;
  }
  void tick_io(std::array<MemArrayModel, hw::kHemispheres> &mems) {
    auto &i = io_;
    const auto &p = i.view.planes[i.plane];
    auto &m = mems[p.hemisphere];
    const auto lane = i.column % hw::kPhysicalVectorBytes;
    if (i.waiting) {
      const auto response = m.take_control_response();
      if (!response)
        return;
      i.waiting = false;
      if (i.write && !i.writing) {
        i.segment = response->bytes;
        i.segment[lane % hw::kLanesPerTile] =
            static_cast<std::uint8_t>(i.bits >> (i.plane * 8));
        i.writing = true;
        return;
      }
      if (!i.write)
        i.bits |= std::uint32_t(response->bytes[lane % hw::kLanesPerTile])
                  << (i.plane * 8);
      i.writing = false;
      if (++i.plane == i.view.bytes()) {
        io_value_ = i.bits;
        state_ = resume_;
        return;
      }
      return;
    }
    if (m.issue_control(p.slice, p.bank, lane / hw::kLanesPerTile,
                        i.view.row(i.row, i.column, i.plane), owner_, i.writing,
                        i.segment, config_.control_latency)) {
      i.waiting = true;
      ++stats_.control_requests;
    } else
      ++stats_.bank_wait_cycles;
  }
  void tick_transfer(std::array<MemArrayModel, hw::kHemispheres> &mems,
                     std::array<StreamRegisterFabric, hw::kHemispheres> &fs) {
    constexpr auto stream_index = 31u;
    const auto stream = StreamId::East(stream_index);
    if (transfer_ == Transfer::Idle) {
      if (address_fifo_.empty())
        return;
      const auto &address = address_fifo_.front();
      const auto source_column =
          mems[address.source.hemisphere].ports().output_column(
              address.source.slice, stream.direction());
      for (std::size_t t = 0; t < hw::kTileRows; ++t)
        if (fs[address.source.hemisphere].segment_valid(source_column, t,
                                                        stream) ||
            fs[address.source.hemisphere].staged_segment_valid(source_column, t,
                                                               stream)) {
          ++stats_.stream_wait_cycles;
          return;
        }
      packet_ = address;
      address_fifo_.pop_front();
      tag_ = ++next_tag_;
      auto &m = mems[packet_.source.hemisphere];
      if (!m.push_gather_address(packet_.source.slice, packet_.source.bank,
                                 packet_.source_row, owner_, tag_))
        throw std::logic_error("routing address FIFO credit");
      auto i = MemInstruction::Gather(stream, StreamId::East(0));
      i.routing_owner = owner_;
      i.routing_transfer = tag_;
      m.enqueue_instruction(packet_.source.slice, packet_.source.bank, i);
      tile_ = 0;
      transfer_ = Transfer::Collect;
      return;
    }
    if (transfer_ == Transfer::Collect) {
      auto &m = mems[packet_.source.hemisphere];
      auto &f = fs[packet_.source.hemisphere];
      StreamInputPort in(
          f, m.ports().output_column(packet_.source.slice, stream.direction()),
          stream.direction(), "Routing Gather collector");
      if (tile_ < hw::kTileRows && in.segment_valid(tile_, stream_index)) {
        const auto cells = in.consume_segment(tile_, stream_index);
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
          if (cells[lane].vector_tag != tag_)
            throw std::logic_error("routing Gather stale/misaligned data");
          payload_[tile_][lane] = cells[lane].data;
        }
        ++tile_;
      } else if (tile_ < hw::kTileRows)
        ++stats_.stream_wait_cycles;
      if (tile_ == hw::kTileRows &&
          m.routing_bank_idle(packet_.source.slice, packet_.source.bank)) {
        tile_ = 0;
        write_issued_ = false;
        transfer_ = Transfer::Emit;
      }
      return;
    }
    if (transfer_ == Transfer::Emit) {
      auto &m = mems[packet_.destination.hemisphere];
      auto &f = fs[packet_.destination.hemisphere];
      const auto column =
          m.ports().input_column(packet_.destination.slice, stream.direction());
      // Reserve a complete destination burst before presenting its first
      // segment. Ordinary MEM Write then consumes one segment each cycle.
      // E31 at these boundaries belongs exclusively to this transfer.
      if (tile_ == 0)
        for (std::size_t t = 0; t < hw::kTileRows; ++t)
          if (f.segment_valid(column, t, stream) ||
              f.staged_segment_valid(column, t, stream)) {
            ++stats_.stream_wait_cycles;
            return;
          }
      if (tile_ && !write_issued_) {
        auto i = MemInstruction::Write(packet_.destination_row, stream);
        i.routing_owner = owner_;
        i.routing_transfer = tag_;
        m.enqueue_instruction(packet_.destination.slice,
                              packet_.destination.bank, i);
        write_issued_ = true;
      }
      if (tile_ < hw::kTileRows) {
        if (f.segment_valid(column, tile_, stream) ||
            f.staged_segment_valid(column, tile_, stream))
          throw std::logic_error("routing reserved SR burst collision");
        StreamOutputPort out(f, column, stream.direction(),
                             "Routing Gather SR bridge");
        out.write_payload_segment(tile_, stream_index, payload_[tile_], tag_);
        ++tile_;
      }
      if (tile_ == hw::kTileRows && write_issued_)
        transfer_ = Transfer::Commit;
      return;
    }
    auto &m = mems[packet_.destination.hemisphere];
    if (m.routing_commit_count(owner_, tag_) == hw::kTileRows &&
        m.routing_bank_idle(packet_.destination.slice,
                            packet_.destination.bank)) {
      m.retire_routing_commit(owner_, tag_);
      ++stats_.gathered_vectors;
      stats_.commit_segments += hw::kTileRows;
      transfer_ = Transfer::Idle;
    }
  }

  RoutingConfig config_{};
  bool cycle_trace_enabled_{};
  CycleTrace cycle_trace_{};
  RoutingStatistics stats_{};
  std::uint64_t clock_{};
  std::deque<RoutingTraceEntry> trace_;
  RoutingInstruction instruction_{};
  RoutingDescriptor descriptor_{};
  State state_{State::Idle}, next_{State::Idle}, resume_{State::Idle};
  std::uint64_t generation_{}, owner_{}, tag_{}, next_tag_{}, next_owner_{};
  std::optional<RoutingCompletion> completion_{};
  std::vector<std::tuple<std::size_t, std::size_t, std::size_t>> banks_{};
  std::uint32_t context_{}, t_{}, e_{}, k_{};
  bool selected_{}, planned_{};
  bool plan_pinned_{};
  std::uint64_t plan_epoch_{}, internal_task_id_{};
  std::optional<RouteMetadataRequest> metadata_request_;
  std::optional<std::uint32_t> metadata_response_;
  std::array<Candidate, RoutingConfig::max_k> candidates_{};
  Candidate incoming_{};
  std::vector<std::uint32_t> ids_, raw_, counts_, offsets_, cursors_, active_;
  std::vector<RoutingAssignment> assignments_;
  std::vector<std::uint32_t> inverse_; // <=128*16 u32, single metadata-port access per cycle
  std::size_t token_{}, expert_{}, slot_{}, compare_{}, pos_{}, dest_{}, row_{},
      feature_{}, plane_{}, base_assignment_{};
  std::uint32_t temp_{}, sum_{}, current_expert_{}, weight_{}, source_token_{};
  struct Io {
    RoutingTensorView view;
    std::size_t row, column, plane;
    std::uint32_t bits;
    bool write, waiting, writing;
    StreamPayloadTileSegment segment;
  };
  Io io_{};
  std::uint32_t io_value_{};
  RouteBlockInfo block_{};
  std::deque<AddressPacket> address_fifo_{};
  AddressPacket packet_{};
  Transfer transfer_{Transfer::Idle};
  std::array<StreamPayloadTileSegment, hw::kTileRows> payload_{};
  std::size_t tile_{};
  bool write_issued_{};
};
} // namespace ftlpu
