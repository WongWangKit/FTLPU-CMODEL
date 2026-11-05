#pragma once
#include "ftlpu/routing/control_unit.hpp"
#include "ftlpu/vxm/slice.hpp"
#include <tuple>

namespace ftlpu {
// One token/feature block at a time. No activation arithmetic is performed by
// this controller: the shared VXM executes Multiply -> accumulator Add.
class RoutingCombineController {
public:
  struct CycleTrace {bool valid{};const char *step{"Idle"};std::uint32_t token{},slot{},feature{},assignment{},weight_bits{};};
  const CycleTrace &last_cycle_trace() const {return cycle_trace_;}
  struct Statistics {
    std::uint64_t cycles{}, bank_wait{}, stream_wait{}, contributions{},
        gathered_vectors{}, committed_segments{};
  };
  bool idle() const { return state_ == State::Idle && !completion_; }
  void reset() { *this = RoutingCombineController{}; }
  const Statistics &statistics() const { return stats_; }
  void invalidate_outputs() {
    if (!idle())
      throw std::logic_error("invalidate active combine");
    outputs_ready_ = false;
  }
  void publish_outputs(std::uint64_t generation, std::uint64_t epoch,
                       std::uint32_t context, const RoutingTensorView &view) {
    if (!idle())
      throw std::logic_error("publish outputs during combine");
    ready_generation_ = generation;
    ready_epoch_ = epoch;
    ready_context_ = context;
    ready_view_ = view;
    outputs_ready_ = true;
  }
  void start(RoutingInstruction i, const RouteCombineDescriptor &d,
             std::uint64_t generation, RoutingControlUnit &rcu) {
    validate_route_combine(d);
    if (!idle() || i.opcode != RoutingOpcode::Combine || !outputs_ready_ ||
        generation != ready_generation_ || d.context_id != ready_context_ ||
        d.source != ready_view_ || rcu.plan_epoch() != ready_epoch_)
      throw std::logic_error(
          "combine requires committed outputs from the current plan");
    const auto epoch = rcu.pin_plan(generation, d.context_id);
    if (epoch != ready_epoch_)
      throw std::logic_error("stale combine plan");
    instruction_ = i;
    descriptor_ = d;
    generation_ = generation;
    stats_ = {};
    token_ = feature_ = slot_ = plane_ = tile_ = 0;
    pending_metadata_ = false;
    banks_.clear();
    for (const auto *v : {&d.source, &d.destination})
      for (std::size_t p = 0; p < 2; ++p) {
        const auto q = v->planes[p];
        const auto key = std::tuple(q.hemisphere, q.slice, q.bank);
        if (std::find(banks_.begin(), banks_.end(), key) == banks_.end())
          banks_.push_back(key);
      }
    state_ = State::Tokens;
  }
  std::optional<RoutingCompletion> take_completion() {
    auto value = completion_;
    completion_.reset();
    return value;
  }
  void tick(RoutingControlUnit &rcu,
            std::array<MemArrayModel, hw::kHemispheres> &mems,
            std::array<StreamRegisterFabric, hw::kHemispheres> &fabric,
            VxmSlice &vxm, bool drained, std::size_t input_column,
            std::size_t output_column) {
    cycle_trace_={state_!=State::Idle,state_name(state_),token_,slot_,feature_,assignment_,weight_};
    if (state_ == State::Idle)
      return;
    ++stats_.cycles;
    // Capture results concurrently with feeding later tiles; a current SR
    // segment advances at the end of this physical cycle and cannot be held.
    if ((state_ == State::Feed || state_ == State::Compute) &&
        slot_ + 1 == k_ && result_tile_ < hw::kTileRows) {
      StreamInputPort input(fabric[0], output_column, StreamDirection::East,
                            "combine VXM result collector");
      if (input.segment_valid(result_tile_, 0) &&
          input.segment_valid(result_tile_, 1)) {
        for (std::size_t p = 0; p < 2; ++p) {
          const auto cells = input.consume_segment(result_tile_, p);
          for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
            payload_[p][result_tile_][lane] = cells[lane].data;
        }
        ++result_tile_;
      }
    }
    switch (state_) {
    case State::Tokens:
      if (auto x = metadata(rcu, RouteMetadataKind::Tokens, 0)) {
        tokens_ = *x;
        state_ = State::K;
      }
      break;
    case State::K:
      if (auto x = metadata(rcu, RouteMetadataKind::K, 0)) {
        k_ = *x;
        if (!tokens_ || !k_ ||
            descriptor_.source.rows != std::uint64_t(tokens_) * k_ ||
            descriptor_.destination.rows != tokens_)
          throw std::logic_error("combine plan shape mismatch");
        state_ = State::Acquire;
      }
      break;
    case State::Acquire: {
      if (!drained) {
        ++stats_.bank_wait;
        break;
      }
      bool available = true;
      for (const auto &[h, s, b] : banks_)
        available &= mems[h].routing_bank_available(s, b);
      if (!available || !vxm.acquire_control(owner_)) {
        ++stats_.bank_wait;
        break;
      }
      for (const auto &[h, s, b] : banks_)
        if (!mems[h].acquire_routing_bank(s, b, owner_))
          throw std::logic_error("combine bank lease");
      saved_input_[0] = vxm.input_group_source(0);
      saved_input_[1] = vxm.input_group_source(8);
      saved_output_[0] = vxm.output_block_destination(0);
      saved_output_[1] = vxm.output_block_destination(4);
      vxm.configure_input_group_source(0, Hemisphere::East);
      vxm.configure_input_group_source(8, Hemisphere::East);
      vxm.configure_output_block_destination(0, Hemisphere::East);
      vxm.configure_output_block_destination(4, Hemisphere::East);
      state_ = State::Inverse;
      break;
    }
    case State::Inverse:
      if (auto x =
              metadata(rcu, RouteMetadataKind::Inverse, token_ * k_ + slot_)) {
        assignment_ = *x;
        if (assignment_ >= descriptor_.source.rows)
          throw std::logic_error("combine inverse index out of range");
        state_ = State::Weight;
      }
      break;
    case State::Weight:
      if (auto x =
              metadata(rcu, RouteMetadataKind::AssignmentWeight, assignment_)) {
        weight_ = *x;
        plane_ = tile_ = 0;
        state_ = State::Gather;
      }
      break;
    case State::Gather: {
      const auto q = descriptor_.source.planes[plane_];
      auto &m = mems[q.hemisphere];
      if (!m.routing_bank_idle(q.slice, q.bank)) {
        ++stats_.bank_wait;
        break;
      }
      const auto col = m.ports().output_column(q.slice, StreamDirection::East);
      if (!burst_free(fabric[q.hemisphere], col, StreamId::East(31))) {
        ++stats_.stream_wait;
        break;
      }
      tag_ = ++next_tag_;
      if (!m.push_gather_address(
              q.slice, q.bank,
              descriptor_.source.row(assignment_, feature_, plane_), owner_,
              tag_)) {
        ++stats_.bank_wait;
        break;
      }
      auto instruction =
          MemInstruction::Gather(StreamId::East(31), StreamId::East(0));
      instruction.routing_owner = owner_;
      instruction.routing_transfer = tag_;
      m.enqueue_instruction(q.slice, q.bank, instruction);
      tile_ = 0;
      state_ = State::Collect;
      break;
    }
    case State::Collect: {
      const auto q = descriptor_.source.planes[plane_];
      auto &m = mems[q.hemisphere];
      StreamInputPort input(
          fabric[q.hemisphere],
          m.ports().output_column(q.slice, StreamDirection::East),
          StreamDirection::East, "combine Gather collector");
      if (tile_ < hw::kTileRows && input.segment_valid(tile_, 31)) {
        const auto cells = input.consume_segment(tile_, 31);
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
          if (cells[lane].vector_tag != tag_)
            throw std::logic_error("combine Gather vector tag mismatch");
          payload_[plane_][tile_][lane] = cells[lane].data;
        }
        ++tile_;
      } else if (tile_ < hw::kTileRows)
        ++stats_.stream_wait;
      if (tile_ == hw::kTileRows && m.routing_bank_idle(q.slice, q.bank)) {
        ++stats_.gathered_vectors;
        tile_ = 0;
        if (++plane_ == 2) {
          plane_ = 0;
          state_ = State::Launch;
        } else
          state_ = State::Gather;
      }
      break;
    }
    case State::Launch: {
      if (!vxm.idle()) {
        ++stats_.bank_wait;
        break;
      }
      if (!burst_free(fabric[0], input_column, StreamId::West(0)) ||
          !burst_free(fabric[0], input_column, StreamId::West(1)) ||
          !burst_free(fabric[0], input_column, StreamId::West(16)) ||
          !burst_free(fabric[0], input_column, StreamId::West(17))) {
        ++stats_.stream_wait;
        break;
      }
      VxmLaneAluInstruction mul;
      mul.operation = VxmAluOpcode::Multiply;
      mul.precision = VxmAluPrecision::Float32;
      mul.lhs = descriptor_.source.element == RoutingElement::BF16
                    ? VxmLaneOperand::StreamBFloat16()
                    : VxmLaneOperand::StreamFloat16();
      mul.rhs = VxmLaneOperand::Imm(std::bit_cast<float>(weight_));
      mul.output_type = descriptor_.destination.element == RoutingElement::BF16
                            ? VxmCastTarget::BFloat16
                            : VxmCastTarget::Float16;
      auto add = mul;
      add.operation = VxmAluOpcode::Add;
      add.lhs = VxmLaneOperand::Previous();
      add.rhs = VxmLaneOperand::Acc();
      add.accumulator_write = true;
      add.accumulator_reset = slot_ == 0;
      add.accumulator_emit = slot_ + 1 == k_;
      if (slot_ + 1 == k_)
        add.output_stream = 0;
      vxm.issue_south(
          0, VxmCompactInstructionCodec::encode(0, VxmChainDepth::Two, mul),
          owner_);
      vxm.issue_south(
          1, VxmCompactInstructionCodec::encode(1, VxmChainDepth::Two, add),
          owner_);
      tile_ = result_tile_ = 0;
      state_ = State::Feed;
      // Config decode is one cycle. Present the first SR tile now; it is
      // visible to the existing VXM input capture on the following cycle.
      feed(fabric[0], input_column);
      break;
    }
    case State::Feed:
      feed(fabric[0], input_column);
      break;
    case State::Compute:
      if (slot_ + 1 != k_) {
        if (drained) {
          ++slot_;
          ++stats_.contributions;
          state_ = State::Inverse;
        }
        break;
      }
      if (result_tile_ == hw::kTileRows && vxm.idle()) {
        ++stats_.contributions;
        plane_ = tile_ = 0;
        state_ = State::WriteBegin;
      }
      break;
    case State::WriteBegin: {
      const auto q = descriptor_.destination.planes[plane_];
      auto &m = mems[q.hemisphere];
      const auto col = m.ports().input_column(q.slice, StreamDirection::East);
      if (!m.routing_bank_idle(q.slice, q.bank) ||
          !burst_free(fabric[q.hemisphere], col, StreamId::East(31))) {
        ++stats_.bank_wait;
        break;
      }
      tag_ = ++next_tag_;
      tile_ = 0;
      write_issued_ = false;
      state_ = State::Write;
      break;
    }
    case State::Write: {
      const auto q = descriptor_.destination.planes[plane_];
      auto &m = mems[q.hemisphere];
      const auto col = m.ports().input_column(q.slice, StreamDirection::East);
      auto &f = fabric[q.hemisphere];
      if (tile_ && !write_issued_) {
        auto i = MemInstruction::Write(
            descriptor_.destination.row(token_, feature_, plane_),
            StreamId::East(31));
        i.routing_owner = owner_;
        i.routing_transfer = tag_;
        m.enqueue_instruction(q.slice, q.bank, i);
        write_issued_ = true;
      }
      if (tile_ < hw::kTileRows) {
        if (f.segment_valid(col, tile_, StreamId::East(31)) ||
            f.staged_segment_valid(col, tile_, StreamId::East(31)))
          throw std::logic_error("combine reserved destination SR collision");
        StreamOutputPort out(f, col, StreamDirection::East,
                             "combine output SR bridge");
        out.write_payload_segment(tile_, 31, payload_[plane_][tile_], tag_);
        ++tile_;
      } else
        state_ = State::Commit;
      break;
    }
    case State::Commit: {
      const auto q = descriptor_.destination.planes[plane_];
      auto &m = mems[q.hemisphere];
      const auto count = m.routing_commit_count(owner_, tag_);
      if (count > hw::kTileRows)
        throw std::logic_error("duplicate combine output commit");
      if (count != hw::kTileRows || !m.routing_bank_idle(q.slice, q.bank))
        break;
      m.retire_routing_commit(owner_, tag_);
      stats_.committed_segments += count;
      if (++plane_ < 2) {
        state_ = State::WriteBegin;
        break;
      }
      slot_ = 0;
      feature_ += hw::kPhysicalVectorBytes;
      if (feature_ == descriptor_.source.columns) {
        feature_ = 0;
        ++token_;
      }
      state_ = token_ == tokens_ ? State::Drain : State::Next;
      break;
    }
    case State::Next:
      if (drained)
        state_ = State::Inverse;
      break;
    case State::Drain:
      if (drained) {
        for (const auto &[h, s, b] : banks_)
          mems[h].release_routing_bank(s, b, owner_);
        vxm.configure_input_group_source(0, saved_input_[0]);
        vxm.configure_input_group_source(8, saved_input_[1]);
        vxm.configure_output_block_destination(0, saved_output_[0]);
        vxm.configure_output_block_destination(4, saved_output_[1]);
        vxm.release_control(owner_);
        rcu.release_plan();
        outputs_ready_ = false;
        completion_ = RoutingCompletion{generation_, instruction_.completion,
                                        RoutingOpcode::Combine};
        state_ = State::Idle;
      }
      break;
    case State::Idle:
      break;
    }
  }

private:
  enum class State {
    Idle,
    Tokens,
    K,
    Acquire,
    Inverse,
    Weight,
    Gather,
    Collect,
    Launch,
    Feed,
    Compute,
    WriteBegin,
    Write,
    Commit,
    Next,
    Drain
  };
  static const char *state_name(State s) {
#define COMBINE_STATE_NAME(name) case State::name:return #name;
    switch(s){COMBINE_STATE_NAME(Idle) COMBINE_STATE_NAME(Tokens) COMBINE_STATE_NAME(K)
      COMBINE_STATE_NAME(Acquire) COMBINE_STATE_NAME(Inverse) COMBINE_STATE_NAME(Weight)
      COMBINE_STATE_NAME(Gather) COMBINE_STATE_NAME(Collect) COMBINE_STATE_NAME(Launch)
      COMBINE_STATE_NAME(Feed) COMBINE_STATE_NAME(Compute) COMBINE_STATE_NAME(WriteBegin)
      COMBINE_STATE_NAME(Write) COMBINE_STATE_NAME(Commit) COMBINE_STATE_NAME(Next) COMBINE_STATE_NAME(Drain)}
#undef COMBINE_STATE_NAME
    return "Unknown";
  }
  static bool burst_free(const StreamRegisterFabric &f, std::size_t column,
                         StreamId stream) {
    for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
      if (f.segment_valid(column, tile, stream) ||
          f.staged_segment_valid(column, tile, stream))
        return false;
    return true;
  }
  void feed(StreamRegisterFabric &f, std::size_t column) {
    StreamOutputPort out(f, column, StreamDirection::West,
                         "combine Gather-to-VXM SR bridge");
    for (unsigned p = 0; p < 2; ++p)
      for (unsigned offset : {0u, 16u})
        out.write_payload_segment(tile_, offset + p, payload_[p][tile_], tag_);
    if (++tile_ == hw::kTileRows) {
      tile_ = 0;
      state_ = State::Compute;
    }
  }
  std::optional<std::uint32_t> metadata(RoutingControlUnit &rcu,
                                        RouteMetadataKind kind,
                                        std::uint32_t index) {
    if (pending_metadata_) {
      auto x = rcu.take_metadata_response();
      if (x)
        pending_metadata_ = false;
      return x;
    }
    pending_metadata_ = rcu.request_metadata({kind, index});
    return {};
  }
  static constexpr std::uint64_t owner_ = UINT64_MAX - 2;
  State state_{State::Idle};
  CycleTrace cycle_trace_{};
  Statistics stats_{};
  RoutingInstruction instruction_;
  RouteCombineDescriptor descriptor_;
  std::optional<RoutingCompletion> completion_;
  bool outputs_ready_{}, pending_metadata_{}, write_issued_{};
  std::uint64_t ready_generation_{}, ready_epoch_{}, generation_{}, tag_{},
      next_tag_{};
  std::uint32_t ready_context_{}, tokens_{}, k_{}, token_{}, feature_{},
      slot_{}, assignment_{}, weight_{}, plane_{}, tile_{}, result_tile_{};
  RoutingTensorView ready_view_;
  std::vector<std::tuple<std::size_t, std::size_t, std::size_t>>
      banks_; // at most four
  std::array<Hemisphere, 2> saved_input_{}, saved_output_{};
  std::array<std::array<StreamPayloadTileSegment, hw::kTileRows>, 2>
      payload_{}; // 64 bytes
};
} // namespace ftlpu
