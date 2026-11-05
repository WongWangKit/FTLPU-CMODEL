#pragma once
#include "ftlpu/c2c/c2c.hpp"
#include "ftlpu/c2c/dma.hpp"
#include "ftlpu/c2c/icu_instruction.hpp"
#include "ftlpu/routing/expert_task_controller.hpp"
#include <tuple>

namespace ftlpu {
struct ExpertWeightStatistics {
  std::uint64_t table_reads{}, dma_segments{}, committed_vectors{},
      residency_hits{}, wait_cycles{};
};

// Area-oriented indirect control adapter. Uses the existing RX and MEM Write
// datapaths; one RX notification/vector may be outstanding. No host pager runs.
class ExpertWeightLoader final : public ExpertWeightService {
public:
  static constexpr std::uint64_t owner = UINT64_MAX;
  static constexpr std::uint32_t sync_tag = 65535;
  bool idle() const { return state_ == State::Idle && banks_.empty(); }
  const ExpertWeightStatistics &statistics() const { return statistics_; }
  void configure(const ExpertWeightTableDescriptor *table) {
    if (!idle())
      throw std::logic_error("configure busy expert weight loader");
    table_ = table;
    resident_ = false;
  }
  void reset() { *this = ExpertWeightLoader{}; }
  void start(const ExpertTaskContext &task) override {
    if (state_ != State::Idle || !table_ || task.expert_id >= table_->experts ||
        !task.identity.task_id)
      throw std::logic_error("invalid expert weight task");
    task_ = task;
    if (resident_ &&
        resident_identity_.generation == task.identity.generation &&
        resident_identity_.plan_epoch == task.identity.plan_epoch &&
        resident_expert_ == task.expert_id) {
      ++statistics_.residency_hits;
      state_ = State::Ready;
      return;
    }
    if (!banks_.empty())
      throw std::logic_error("previous expert weight lease not released");
    resident_ = false;
    field_ = 0;
    state_ = State::Entry;
  }
  bool ready(ExpertTaskIdentity identity) const override {
    if (identity.generation < task_.identity.generation ||
        (identity.generation == task_.identity.generation &&
         identity.task_id < task_.identity.task_id))
      return false;
    if (identity != task_.identity)
      throw std::logic_error("wrong weight completion identity");
    return state_ == State::Ready;
  }
  void release_task(ExpertTaskIdentity identity) override {
    if (identity != task_.identity || state_ != State::Ready)
      throw std::logic_error("release unfinished weights");
    state_ = task_.more ? State::Idle : State::Release;
  }
  bool matches(std::size_t hemisphere, std::uint32_t tag) const {
    return state_ == State::Transfer && hemisphere == segment_.hemisphere &&
           tag == sync_tag;
  }
  bool can_receive() const {
    return state_ == State::Transfer && !pending_vector_ &&
           received_ < segment_.vector_count;
  }
  void notify_vector(std::uint64_t vector_tag) {
    if (!can_receive())
      throw std::logic_error("weight RX without credit");
    pending_vector_ = true;
    write_issued_ = false;
    pending_tag_ = vector_tag;
    due_ = clock_ + hw::kMemEastBoundaryStreamRegisterColumn -
           (segment_.slice / hw::kMemSlicesPerGroup + 1) + 2;
    ++received_;
  }
  void tick(std::array<MemArrayModel, hw::kHemispheres> &mems,
            const std::array<C2cEndpoint *, hw::kHemispheres> &endpoints,
            const std::array<C2cDmaEngine *, hw::kHemispheres> &dmas) {
    ++clock_;
    // A single descriptor-RAM request and response slot; response appears
    // on a later tick and remains until consumed. One u32 read per cycle.
    if (read_request_) {
      read_response_ = table_->words.at(*read_request_);
      read_request_.reset();
      ++statistics_.table_reads;
    }
    switch (state_) {
    case State::Idle:
    case State::Ready:
      break;
    case State::Entry:
      if (auto value = read(2 * task_.expert_id + field_)) {
        entry_[field_++] = *value;
        if (field_ == 2) {
          if (!entry_[1] ||
              std::uint64_t(entry_[0]) + entry_[1] > table_->segments)
            throw std::logic_error("invalid weight entry");
          field_ = 0;
          segment_index_ = 0;
          state_ = State::Scan;
        }
      }
      break;
    case State::Scan:
    case State::LoadSegment:
      if (auto value = read(2 * table_->experts +
                            8 * (entry_[0] + segment_index_) + field_)) {
        raw_[field_++] = *value;
        if (field_ == 8) {
          segment_ = decode_expert_weight_segment(raw_);
          field_ = 0;
          if (state_ == State::Scan) {
            banks_.emplace_back(segment_.hemisphere, segment_.slice,
                                segment_.bank);
            if (++segment_index_ == entry_[1]) {
              std::sort(banks_.begin(), banks_.end());
              banks_.erase(std::unique(banks_.begin(), banks_.end()),
                           banks_.end());
              segment_index_ = 0;
              state_ = State::Acquire;
            }
          } else
            state_ = State::Submit;
        }
      }
      break;
    case State::Acquire:
      for (auto [h, s, b] : banks_)
        if (!mems[h].routing_bank_available(s, b)) {
          ++statistics_.wait_cycles;
          return;
        }
      for (auto [h, s, b] : banks_)
        if (!mems[h].acquire_routing_bank(s, b, owner))
          throw std::logic_error("weight bank lease race");
      state_ = State::LoadSegment;
      break;
    case State::Submit: {
      auto *dma = dmas[segment_.hemisphere];
      auto *endpoint = endpoints[segment_.hemisphere];
      if (!dma || !endpoint)
        throw std::logic_error("weights require real C2C DMA");
      if (!dma->drained() || !endpoint->rx().idle()) {
        ++statistics_.wait_cycles;
        break;
      }
      // RX is installed before DMA. Lane 0 and the configured West SR burst
      // are reserved by the program for this serial transfer.
      const auto rx = C2cRxIcuInstruction::Receive(
          static_cast<Hemisphere>(segment_.hemisphere), 0,
          segment_.fabric_stream, segment_.vector_count, sync_tag,
          C2cMemNotifyRoute::Mem(static_cast<Hemisphere>(segment_.hemisphere),
                                 segment_.slice, segment_.bank));
      (void)C2cIcuPacketCodec::encode(rx);
      endpoint->rx().issue(rx.to_legacy(segment_.base_row));
      auto load =
          C2cDmaInstruction::Load(segment_.ddr_address, segment_.vector_count,
                                  hw::kPhysicalVectorBytes, ++next_tag_, 0);
      next_tag_ += segment_.vector_count;
      load.sync_tag = sync_tag;
      dma->issue(load);
      received_ = written_ = 0;
      pending_vector_ = false;
      ++statistics_.dma_segments;
      state_ = State::Transfer;
      break;
    }
    case State::Transfer: {
      auto &mem = mems[segment_.hemisphere];
      if (pending_vector_) {
        if (!write_issued_ && clock_ >= due_) {
          auto write =
              MemInstruction::Write(segment_.base_row + written_,
                                    StreamId::West(segment_.fabric_stream));
          write.routing_owner = owner;
          write.routing_transfer = pending_tag_;
          mem.enqueue_instruction(segment_.slice, segment_.bank, write);
          write_issued_ = true;
        } else if (write_issued_ &&
                   mem.routing_commit_count(owner, pending_tag_) ==
                       hw::kTileRows &&
                   mem.routing_bank_idle(segment_.slice, segment_.bank)) {
          mem.retire_routing_commit(owner, pending_tag_);
          pending_vector_ = false;
          ++written_;
          ++statistics_.committed_vectors;
        }
      }
      if (written_ == segment_.vector_count && !pending_vector_ &&
          dmas[segment_.hemisphere]->drained() &&
          endpoints[segment_.hemisphere]->rx().idle()) {
        if (++segment_index_ == entry_[1]) {
          for (auto [h, s, b] : banks_)
            mems[h].retain_weight_bank(s, b, owner);
          resident_ = true;
          resident_identity_ = task_.identity;
          resident_expert_ = task_.expert_id;
          state_ = State::Ready;
        } else
          state_ = State::LoadSegment;
      }
      break;
    }
    case State::Release:
      for (auto [h, s, b] : banks_)
        if (!mems[h].routing_bank_idle(s, b)) {
          ++statistics_.wait_cycles;
          return;
        }
      for (auto [h, s, b] : banks_)
        mems[h].release_weight_bank(s, b);
      banks_.clear();
      resident_ = false;
      state_ = State::Idle;
      break;
    }
  }

private:
  std::optional<std::uint32_t> read(std::size_t address) {
    if (read_response_) {
      auto result = read_response_;
      read_response_.reset();
      return result;
    }
    if (!read_request_)
      read_request_ = address;
    return {};
  }
  enum class State {
    Idle,
    Entry,
    Scan,
    Acquire,
    LoadSegment,
    Submit,
    Transfer,
    Ready,
    Release
  };
  State state_{State::Idle};
  const ExpertWeightTableDescriptor *table_{};
  ExpertTaskContext task_{};
  ExpertTaskIdentity resident_identity_{};
  ExpertWeightStatistics statistics_{};
  std::array<std::uint32_t, 2> entry_{};
  std::array<std::uint32_t, 8> raw_{};
  ExpertWeightSegment segment_{};
  std::vector<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t>> banks_;
  std::optional<std::size_t> read_request_;
  std::optional<std::uint32_t> read_response_;
  std::uint64_t clock_{}, due_{}, pending_tag_{}, next_tag_{};
  std::uint32_t field_{}, segment_index_{}, received_{}, written_{},
      resident_expert_{};
  bool pending_vector_{}, write_issued_{}, resident_{};
};
} // namespace ftlpu
