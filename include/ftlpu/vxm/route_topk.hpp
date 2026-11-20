#pragma once

#include "ftlpu/core/hemisphere.hpp"
#include "ftlpu/routing/instruction.hpp"
#include "ftlpu/vxm/input_buffer.hpp"
#include "ftlpu/vxm/topk_instruction.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <optional>

namespace ftlpu {

// The scoring path borrows the eight pairs in each physical lane. Local ALU
// latches and scalar registers hold candidates, and this controller owns the
// input phase, rank and bounded result buffer.
class VxmRouteTopK {
public:
    static constexpr std::size_t chains = RouteVxmSelectSchedule::chains;
    static constexpr std::size_t lanes = hw::kTileRows * hw::kLanesPerTile;
    static constexpr std::size_t scan_to_result_cycles = chains + 1;
    static constexpr std::size_t merge_input_gap = scan_to_result_cycles + 1;
    using Word = std::uint32_t;

    struct Statistics {
        std::size_t batches{}, score_bundles{}, eligibility_bundles{}, raw_bundles{};
        std::size_t input_wait_cycles{}, filter_comparisons{}, accumulator_comparisons{};
        std::size_t merge_steps{}, submitted_words{}, submit_wait_cycles{}, output_tiles{};
    };

    void begin(const RoutingDescriptor& d, std::uint64_t generation, Word completion)
    {
        std::uint16_t west_groups{};
        for (std::size_t group = 0; group < VxmLane::kStreamGroupCount; ++group)
            if (d.vxm_select.score_planes[group * VxmLane::kStreamGroupBytes].hemisphere
                    == hemisphere_index(Hemisphere::West))
                west_groups |= std::uint16_t(1u << group);
        start({d.select.tokens, d.select.experts, d.select.k,
               west_groups, d.select.eligibility.present()},
              Handoff{generation, d.select.context_id, completion},
              d.select.raw_scores.present() && d.select.raw_scores != d.select.choice_scores);
    }

    void begin(VxmTopKInstruction instruction) { start(instruction, std::nullopt, false); }

    bool active() const { return active_; }
    const Statistics& statistics() const { return stats_; }
    Hemisphere group_source(std::size_t group) const
    {
        return (instruction_.west_input_groups & (1u << group))
            ? Hemisphere::West : Hemisphere::East;
    }

    bool wants_input(std::size_t tile) const
    {
        return active_ && tiles_.at(tile).state == State::Scan;
    }

    template<class LaneAt>
    void tick_tile(std::size_t tile, VxmInputBuffer& buffer, LaneAt lane_at)
    {
        auto& progress = tiles_.at(tile);
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
            stats_.accumulator_comparisons += lane_at(lane).retire_route_candidates();
        switch (progress.state) {
        case State::Begin:
            for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
                lane_at(lane).start_route_token();
            progress.state = State::Scan;
            break;
        case State::Scan:
            if (buffer.ready()) {
                scan(tile, progress, buffer.bundle(), lane_at);
                buffer.release_after_issue();
            } else {
                ++stats_.input_wait_cycles;
            }
            break;
        case State::Drain:
            progress.merge_chain = 1;
            progress.state = State::Merge;
            break;
        case State::Merge:
            if (progress.merge_chain < chains) {
                for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
                    lane_at(lane).route_merge(progress.merge_chain);
                ++progress.merge_chain;
                ++stats_.merge_steps;
            } else {
                for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
                    auto& alu_lane = lane_at(lane);
                    const auto candidate = alu_lane.finish_route_round();
                    const auto local = tile * hw::kLanesPerTile + lane;
                    if (local < batch_rows()) {
                        if (!candidate.valid)
                            throw std::invalid_argument("VXM routing has fewer eligible experts than K");
                        results_[local * instruction_.k + progress.rank] =
                            {candidate.expert, std::bit_cast<Word>(candidate.score)};
                    }
                    alu_lane.start_route_round();
                }
                if (++progress.rank == instruction_.k) {
                    progress.state = State::Done;
                } else {
                    progress.block = 0;
                    progress.part = first_part(progress.rank);
                    progress.state = State::Scan;
                }
            }
            break;
        case State::Done:
            break;
        }
    }

    std::optional<VxmSelectionWord> submission_word() const
    {
        if (!handoff_ || !results_ready()) return std::nullopt;
        const auto& entry = results_[submitted_ / 2];
        return VxmSelectionWord{handoff_->generation, handoff_->context, handoff_->completion,
            batch_base_ * instruction_.k * 2 + submitted_,
            submitted_ % 2 ? entry.bits : entry.expert};
    }

    void accept_submission()
    {
        if (!submission_word())
            throw std::logic_error("VXM Top-K submission without a word");
        ++stats_.submitted_words;
        if (++submitted_ == 2 * batch_rows() * instruction_.k) advance_batch();
    }
    void wait_submission() { ++stats_.submit_wait_cycles; }

    struct Result { Word expert{}, bits{}; };

    std::optional<std::size_t> stream_rank(std::size_t tile) const
    {
        if (handoff_ || !results_ready() || stream_cycle_ < tile ||
            stream_cycle_ - tile >= instruction_.k) return std::nullopt;
        return stream_cycle_ - tile;
    }

    Result result(std::size_t local_row, std::size_t rank) const
    {
        return local_row < batch_rows() ? results_[local_row * instruction_.k + rank] : Result{};
    }

    void tick_stream_output()
    {
        if (handoff_ || !results_ready()) return;
        for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
            stats_.output_tiles += stream_rank(tile).has_value();
        if (++stream_cycle_ == instruction_.k + hw::kTileRows - 1) advance_batch();
    }

private:
    enum class State { Begin, Scan, Drain, Merge, Done };
    enum class Part { Eligibility, Raw, Score };
    struct Progress {
        State state{State::Begin};
        Part part{Part::Score};
        std::size_t block{}, rank{}, merge_chain{};
        std::array<std::array<bool, chains>, hw::kLanesPerTile> eligible{};
    };
    struct Handoff {
        std::uint64_t generation;
        Word context, completion;
    };

    void start(VxmTopKInstruction instruction, std::optional<Handoff> handoff, bool check_raw)
    {
        if (active_) throw std::logic_error("VXM Top-K is busy");
        (void)encode_vxm_topk(instruction);
        instruction_ = instruction;
        handoff_ = handoff;
        has_distinct_raw_ = check_raw;
        active_ = true;
        batch_base_ = 0;
        stats_ = {};
        start_batch();
    }

    bool results_ready() const
    {
        return active_ && std::ranges::all_of(tiles_, [](const auto& tile) {
            return tile.state == State::Done;
        });
    }

    void advance_batch()
    {
        batch_base_ += batch_rows();
        ++stats_.batches;
        if (batch_base_ == instruction_.rows) active_ = false;
        else start_batch();
    }

    std::size_t batch_rows() const
    {
        return std::min(lanes, std::size_t(instruction_.rows) - batch_base_);
    }

    bool check_raw(std::size_t rank) const
    {
        return !rank && has_distinct_raw_;
    }

    Part first_part(std::size_t rank) const
    {
        if (instruction_.has_mask) return Part::Eligibility;
        return check_raw(rank) ? Part::Raw : Part::Score;
    }

    void start_batch()
    {
        submitted_ = stream_cycle_ = 0;
        for (auto& tile : tiles_) {
            tile = {};
            tile.part = first_part(0);
        }
    }

    static Word bits_at(const VxmLane::StreamBytes& bytes, std::size_t chain)
    {
        Word result = 0;
        for (std::size_t p = 0; p < RouteVxmSelectSchedule::bytes_per_score; ++p)
            result |= Word(bytes[4 * chain + p]) << (8 * p);
        return result;
    }

    template<class LaneAt>
    void scan(std::size_t tile, Progress& progress,
              const VxmInputBuffer::StreamMatrix& bundle, LaneAt lane_at)
    {
        if (progress.part == Part::Eligibility) ++stats_.eligibility_bundles;
        else if (progress.part == Part::Raw) ++stats_.raw_bundles;
        else ++stats_.score_bundles;
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane) {
            const bool live = tile * hw::kLanesPerTile + lane < batch_rows();
            for (std::size_t chain = 0; chain < chains; ++chain) {
                const auto expert = static_cast<Word>(progress.block * chains + chain);
                const auto bits = bits_at(bundle[lane], chain);
                if (progress.part == Part::Eligibility) {
                    progress.eligible[lane][chain] = bits != 0;
                    continue;
                }
                const bool eligible = live && expert < instruction_.columns
                    && (!instruction_.has_mask || progress.eligible[lane][chain]);
                const auto score = std::bit_cast<float>(bits);
                if (eligible && std::isnan(score))
                    throw std::invalid_argument(progress.part == Part::Raw
                        ? "NaN routing raw score" : "NaN VXM Top-K value");
                if (progress.part == Part::Score) {
                    stats_.filter_comparisons += eligible && progress.rank != 0;
                    lane_at(lane).route_filter(chain, {score, expert, eligible});
                }
            }
        }
        if (progress.part == Part::Eligibility) {
            progress.part = check_raw(progress.rank) ? Part::Raw : Part::Score;
        } else if (progress.part == Part::Raw) {
            progress.part = Part::Score;
        } else if (++progress.block * chains >= instruction_.columns) {
            progress.state = State::Drain;
        } else {
            progress.part = first_part(progress.rank);
        }
    }

    static_assert(VxmLane::kBlockCount == chains);
    static_assert(VxmLane::kInputStreams >= RouteVxmSelectSchedule::streams);

    VxmTopKInstruction instruction_{};
    std::optional<Handoff> handoff_;
    std::array<Progress, hw::kTileRows> tiles_{};
    std::array<Result, lanes * VxmTopKInstruction::max_k> results_{};
    std::size_t batch_base_{}, submitted_{}, stream_cycle_{};
    bool active_{}, has_distinct_raw_{};
    Statistics stats_{};
};

} // namespace ftlpu
