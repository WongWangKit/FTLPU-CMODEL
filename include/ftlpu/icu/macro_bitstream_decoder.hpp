#pragma once

#include "ftlpu/core/instruction_codec.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace ftlpu {

enum class IcuQueueMode : std::uint8_t {
    Native = 0,
    Macro = 1,
    Reserved2 = 2,
    Reserved3 = 3,
};

enum class IcuMacroQueueKind : std::uint8_t {
    Mem,
    MxmLoad,
    MxmCompute,
    MxmDequant,
};

template <std::size_t Bits>
struct IcuRawImemWord {
    static_assert(Bits >= 32 && Bits % 32 == 0);

    static constexpr std::size_t bit_count = Bits;
    static constexpr std::size_t lane_count = Bits / 32;
    std::array<std::uint32_t, lane_count> lanes{};

    bool bit(std::size_t index) const
    {
        if (index >= Bits)
            throw std::out_of_range(
                "ICU raw i-MEM bit index is outside the word");
        return ((lanes[index / 32] >> (index % 32)) & 1u) != 0;
    }
};

using IcuRawImemWord96 = IcuRawImemWord<96>;
using IcuRawImemWord128 = IcuRawImemWord<128>;

// All counters are cycle-accurate frontend counters. The legacy fields remain
// at the top so existing performance consumers keep source compatibility.
struct IcuMacroDecoderStatistics {
    std::size_t fetched_words{0};
    std::size_t decoded_contexts{0};
    std::size_t bit_wait_cycles{0};
    std::size_t context_stall_cycles{0};
    std::size_t peak_reservoir_bits{0};

    std::size_t cycles{0};
    std::size_t payload_bits_consumed{0};
    std::size_t reservoir_empty_cycles{0};
    std::size_t reservoir_full_cycles{0};
    std::size_t reservoir_occupancy_bit_samples{0};
    std::size_t decoder_active_cycles{0};
    std::size_t decoder_starvation_cycles{0};
    std::size_t decoder_ddb_stall_cycles{0};
    std::size_t descriptor_count{0};
    std::size_t mem_descriptor_count{0};
    std::size_t mxm_descriptor_count{0};
    std::size_t descriptor_decode_cycles{0};
    std::size_t max_descriptor_decode_cycles{0};
    std::size_t ddb_entries_committed{0};
    std::size_t peak_ddb_occupancy{0};
    std::size_t ddb_full_cycles{0};
    std::size_t ddb_occupancy_samples{0};
    std::size_t ddb_residency_cycles{0};
    std::size_t max_ddb_residency_cycles{0};
    std::size_t contexts_generated{0};
    std::size_t timing_window_block_cycles{0};
    std::size_t active_ram_full_cycles{0};
    std::size_t capacity_feasible_wait_cycles{0};
    std::size_t admission_count{0};
    std::size_t admission_stall_cycles{0};
    std::size_t admission_to_start_cycles{0};
    std::size_t max_admission_to_start_cycles{0};
    std::size_t active_occupancy_samples{0};
    std::size_t peak_active_occupancy{0};
    std::size_t context_active_lifetime_cycles{0};
    std::size_t max_context_active_lifetime_cycles{0};
};

template <typename FuncInstruction>
struct DecodedIcuMacroContext {
    std::size_t record_index{0};
    IcuMacroSchedule schedule{};
    FuncInstruction instruction{};
    std::size_t admission_cycle{0};
};

// Physical packed-v1 Macro frontend. The stages are advanced concurrently,
// once per tick:
//
//   fixed-width i-MEM -> word reservoir -> parser FSM -> finite DDB
//       -> one-context lazy expander / admission interface
//
// The reservoir never shifts all remaining bits. It retains complete words
// and advances only head_bit_offset; an aligned window reads at most the
// current and next word because DecodeWindowBits <= WordBits.
template <typename FuncInstruction,
          std::size_t WordBits,
          std::size_t FetchLatency = 1,
          std::size_t ReservoirWords = 3,
          std::size_t DecodeWindowBits = 64,
          std::size_t DdbDepth = 8,
          std::size_t AdmissionLookahead = 8,
          std::size_t DdbRunCapacity = 8>
class IcuMacroV1Decoder {
public:
    using RawWord = IcuRawImemWord<WordBits>;
    using DecodedContext = DecodedIcuMacroContext<FuncInstruction>;

    static_assert(FetchLatency > 0);
    static_assert(ReservoirWords >= 3);
    static_assert(DecodeWindowBits > 0 && DecodeWindowBits <= WordBits);
    static_assert(DecodeWindowBits <= 64);
    static_assert(DdbDepth > 0);
    static_assert(DdbRunCapacity > 0);

    static constexpr std::size_t word_bits = WordBits;
    static constexpr std::size_t reservoir_words = ReservoirWords;
    static constexpr std::size_t decode_window_bits = DecodeWindowBits;
    static constexpr std::size_t ddb_depth = DdbDepth;
    static constexpr std::size_t admission_lookahead = AdmissionLookahead;
    static constexpr std::size_t context_expand_width = 1;

    IcuMacroV1Decoder(IcuMacroQueueKind kind, std::vector<RawWord> image)
        : kind_(kind), image_(std::move(image))
    {
        validate_kind();
        if (image_.empty())
            throw StaticScheduleError(
                "Macro raw i-MEM image has no control word");
    }

    // Compatibility path for decoder unit tests: an unbounded current cycle
    // disables the timing gate while retaining one-context-per-cycle output.
    std::optional<DecodedContext> tick(bool context_available)
    {
        return tick_impl(std::numeric_limits<std::size_t>::max(),
            context_available ? 0 : 1, 1, std::nullopt);
    }

    // Runtime path. active_occupancy/active_capacity are the credit interface
    // to the finite Active Context RAM; no scheduler implementation leaks into
    // this frontend boundary.
    std::optional<DecodedContext> tick(std::size_t current_cycle,
        std::size_t active_occupancy, std::size_t active_capacity)
    {
        return tick(current_cycle, active_occupancy, active_capacity,
            std::nullopt);
    }

    std::optional<DecodedContext> tick(std::size_t current_cycle,
        std::size_t active_occupancy, std::size_t active_capacity,
        std::optional<std::size_t> earliest_active_release_cycle)
    {
        if (active_capacity == 0 || active_occupancy > active_capacity)
            throw std::invalid_argument("invalid Active Context RAM credit");
        return tick_impl(
            current_cycle, active_occupancy, active_capacity,
            earliest_active_release_cycle);
    }

    bool done() const noexcept
    {
        return disabled_ || (state_ == State::Done
            && !commit_buffer_ && ddb_.empty());
    }

    // Priming stops only at an actual decode-ahead boundary: the parser has
    // consumed the image or the finite DDB is full.
    bool primed() const noexcept
    {
        return control_latched_
            && (disabled_
                || (state_ == State::Done && !commit_buffer_)
                || ddb_.size() == DdbDepth);
    }

    bool control_latched() const noexcept { return control_latched_; }
    bool enabled() const noexcept { return control_latched_ && !disabled_; }
    std::uint32_t command_count() const noexcept { return command_count_; }
    std::size_t decoded_count() const noexcept { return decoded_count_; }
    std::size_t ddb_occupancy() const noexcept { return ddb_.size(); }
    std::size_t pending_decoded_contexts() const noexcept
    {
        std::size_t result = 0;
        for (const auto& descriptor : ddb_)
            result += descriptor.run_count
                - descriptor.next_context_index;
        if (commit_buffer_)
            result += commit_buffer_->run_count
                - commit_buffer_->next_context_index;
        return result;
    }
    std::size_t reservoir_occupancy_bits() const noexcept
    {
        return reservoir_.available_bits();
    }

    void record_context_completion(
        std::size_t admitted_cycle, std::size_t completed_cycle)
    {
        if (completed_cycle < admitted_cycle)
            throw std::logic_error(
                "Macro context completed before it was admitted");
        const auto lifetime = completed_cycle - admitted_cycle + 1;
        statistics_.context_active_lifetime_cycles += lifetime;
        statistics_.max_context_active_lifetime_cycles = std::max(
            statistics_.max_context_active_lifetime_cycles, lifetime);
    }

    const IcuMacroDecoderStatistics& statistics() const noexcept
    {
        return statistics_;
    }

private:
    static constexpr unsigned kCompactStartDeltaBits = 22;
    static constexpr unsigned kCompactOperandDeltaBits = 14;
    static constexpr std::size_t kExtendedTemplatePayloadBits = 6 * 32 + 2;
    static constexpr std::size_t kTemplatePayloadWords =
        (kExtendedTemplatePayloadBits + 63) / 64;
    static constexpr unsigned kMemAddressBits = static_cast<unsigned>(
        std::bit_width(hw::kSramDepthRows - 1));

    enum class State {
        WaitingForControl,
        DictionaryCount,
        DictionaryEntry,
        InitialState,
        RunPrefix,
        RunLength,
        TemplatePrefix,
        TemplatePayload,
        TemplateDecode,
        Record,
        Delta,
        Done,
    };

    enum class Shape : std::uint8_t {
        Single = 0,
        Inner1D = 1,
        Outer1D = 2,
        Full2D = 3,
    };

    struct Delta {
        std::uint32_t start_cycle{0};
        std::int32_t operand{0};
    };

    struct Template {
        IcuMacroSchedule schedule{};
        FuncInstruction instruction{};
    };

    struct TemplatePlan {
        bool extended{false};
        std::uint64_t native{0};
        Shape shape{Shape::Single};
        std::uint8_t target{0};
        bool mem_write{false};
        std::uint8_t mem_stream{0};
        bool mem_preserve{false};
        std::size_t payload_bits{0};
    };

    struct DecodedDescriptor {
        std::size_t first_record_index{0};
        Template descriptor_template{};
        std::uint64_t next_start_cycle{0};
        std::int64_t next_operand{0};
        std::array<Delta, DdbRunCapacity - 1> deltas{};
        std::size_t run_count{0};
        std::size_t next_context_index{0};
        std::size_t decode_start_cycle{0};
        std::size_t commit_cycle{0};
        bool completes_run{false};
    };

    class Reservoir {
    public:
        std::size_t available_bits() const noexcept { return valid_bits_; }
        std::size_t valid_words() const noexcept { return words_.size(); }
        std::size_t head_bit_offset() const noexcept
        {
            return head_bit_offset_;
        }

        bool can_refill(std::size_t reserved_words = 0) const noexcept
        {
            return words_.size() + reserved_words < ReservoirWords;
        }

        void refill(RawWord word)
        {
            if (!can_refill())
                throw std::logic_error("Macro reservoir overflow");
            words_.push_back(std::move(word));
            valid_bits_ += WordBits;
        }

        std::optional<std::uint64_t> peek(
            std::size_t width, std::size_t offset = 0) const
        {
            if (width > DecodeWindowBits || width > 64)
                throw std::invalid_argument(
                    "Macro decode request exceeds the aligned window");
            if (offset + width > valid_bits_) return std::nullopt;
            std::uint64_t result = 0;
            for (std::size_t bit = 0; bit < width; ++bit) {
                const auto absolute = head_bit_offset_ + offset + bit;
                const auto word = absolute / WordBits;
                const auto word_bit = absolute % WordBits;
                if (words_[word].bit(word_bit))
                    result |= std::uint64_t{1} << bit;
            }
            return result;
        }

        void consume(std::size_t width)
        {
            if (width > valid_bits_)
                throw std::logic_error(
                    "Macro decoder consumed beyond its reservoir");
            valid_bits_ -= width;
            head_bit_offset_ += width;
            while (head_bit_offset_ >= WordBits) {
                head_bit_offset_ -= WordBits;
                words_.pop_front();
            }
            if (words_.empty()) head_bit_offset_ = 0;
        }

    private:
        std::deque<RawWord> words_{};
        std::size_t head_bit_offset_{0};
        std::size_t valid_bits_{0};
    };

    class Cursor {
    public:
        explicit Cursor(const std::deque<std::uint8_t>& bits) : bits_(bits) {}

        std::optional<std::uint64_t> read(unsigned width)
        {
            if (width > 64)
                throw std::invalid_argument(
                    "Macro decoder field is wider than 64 bits");
            if (position_ + width > bits_.size()) return std::nullopt;
            std::uint64_t result = 0;
            for (unsigned bit = 0; bit < width; ++bit)
                if (bits_[position_ + bit] != 0)
                    result |= std::uint64_t{1} << bit;
            position_ += width;
            return result;
        }

        std::size_t consumed() const noexcept { return position_; }

    private:
        const std::deque<std::uint8_t>& bits_;
        std::size_t position_{0};
    };

    class TemplatePayloadCursor {
    public:
        TemplatePayloadCursor(
            const std::array<std::uint64_t,
                kTemplatePayloadWords>& bits,
            std::size_t size)
            : bits_(bits), size_(size)
        {
        }

        std::optional<std::uint64_t> read(unsigned width)
        {
            if (width > 64)
                throw std::invalid_argument(
                    "Macro decoder field is wider than 64 bits");
            if (position_ + width > size_) return std::nullopt;
            std::uint64_t result = 0;
            for (unsigned bit = 0; bit < width; ++bit)
                if (((bits_[(position_ + bit) / 64]
                          >> ((position_ + bit) % 64))
                        & std::uint64_t{1})
                    != 0)
                    result |= std::uint64_t{1} << bit;
            position_ += width;
            return result;
        }

        std::size_t consumed() const noexcept { return position_; }

    private:
        const std::array<std::uint64_t,
            kTemplatePayloadWords>& bits_;
        std::size_t size_{0};
        std::size_t position_{0};
    };

    struct PendingFetch {
        std::size_t address{0};
        std::size_t remaining_cycles{FetchLatency};
    };

    static std::int64_t signed_value(std::uint64_t raw, unsigned width)
    {
        if (width == 0 || width > 64)
            throw std::invalid_argument("invalid signed Macro field width");
        if (width == 64) return static_cast<std::int64_t>(raw);
        const auto sign = std::uint64_t{1} << (width - 1);
        if ((raw & sign) == 0) return static_cast<std::int64_t>(raw);
        return static_cast<std::int64_t>(
            raw | (~std::uint64_t{0} << width));
    }

    template <typename BitCursor>
    static bool read(
        BitCursor& cursor, unsigned width, std::uint64_t& value)
    {
        const auto result = cursor.read(width);
        if (!result.has_value()) return false;
        value = *result;
        return true;
    }

    void validate_kind() const
    {
        if constexpr (std::is_same_v<FuncInstruction, MemInstruction>) {
            if (kind_ != IcuMacroQueueKind::Mem)
                throw std::invalid_argument(
                    "MEM Macro decoder has a non-MEM queue kind");
        } else if constexpr (std::is_same_v<FuncInstruction,
                                 MxmControlInstruction>) {
            if (kind_ != IcuMacroQueueKind::MxmLoad
                && kind_ != IcuMacroQueueKind::MxmCompute)
                throw std::invalid_argument(
                    "MXM control Macro decoder has an invalid queue kind");
        } else if constexpr (std::is_same_v<FuncInstruction,
                                 MxmDequantInstruction>) {
            if (kind_ != IcuMacroQueueKind::MxmDequant)
                throw std::invalid_argument(
                    "MXM dequant Macro decoder has an invalid queue kind");
        } else {
            throw std::invalid_argument(
                "Macro v1 is unsupported for this ICU instruction type");
        }
    }

    unsigned operand_bits() const
    {
        switch (kind_) {
        case IcuMacroQueueKind::Mem: return kMemAddressBits;
        case IcuMacroQueueKind::MxmLoad: return 2;
        case IcuMacroQueueKind::MxmCompute: return 13;
        case IcuMacroQueueKind::MxmDequant: return 0;
        }
        throw std::logic_error("unknown Macro queue kind");
    }

    unsigned native_instruction_bits() const
    {
        switch (kind_) {
        case IcuMacroQueueKind::Mem: return 32;
        case IcuMacroQueueKind::MxmLoad:
        case IcuMacroQueueKind::MxmDequant: return 16;
        case IcuMacroQueueKind::MxmCompute: return 49;
        }
        throw std::logic_error("unknown Macro queue kind");
    }

    std::optional<DecodedContext> tick_impl(std::size_t current_cycle,
        std::size_t active_occupancy, std::size_t active_capacity,
        std::optional<std::size_t> earliest_active_release_cycle)
    {
        ++statistics_.cycles;
        waiting_for_bits_ = false;
        commit_ready_fetch();

        auto admitted = expand_and_admit(
            current_cycle, active_occupancy, active_capacity,
            earliest_active_release_cycle);
        // The DDB may pop above and accept the buffered descriptor in the same
        // cycle. The parser then sees the released commit credit immediately.
        drain_commit_buffer();
        parser_step();
        begin_fetch_if_possible();
        age_pending_fetch();

        if (waiting_for_bits_) {
            ++statistics_.bit_wait_cycles;
            ++statistics_.decoder_starvation_cycles;
            if (!pending_fetch_.has_value()
                && next_fetch_address_ == image_.size())
                throw StaticScheduleError(
                    "truncated Macro v1 raw i-MEM image");
        }
        if (reservoir_.available_bits() == 0)
            ++statistics_.reservoir_empty_cycles;
        statistics_.reservoir_occupancy_bit_samples +=
            reservoir_.available_bits();
        if (!reservoir_.can_refill(pending_fetch_.has_value() ? 1 : 0))
            ++statistics_.reservoir_full_cycles;
        if (ddb_.size() == DdbDepth) ++statistics_.ddb_full_cycles;
        statistics_.ddb_occupancy_samples += ddb_.size();
        statistics_.peak_ddb_occupancy = std::max(
            statistics_.peak_ddb_occupancy, ddb_.size());
        const auto visible_occupancy = active_occupancy
            + (admitted.has_value() ? 1 : 0);
        statistics_.active_occupancy_samples += visible_occupancy;
        statistics_.peak_active_occupancy = std::max(
            statistics_.peak_active_occupancy, visible_occupancy);
        return admitted;
    }

    void commit_ready_fetch()
    {
        if (!pending_fetch_.has_value()
            || pending_fetch_->remaining_cycles != 0)
            return;
        const auto address = pending_fetch_->address;
        const auto& word = image_[address];
        pending_fetch_.reset();
        ++statistics_.fetched_words;

        if (address == 0) {
            const std::uint32_t low = word.lanes[0];
            if ((low & 0xf0000000u) != 0)
                throw StaticScheduleError(
                    "Macro control word has non-zero reserved bits");
            for (std::size_t lane = 1; lane < word.lanes.size(); ++lane)
                if (word.lanes[lane] != 0)
                    throw StaticScheduleError(
                        "Macro control word has non-zero reserved bits");
            const bool valid = (low & 1u) != 0;
            const bool enable = (low & 2u) != 0;
            const auto mode =
                static_cast<IcuQueueMode>((low >> 2) & 3u);
            if (!valid)
                throw StaticScheduleError(
                    "Macro raw i-MEM queue is not valid");
            if (mode != IcuQueueMode::Macro)
                throw StaticScheduleError(
                    "Macro raw i-MEM queue has a non-Macro QueueMode");
            command_count_ = (low >> 4) & 0x00ffffffu;
            control_latched_ = true;
            disabled_ = !enable;
            state_ = disabled_ || command_count_ == 0
                ? State::Done : State::DictionaryCount;
            return;
        }

        reservoir_.refill(word);
        statistics_.peak_reservoir_bits = std::max(
            statistics_.peak_reservoir_bits,
            reservoir_.available_bits());
    }

    void begin_fetch_if_possible()
    {
        if (pending_fetch_.has_value() || disabled_
            || state_ == State::Done
            || next_fetch_address_ == image_.size())
            return;
        if (next_fetch_address_ != 0 && !reservoir_.can_refill())
            return;
        pending_fetch_ = PendingFetch{next_fetch_address_, FetchLatency};
        ++next_fetch_address_;
    }

    void age_pending_fetch()
    {
        if (pending_fetch_.has_value()
            && pending_fetch_->remaining_cycles != 0)
            --pending_fetch_->remaining_cycles;
    }

    std::optional<std::uint64_t> peek_bits(
        std::size_t width, std::size_t offset = 0)
    {
        const auto result = reservoir_.peek(width, offset);
        if (!result.has_value()) waiting_for_bits_ = true;
        return result;
    }

    void consume_bits(std::size_t width)
    {
        reservoir_.consume(width);
        statistics_.payload_bits_consumed += width;
    }

    bool assemble_bits(std::size_t total_width)
    {
        if (assembly_target_bits_ == 0)
            assembly_target_bits_ = total_width;
        if (assembly_target_bits_ != total_width)
            throw std::logic_error("Macro parser assembly target changed");
        const auto remaining = total_width - assembly_bits_.size();
        const auto width = std::min(remaining, DecodeWindowBits);
        const auto value = peek_bits(width);
        if (!value.has_value()) return false;
        for (std::size_t bit = 0; bit < width; ++bit)
            assembly_bits_.push_back(
                ((*value >> bit) & std::uint64_t{1}) != 0 ? 1u : 0u);
        consume_bits(width);
        return assembly_bits_.size() == total_width;
    }

    void clear_assembly()
    {
        assembly_bits_.clear();
        assembly_target_bits_ = 0;
    }

    void parser_step()
    {
        if (state_ == State::WaitingForControl || state_ == State::Done)
            return;
        const bool next_record_completes_descriptor =
            (state_ == State::Record || state_ == State::Delta)
            && (building_descriptor_.run_count + 1 == DdbRunCapacity
                || run_remaining_ == 1);
        if (next_record_completes_descriptor
            && commit_buffer_) {
            ++statistics_.decoder_ddb_stall_cycles;
            return;
        }
        ++statistics_.decoder_active_cycles;

        switch (state_) {
        case State::WaitingForControl:
        case State::Done: return;
        case State::DictionaryCount: read_dictionary_count(); return;
        case State::DictionaryEntry: read_dictionary_entry(); return;
        case State::InitialState: read_initial_state(); return;
        case State::RunPrefix: read_run_prefix(); return;
        case State::RunLength: read_run_length(); return;
        case State::TemplatePrefix: read_template_prefix(); return;
        case State::TemplatePayload: read_template_payload(); return;
        case State::TemplateDecode: decode_template(); return;
        case State::Record: read_record_without_delta(); return;
        case State::Delta: read_delta_step(); return;
        }
    }

    void read_dictionary_count()
    {
        const auto count = peek_bits(3);
        if (!count.has_value()) return;
        if (*count > dictionary_.size())
            throw StaticScheduleError(
                "Macro dictionary count exceeds seven entries");
        dictionary_count_ = static_cast<std::size_t>(*count);
        consume_bits(3);
        state_ = dictionary_count_ == 0
            ? State::InitialState : State::DictionaryEntry;
    }

    void read_dictionary_entry()
    {
        constexpr std::size_t width =
            kCompactStartDeltaBits + kCompactOperandDeltaBits;
        if (!assemble_bits(width)) return;
        Cursor cursor(assembly_bits_);
        std::uint64_t start = 0;
        std::uint64_t operand = 0;
        if (!read(cursor, kCompactStartDeltaBits, start)
            || !read(cursor, kCompactOperandDeltaBits, operand))
            throw std::logic_error(
                "Macro dictionary assembler produced an incomplete field");
        dictionary_[dictionary_index_++] = Delta{
            static_cast<std::uint32_t>(start),
            static_cast<std::int32_t>(signed_value(
                operand, kCompactOperandDeltaBits))};
        clear_assembly();
        if (dictionary_index_ == dictionary_count_)
            state_ = State::InitialState;
    }

    void read_initial_state()
    {
        const auto width = 32u + operand_bits();
        if (!assemble_bits(width)) return;
        Cursor cursor(assembly_bits_);
        std::uint64_t start = 0;
        std::uint64_t operand = 0;
        if (!read(cursor, 32, start)
            || (operand_bits() != 0
                && !read(cursor, operand_bits(), operand)))
            throw std::logic_error(
                "Macro initial-state assembler produced an incomplete field");
        current_start_cycle_ = start;
        current_operand_ = static_cast<std::int64_t>(operand);
        clear_assembly();
        state_ = State::RunPrefix;
    }

    void read_run_prefix()
    {
        const auto long_run = peek_bits(1);
        if (!long_run.has_value()) return;
        consume_bits(1);
        ++statistics_.descriptor_count;
        if constexpr (std::is_same_v<FuncInstruction, MemInstruction>)
            ++statistics_.mem_descriptor_count;
        else
            ++statistics_.mxm_descriptor_count;
        run_decode_start_cycle_ = statistics_.cycles;
        if (*long_run == 0) {
            run_remaining_ = 1;
            state_ = State::TemplatePrefix;
        } else {
            state_ = State::RunLength;
        }
    }

    void read_run_length()
    {
        const auto encoded = peek_bits(16);
        if (!encoded.has_value()) return;
        run_remaining_ = static_cast<std::size_t>(*encoded + 1);
        if (run_remaining_ > command_count_ - decoded_count_)
            throw StaticScheduleError(
                "Macro run exceeds declared command count");
        consume_bits(16);
        state_ = State::TemplatePrefix;
    }

    std::size_t compact_schedule_bits(Shape shape) const noexcept
    {
        std::size_t result = 0;
        if (shape == Shape::Inner1D || shape == Shape::Full2D)
            result += 29;
        if (shape == Shape::Outer1D || shape == Shape::Full2D)
            result += 36;
        return result;
    }

    static std::uint64_t bit_field(
        std::uint64_t value, unsigned offset, unsigned width)
    {
        if (width == 0 || width > 64 || offset + width > 64)
            throw std::invalid_argument("invalid Macro bit field");
        const auto mask = width == 64
            ? ~std::uint64_t{0}
            : (std::uint64_t{1} << width) - 1;
        return (value >> offset) & mask;
    }

    void read_template_prefix()
    {
        const auto extended = peek_bits(1);
        if (!extended.has_value()) return;

        TemplatePlan plan;
        plan.extended = *extended != 0;
        std::size_t header_bits = 0;

        if constexpr (std::is_same_v<FuncInstruction, MemInstruction>) {
            header_bits = plan.extended ? 1 + 32 : 1 + 2 + 1 + 6 + 1;
            const auto header = peek_bits(header_bits);
            if (!header.has_value()) return;
            if (plan.extended) {
                plan.native = bit_field(*header, 1, 32);
                plan.payload_bits = kExtendedTemplatePayloadBits;
            } else {
                plan.shape = static_cast<Shape>(bit_field(*header, 1, 2));
                plan.mem_write = bit_field(*header, 3, 1) != 0;
                plan.mem_stream = static_cast<std::uint8_t>(
                    bit_field(*header, 4, 6));
                plan.mem_preserve = bit_field(*header, 10, 1) != 0;
                plan.payload_bits = compact_schedule_bits(plan.shape);
            }
        } else {
            const auto native_bits = native_instruction_bits();
            header_bits = 1 + native_bits + (plan.extended ? 0 : 4);
            const auto header = peek_bits(header_bits);
            if (!header.has_value()) return;
            plan.native = bit_field(*header, 1, native_bits);
            if (plan.extended) {
                plan.payload_bits = kExtendedTemplatePayloadBits;
            } else {
                plan.shape = static_cast<Shape>(
                    bit_field(*header, 1 + native_bits, 2));
                plan.target = static_cast<std::uint8_t>(
                    bit_field(*header, 1 + native_bits + 2, 2));
                plan.payload_bits = compact_schedule_bits(plan.shape);
            }
        }

        template_plan_ = plan;
        template_payload_count_ = 0;
        consume_bits(header_bits);
        state_ = plan.payload_bits == 0
            ? State::TemplateDecode : State::TemplatePayload;
    }

    void read_template_payload()
    {
        if (template_payload_count_ >= template_plan_.payload_bits)
            throw std::logic_error("Macro template payload is already complete");
        const auto remaining =
            template_plan_.payload_bits - template_payload_count_;
        const auto width = std::min(remaining, DecodeWindowBits);
        const auto value = peek_bits(width);
        if (!value.has_value()) return;
        for (std::size_t bit = 0; bit < width; ++bit) {
            const auto destination = template_payload_count_ + bit;
            const auto mask = std::uint64_t{1} << (destination % 64);
            auto& word = template_payload_words_[destination / 64];
            if (((*value >> bit) & std::uint64_t{1}) != 0)
                word |= mask;
            else
                word &= ~mask;
        }
        consume_bits(width);
        template_payload_count_ += width;
        if (template_payload_count_ == template_plan_.payload_bits)
            state_ = State::TemplateDecode;
    }

    void decode_template()
    {
        TemplatePayloadCursor cursor(
            template_payload_words_, template_plan_.payload_bits);
        Template result;
        bool decoded = false;

        if constexpr (std::is_same_v<FuncInstruction, MemInstruction>) {
            if (template_plan_.extended) {
                result.instruction =
                    isa::decode_mem_instruction(template_plan_.native);
                decoded = read_extended_schedule(cursor, result.schedule);
            } else {
                result.instruction = template_plan_.mem_write
                    ? (template_plan_.mem_preserve
                        ? MemInstruction::WriteTap(
                            0, template_plan_.mem_stream)
                        : MemInstruction::Write(
                            0, template_plan_.mem_stream))
                    : MemInstruction::Read(0, template_plan_.mem_stream);
                result.schedule = IcuMacroSchedule{
                    0, 1, 1, 0, 1, 1, 0,
                    IcuInductionTarget::MemAddress};
                decoded = read_compact_schedule(
                    cursor, template_plan_.shape, result.schedule);
            }
        } else {
            result.instruction = decode_native(template_plan_.native);
            if (template_plan_.extended) {
                decoded = read_extended_schedule(cursor, result.schedule);
            } else {
                result.schedule = IcuMacroSchedule{
                    0, 1, 1, 0, 1, 1, 0,
                    static_cast<IcuInductionTarget>(template_plan_.target)};
                decoded = read_compact_schedule(
                    cursor, template_plan_.shape, result.schedule);
            }
        }

        if (!decoded
            || cursor.consumed() != template_plan_.payload_bits)
            throw std::logic_error(
                "Macro template payload decoder consumed the wrong width");
        current_template_ = std::move(result);
        begin_descriptor_chunk();
        state_ = decoded_count_ == 0 ? State::Record : State::Delta;
    }

    template <typename BitCursor>
    static bool read_extended_schedule(
        BitCursor& cursor, IcuMacroSchedule& schedule)
    {
        std::uint64_t inner_count = 0, inner_interval = 0;
        std::uint64_t inner_stride = 0, outer_count = 0;
        std::uint64_t outer_interval = 0, outer_stride = 0, target = 0;
        if (!read(cursor, 32, inner_count)
            || !read(cursor, 32, inner_interval)
            || !read(cursor, 32, inner_stride)
            || !read(cursor, 32, outer_count)
            || !read(cursor, 32, outer_interval)
            || !read(cursor, 32, outer_stride)
            || !read(cursor, 2, target))
            return false;
        schedule = IcuMacroSchedule{
            0,
            static_cast<std::size_t>(inner_count),
            static_cast<std::size_t>(inner_interval),
            static_cast<std::int32_t>(inner_stride),
            static_cast<std::size_t>(outer_count),
            static_cast<std::size_t>(outer_interval),
            static_cast<std::int32_t>(outer_stride),
            static_cast<IcuInductionTarget>(target)};
        return true;
    }

    template <typename BitCursor>
    static bool read_compact_schedule(
        BitCursor& cursor, Shape shape, IcuMacroSchedule& schedule)
    {
        if (shape == Shape::Inner1D || shape == Shape::Full2D) {
            std::uint64_t count = 0, interval = 0, stride = 0;
            if (!read(cursor, 11, count)
                || !read(cursor, 8, interval)
                || !read(cursor, 10, stride))
                return false;
            schedule.inner_count = static_cast<std::size_t>(count + 1);
            schedule.inner_interval = static_cast<std::size_t>(interval + 1);
            schedule.inner_stride = signed_value(stride, 10);
        }
        if (shape == Shape::Outer1D || shape == Shape::Full2D) {
            std::uint64_t count = 0, residual = 0, stride = 0;
            if (!read(cursor, 9, count)
                || !read(cursor, 15, residual)
                || !read(cursor, 12, stride))
                return false;
            schedule.outer_count = static_cast<std::size_t>(count + 1);
            schedule.outer_interval = static_cast<std::size_t>(residual + 1)
                + (schedule.inner_count - 1) * schedule.inner_interval;
            schedule.outer_stride = signed_value(stride, 12);
        }
        return true;
    }

    FuncInstruction decode_native(std::uint64_t native) const
    {
        if constexpr (std::is_same_v<FuncInstruction,
                          MxmControlInstruction>) {
            return isa::decode_mxm_instruction(native);
        } else if constexpr (std::is_same_v<FuncInstruction,
                                 MxmDequantInstruction>) {
            return isa::decode_mxm_dequant_instruction(
                static_cast<std::uint16_t>(native));
        } else {
            throw std::logic_error(
                "Macro v1 native decode is unsupported for this instruction type");
        }
    }

    std::optional<std::pair<std::size_t, std::size_t>> delta_layout()
    {
        std::size_t prefix = 0;
        std::size_t symbol = 0;
        const auto bit0 = peek_bits(1);
        if (!bit0.has_value()) return std::nullopt;
        if (*bit0 == 0) {
            prefix = 1;
            symbol = 0;
        } else {
            const auto bit1 = peek_bits(1, 1);
            if (!bit1.has_value()) return std::nullopt;
            if (*bit1 == 0) {
                prefix = 2;
                symbol = 1;
            } else {
                const auto bit2 = peek_bits(1, 2);
                if (!bit2.has_value()) return std::nullopt;
                if (*bit2 == 0) {
                    const auto select = peek_bits(1, 3);
                    if (!select.has_value()) return std::nullopt;
                    prefix = 4;
                    symbol = *select != 0 ? 3 : 2;
                } else {
                    const auto bit3 = peek_bits(1, 3);
                    const auto select = peek_bits(1, 4);
                    if (!bit3.has_value() || !select.has_value())
                        return std::nullopt;
                    prefix = 5;
                    symbol = *bit3 == 0
                        ? (*select != 0 ? 5 : 4)
                        : (*select != 0 ? 7 : 6);
                }
            }
        }

        if (symbol < dictionary_count_) return {{prefix, symbol}};
        if (symbol != 7)
            throw StaticScheduleError(
                "Macro delta references a missing dictionary entry");
        const auto wide = peek_bits(1, prefix);
        if (!wide.has_value()) return std::nullopt;
        return {{prefix + 1 + (*wide != 0 ? 64 : 36), symbol}};
    }

    Delta decode_delta_assembly(std::size_t symbol) const
    {
        if (symbol < dictionary_count_) return dictionary_[symbol];
        Cursor cursor(assembly_bits_);
        std::uint64_t bit = 0;
        // Consume the prefix using the same prefix tree as delta_layout().
        if (!read(cursor, 1, bit))
            throw std::logic_error("empty Macro delta assembly");
        if (bit != 0) {
            if (!read(cursor, 1, bit)) throw std::logic_error("bad delta");
            if (bit != 0) {
                if (!read(cursor, 1, bit))
                    throw std::logic_error("bad delta");
                if (bit == 0) {
                    if (!read(cursor, 1, bit))
                        throw std::logic_error("bad delta");
                } else {
                    if (!read(cursor, 1, bit)
                        || !read(cursor, 1, bit))
                        throw std::logic_error("bad delta");
                }
            }
        }
        std::uint64_t wide = 0, start = 0, operand = 0;
        if (!read(cursor, 1, wide)) throw std::logic_error("bad delta");
        if (wide == 0) {
            if (!read(cursor, kCompactStartDeltaBits, start)
                || !read(cursor, kCompactOperandDeltaBits, operand))
                throw std::logic_error("bad compact delta");
            return Delta{static_cast<std::uint32_t>(start),
                static_cast<std::int32_t>(signed_value(
                    operand, kCompactOperandDeltaBits))};
        }
        if (!read(cursor, 32, start) || !read(cursor, 32, operand))
            throw std::logic_error("bad wide delta");
        return Delta{static_cast<std::uint32_t>(start),
            static_cast<std::int32_t>(operand)};
    }

    void read_record_without_delta()
    {
        finish_record(Delta{}, false);
    }

    void read_delta_step()
    {
        if (assembly_target_bits_ == 0) {
            const auto layout = delta_layout();
            if (!layout.has_value()) return;
            assembly_target_bits_ = layout->first;
            delta_symbol_ = layout->second;
        }
        if (!assemble_bits(assembly_target_bits_)) return;
        const auto delta = delta_symbol_ < dictionary_count_
            ? dictionary_[delta_symbol_]
            : decode_delta_assembly(delta_symbol_);
        clear_assembly();
        finish_record(delta, true);
    }

    void begin_descriptor_chunk()
    {
        building_descriptor_ = DecodedDescriptor{};
        building_descriptor_.first_record_index = decoded_count_;
        building_descriptor_.descriptor_template = current_template_;
        building_descriptor_.decode_start_cycle =
            run_decode_start_cycle_;
    }

    void finish_record(const Delta& delta, bool has_delta)
    {
        if (has_delta) {
            current_start_cycle_ += delta.start_cycle;
            current_operand_ += delta.operand;
        }
        validate_current_state();

        auto& descriptor = building_descriptor_;
        if (descriptor.run_count == 0) {
            descriptor.next_start_cycle = current_start_cycle_;
            descriptor.next_operand = current_operand_;
        } else {
            descriptor.deltas[descriptor.run_count - 1] = delta;
        }
        ++descriptor.run_count;
        ++decoded_count_;
        --run_remaining_;

        if (descriptor.run_count == DdbRunCapacity
            || run_remaining_ == 0) {
            if (commit_buffer_)
                throw std::logic_error(
                    "Macro parser completed a descriptor without commit credit");
            descriptor.completes_run = run_remaining_ == 0;
            commit_buffer_ = std::make_unique<DecodedDescriptor>(
                std::move(building_descriptor_));
            if (run_remaining_ != 0) {
                begin_descriptor_chunk();
                state_ = State::Delta;
            } else if (decoded_count_ == command_count_) {
                state_ = State::Done;
            } else {
                state_ = State::RunPrefix;
            }
            return;
        }
        state_ = State::Delta;
    }

    void validate_current_state() const
    {
        if (current_start_cycle_ > std::numeric_limits<std::uint32_t>::max())
            throw StaticScheduleError(
                "decoded Macro start cycle is out of range");
        const auto width = operand_bits();
        if (current_operand_ < 0
            || (width != 0
                && static_cast<std::uint64_t>(current_operand_)
                    >= (std::uint64_t{1} << width)))
            throw StaticScheduleError(
                "decoded Macro operand is out of range");
    }

    void drain_commit_buffer()
    {
        if (!commit_buffer_ || ddb_.size() == DdbDepth)
            return;
        auto descriptor = std::move(*commit_buffer_);
        commit_buffer_.reset();
        descriptor.commit_cycle = statistics_.cycles;
        if (descriptor.completes_run) {
            const auto latency = statistics_.cycles
                - descriptor.decode_start_cycle + 1;
            statistics_.descriptor_decode_cycles += latency;
            statistics_.max_descriptor_decode_cycles = std::max(
                statistics_.max_descriptor_decode_cycles, latency);
        }
        ddb_.push_back(std::move(descriptor));
        ++statistics_.ddb_entries_committed;
    }

    std::optional<DecodedContext> expand_and_admit(
        std::size_t current_cycle, std::size_t active_occupancy,
        std::size_t active_capacity,
        std::optional<std::size_t> earliest_active_release_cycle)
    {
        if (ddb_.empty()) return std::nullopt;
        auto& descriptor = ddb_.front();
        const bool ignore_timing =
            current_cycle == std::numeric_limits<std::size_t>::max();
        const auto start_cycle = static_cast<std::size_t>(
            descriptor.next_start_cycle);
        const auto admit_cycle = start_cycle > AdmissionLookahead
            ? start_cycle - AdmissionLookahead : 0;
        if (!ignore_timing && current_cycle < admit_cycle) {
            ++statistics_.timing_window_block_cycles;
            return std::nullopt;
        }
        if (active_occupancy >= active_capacity) {
            if (!ignore_timing
                && earliest_active_release_cycle.has_value()) {
                // Admission writes after issue and becomes visible on the next
                // cycle, so the last safe admission cycle is start_cycle - 1.
                const auto latest_admit_cycle = start_cycle == 0
                    ? 0 : start_cycle - 1;
                if (*earliest_active_release_cycle > latest_admit_cycle) {
                    throw StaticScheduleError(
                        "Active Context RAM has no release before the Macro admission deadline");
                }
                ++statistics_.capacity_feasible_wait_cycles;
            }
            ++statistics_.active_ram_full_cycles;
            ++statistics_.admission_stall_cycles;
            ++statistics_.context_stall_cycles;
            return std::nullopt;
        }

        auto result = DecodedContext{
            descriptor.first_record_index
                + descriptor.next_context_index,
            descriptor.descriptor_template.schedule,
            descriptor.descriptor_template.instruction,
            ignore_timing ? 0 : current_cycle};
        result.schedule.start_cycle = start_cycle;
        set_operand(result.instruction,
            static_cast<std::size_t>(descriptor.next_operand));
        validate_context(result);

        ++statistics_.decoded_contexts;
        ++statistics_.contexts_generated;
        ++statistics_.admission_count;
        if (!ignore_timing && start_cycle >= current_cycle) {
            const auto latency = start_cycle - current_cycle;
            statistics_.admission_to_start_cycles += latency;
            statistics_.max_admission_to_start_cycles = std::max(
                statistics_.max_admission_to_start_cycles, latency);
        }

        ++descriptor.next_context_index;
        if (descriptor.next_context_index == descriptor.run_count) {
            const auto residency = statistics_.cycles
                - descriptor.commit_cycle + 1;
            statistics_.ddb_residency_cycles += residency;
            statistics_.max_ddb_residency_cycles = std::max(
                statistics_.max_ddb_residency_cycles, residency);
            ddb_.pop_front();
        } else {
            const auto& delta =
                descriptor.deltas[descriptor.next_context_index - 1];
            descriptor.next_start_cycle += delta.start_cycle;
            descriptor.next_operand += delta.operand;
        }
        return result;
    }

    void set_operand(FuncInstruction& instruction, std::size_t operand) const
    {
        if constexpr (std::is_same_v<FuncInstruction, MemInstruction>) {
            instruction.address = operand;
        } else if constexpr (std::is_same_v<FuncInstruction,
                                 MxmControlInstruction>) {
            if (kind_ == IcuMacroQueueKind::MxmLoad)
                instruction.weight_column = operand;
            else
                instruction.accumulator_address = operand;
        } else {
            if (operand != 0)
                throw StaticScheduleError(
                    "MXM dequant Macro has an operand");
        }
    }

    void validate_context(const DecodedContext& context) const
    {
        const auto& schedule = context.schedule;
        if (schedule.inner_count == 0 || schedule.outer_count == 0
            || schedule.inner_interval == 0
            || schedule.outer_interval == 0
            || (schedule.outer_count > 1
                && schedule.outer_interval
                    <= (schedule.inner_count - 1)
                        * schedule.inner_interval))
            throw StaticScheduleError(
                "decoded Macro has an invalid iteration space");

        if constexpr (std::is_same_v<FuncInstruction, MemInstruction>) {
            if (schedule.induction_target
                != IcuInductionTarget::MemAddress)
                throw StaticScheduleError(
                    "decoded MEM Macro has an invalid induction target");
        } else if constexpr (std::is_same_v<FuncInstruction,
                                 MxmControlInstruction>) {
            if (kind_ == IcuMacroQueueKind::MxmLoad) {
                if (context.instruction.opcode != MxmControlOpcode::IW
                    || (schedule.induction_target
                            != IcuInductionTarget::None
                        && schedule.induction_target
                            != IcuInductionTarget::MxmWeightColumn))
                    throw StaticScheduleError(
                        "decoded MXM load Macro is invalid");
            } else if (context.instruction.opcode == MxmControlOpcode::IW
                || (schedule.induction_target
                        != IcuInductionTarget::None
                    && schedule.induction_target
                        != IcuInductionTarget::MxmAccumulatorAddress)) {
                throw StaticScheduleError(
                    "decoded MXM compute Macro is invalid");
            }
        } else {
            if (schedule.induction_target != IcuInductionTarget::None
                || schedule.inner_stride != 0
                || schedule.outer_stride != 0)
                throw StaticScheduleError(
                    "decoded MXM dequant Macro is invalid");
        }
    }

    IcuMacroQueueKind kind_;
    std::vector<RawWord> image_;
    std::optional<PendingFetch> pending_fetch_{};
    std::size_t next_fetch_address_{0};
    Reservoir reservoir_{};
    State state_{State::WaitingForControl};
    bool control_latched_{false};
    bool disabled_{false};
    bool waiting_for_bits_{false};
    std::uint32_t command_count_{0};
    std::array<Delta, 7> dictionary_{};
    std::size_t dictionary_count_{0};
    std::size_t dictionary_index_{0};
    std::uint64_t current_start_cycle_{0};
    std::int64_t current_operand_{0};
    Template current_template_{};
    TemplatePlan template_plan_{};
    std::array<std::uint64_t, kTemplatePayloadWords>
        template_payload_words_{};
    std::size_t template_payload_count_{0};
    std::size_t run_remaining_{0};
    std::size_t decoded_count_{0};
    std::size_t run_decode_start_cycle_{0};
    DecodedDescriptor building_descriptor_{};
    std::unique_ptr<DecodedDescriptor> commit_buffer_{};
    std::deque<DecodedDescriptor> ddb_{};
    std::deque<std::uint8_t> assembly_bits_{};
    std::size_t assembly_target_bits_{0};
    std::size_t delta_symbol_{0};
    IcuMacroDecoderStatistics statistics_{};
};

} // namespace ftlpu
