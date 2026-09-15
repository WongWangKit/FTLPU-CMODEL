#pragma once

#include "ftlpu/core/instruction_codec.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
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
            throw std::out_of_range("ICU raw i-MEM bit index is outside the word");
        return ((lanes[index / 32] >> (index % 32)) & 1u) != 0;
    }
};

using IcuRawImemWord96 = IcuRawImemWord<96>;
using IcuRawImemWord128 = IcuRawImemWord<128>;

struct IcuMacroDecoderStatistics {
    std::size_t fetched_words{0};
    std::size_t decoded_contexts{0};
    std::size_t bit_wait_cycles{0};
    std::size_t context_stall_cycles{0};
    std::size_t peak_reservoir_bits{0};
};

template <typename FuncInstruction>
struct DecodedIcuMacroContext {
    std::size_t record_index{0};
    IcuMacroSchedule schedule{};
    FuncInstruction instruction{};
};

// Cycle-stepped decoder for the independent QueueMode Macro packed-v1 image.
// It owns a finite raw i-MEM image, starts at word 0, fetches at most one fixed
// width word per tick, and emits at most one decoded Macro context per tick.
// Integer fields and words are consumed low-bit first.
template <typename FuncInstruction,
          std::size_t WordBits,
          std::size_t FetchLatency = 1,
          std::size_t ReservoirWords = 4>
class IcuMacroV1Decoder {
  public:
    using RawWord = IcuRawImemWord<WordBits>;
    using DecodedContext = DecodedIcuMacroContext<FuncInstruction>;

    static_assert(FetchLatency > 0);
    static_assert(ReservoirWords >= 3);

    IcuMacroV1Decoder(IcuMacroQueueKind kind, std::vector<RawWord> image)
        : kind_(kind), image_(std::move(image))
    {
        validate_kind();
        if (image_.empty())
            throw StaticScheduleError("Macro raw i-MEM image has no control word");
    }

    std::optional<DecodedContext> tick(bool context_available)
    {
        waiting_for_bits_ = false;
        commit_ready_fetch();
        auto decoded = process(context_available);
        begin_fetch_if_possible();
        age_pending_fetch();

        if (waiting_for_bits_) {
            ++statistics_.bit_wait_cycles;
            if (!pending_fetch_.has_value()
                && next_fetch_address_ == image_.size())
                throw StaticScheduleError("truncated Macro v1 raw i-MEM image");
        }
        return decoded;
    }

    bool done() const noexcept
    {
        return disabled_ || state_ == State::Done;
    }

    bool control_latched() const noexcept { return control_latched_; }
    bool enabled() const noexcept { return control_latched_ && !disabled_; }
    std::uint32_t command_count() const noexcept { return command_count_; }
    std::size_t decoded_count() const noexcept { return decoded_count_; }
    const IcuMacroDecoderStatistics& statistics() const noexcept
    {
        return statistics_;
    }

  private:
    static constexpr std::size_t kReservoirCapacity =
        ReservoirWords * WordBits;
    static constexpr unsigned kCompactStartDeltaBits = 22;
    static constexpr unsigned kCompactOperandDeltaBits = 14;
    static constexpr unsigned kMemAddressBits = 13;

    enum class State {
        WaitingForControl,
        DictionaryCount,
        DictionaryEntry,
        InitialState,
        RunHeader,
        Template,
        Record,
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

    class Cursor {
      public:
        explicit Cursor(const std::deque<std::uint8_t>& bits) : bits_(bits) {}

        std::optional<std::uint64_t> read(unsigned width)
        {
            if (width > 64)
                throw std::invalid_argument("Macro decoder field is wider than 64 bits");
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
        return static_cast<std::int64_t>(raw | (~std::uint64_t{0} << width));
    }

    static bool read(Cursor& cursor, unsigned width, std::uint64_t& value)
    {
        const auto result = cursor.read(width);
        if (!result.has_value()) return false;
        value = *result;
        return true;
    }

    void consume(std::size_t count)
    {
        if (count > reservoir_.size())
            throw std::logic_error("Macro decoder consumed beyond its reservoir");
        while (count-- != 0) reservoir_.pop_front();
    }

    void validate_kind() const
    {
        if constexpr (std::is_same_v<FuncInstruction, MemInstruction>) {
            if (kind_ != IcuMacroQueueKind::Mem)
                throw std::invalid_argument("MEM Macro decoder has a non-MEM queue kind");
        } else if constexpr (std::is_same_v<FuncInstruction,
                                 MxmControlInstruction>) {
            if (kind_ != IcuMacroQueueKind::MxmLoad
                && kind_ != IcuMacroQueueKind::MxmCompute)
                throw std::invalid_argument("MXM control Macro decoder has an invalid queue kind");
        } else if constexpr (std::is_same_v<FuncInstruction,
                                 MxmDequantInstruction>) {
            if (kind_ != IcuMacroQueueKind::MxmDequant)
                throw std::invalid_argument("MXM dequant Macro decoder has an invalid queue kind");
        } else {
            throw std::invalid_argument("Macro v1 is unsupported for this ICU instruction type");
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
            std::uint32_t low = word.lanes[0];
            if ((low & 0xf0000000u) != 0)
                throw StaticScheduleError(
                    "Macro control word has non-zero reserved bits");
            for (std::size_t lane = 1; lane < word.lanes.size(); ++lane)
                if (word.lanes[lane] != 0)
                    throw StaticScheduleError("Macro control word has non-zero reserved bits");
            const bool valid = (low & 1u) != 0;
            const bool enable = (low & 2u) != 0;
            const auto mode = static_cast<IcuQueueMode>((low >> 2) & 3u);
            if (!valid)
                throw StaticScheduleError("Macro raw i-MEM queue is not valid");
            if (mode != IcuQueueMode::Macro)
                throw StaticScheduleError("Macro raw i-MEM queue has a non-Macro QueueMode");
            command_count_ = (low >> 4) & 0x00ffffffu;
            control_latched_ = true;
            disabled_ = !enable;
            state_ = disabled_ || command_count_ == 0
                ? State::Done : State::DictionaryCount;
            return;
        }

        for (std::size_t bit = 0; bit < WordBits; ++bit)
            reservoir_.push_back(word.bit(bit) ? 1u : 0u);
        statistics_.peak_reservoir_bits = std::max(
            statistics_.peak_reservoir_bits, reservoir_.size());
    }

    void begin_fetch_if_possible()
    {
        if (pending_fetch_.has_value() || done()
            || next_fetch_address_ == image_.size())
            return;
        if (next_fetch_address_ != 0
            && reservoir_.size() + WordBits > kReservoirCapacity)
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

    std::optional<DecodedContext> process(bool context_available)
    {
        if (state_ == State::WaitingForControl) {
            waiting_for_bits_ = true;
            return std::nullopt;
        }

        for (;;) {
            switch (state_) {
            case State::WaitingForControl:
                waiting_for_bits_ = true;
                return std::nullopt;
            case State::DictionaryCount:
                if (!read_dictionary_count()) return wait_for_bits();
                break;
            case State::DictionaryEntry:
                if (!read_dictionary_entry()) return wait_for_bits();
                break;
            case State::InitialState:
                if (!read_initial_state()) return wait_for_bits();
                break;
            case State::RunHeader:
                if (!read_run_header()) return wait_for_bits();
                break;
            case State::Template:
                if (!read_template()) return wait_for_bits();
                break;
            case State::Record:
                if (!context_available) {
                    ++statistics_.context_stall_cycles;
                    return std::nullopt;
                }
                return read_record();
            case State::Done:
                return std::nullopt;
            }
        }
    }

    std::optional<DecodedContext> wait_for_bits()
    {
        waiting_for_bits_ = true;
        return std::nullopt;
    }

    bool read_dictionary_count()
    {
        Cursor cursor(reservoir_);
        std::uint64_t count = 0;
        if (!read(cursor, 3, count)) return false;
        if (count > dictionary_.size())
            throw StaticScheduleError("Macro dictionary count exceeds seven entries");
        dictionary_count_ = static_cast<std::size_t>(count);
        consume(cursor.consumed());
        state_ = dictionary_count_ == 0
            ? State::InitialState : State::DictionaryEntry;
        return true;
    }

    bool read_dictionary_entry()
    {
        Cursor cursor(reservoir_);
        std::uint64_t start = 0;
        std::uint64_t operand = 0;
        if (!read(cursor, kCompactStartDeltaBits, start)
            || !read(cursor, kCompactOperandDeltaBits, operand))
            return false;
        dictionary_[dictionary_index_++] = Delta{
            static_cast<std::uint32_t>(start),
            static_cast<std::int32_t>(signed_value(
                operand, kCompactOperandDeltaBits))};
        consume(cursor.consumed());
        if (dictionary_index_ == dictionary_count_)
            state_ = State::InitialState;
        return true;
    }

    bool read_initial_state()
    {
        Cursor cursor(reservoir_);
        std::uint64_t start = 0;
        std::uint64_t operand = 0;
        if (!read(cursor, 32, start)
            || (operand_bits() != 0
                && !read(cursor, operand_bits(), operand)))
            return false;
        current_start_cycle_ = start;
        current_operand_ = static_cast<std::int64_t>(operand);
        consume(cursor.consumed());
        state_ = State::RunHeader;
        return true;
    }

    bool read_run_header()
    {
        Cursor cursor(reservoir_);
        std::uint64_t long_run = 0;
        std::uint64_t encoded_length = 0;
        if (!read(cursor, 1, long_run)
            || (long_run != 0 && !read(cursor, 16, encoded_length)))
            return false;
        run_remaining_ = long_run != 0
            ? static_cast<std::size_t>(encoded_length + 1) : 1;
        if (run_remaining_ > command_count_ - decoded_count_)
            throw StaticScheduleError("Macro run exceeds declared command count");
        consume(cursor.consumed());
        state_ = State::Template;
        return true;
    }

    bool read_template()
    {
        Cursor cursor(reservoir_);
        Template result;
        if constexpr (std::is_same_v<FuncInstruction, MemInstruction>) {
            if (!read_mem_template(cursor, result)) return false;
        } else {
            if (!read_mxm_template(cursor, result)) return false;
        }
        current_template_ = std::move(result);
        consume(cursor.consumed());
        state_ = State::Record;
        return true;
    }

    bool read_mem_template(Cursor& cursor, Template& result)
    {
        std::uint64_t extended = 0;
        if (!read(cursor, 1, extended)) return false;
        if (extended != 0) {
            std::uint64_t native = 0, inner_count = 0, inner_interval = 0;
            std::uint64_t inner_stride = 0, outer_count = 0;
            std::uint64_t outer_interval = 0, outer_stride = 0, target = 0;
            if (!read(cursor, 32, native)
                || !read(cursor, 32, inner_count)
                || !read(cursor, 32, inner_interval)
                || !read(cursor, 32, inner_stride)
                || !read(cursor, 32, outer_count)
                || !read(cursor, 32, outer_interval)
                || !read(cursor, 32, outer_stride)
                || !read(cursor, 2, target))
                return false;
            result.instruction = isa::decode_mem_instruction(native);
            result.schedule = IcuMacroSchedule{
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

        std::uint64_t shape_value = 0, write = 0, stream = 0, preserve = 0;
        if (!read(cursor, 2, shape_value)
            || !read(cursor, 1, write)
            || !read(cursor, 6, stream)
            || !read(cursor, 1, preserve))
            return false;
        const auto shape = static_cast<Shape>(shape_value);
        result.instruction = write != 0
            ? (preserve != 0
                ? MemInstruction::WriteTap(0, stream)
                : MemInstruction::Write(0, stream))
            : MemInstruction::Read(0, stream);
        result.schedule = IcuMacroSchedule{
            0, 1, 1, 0, 1, 1, 0, IcuInductionTarget::MemAddress};
        return read_compact_schedule(cursor, shape, result.schedule);
    }

    bool read_mxm_template(Cursor& cursor, Template& result)
    {
        std::uint64_t extended = 0;
        std::uint64_t native = 0;
        if (!read(cursor, 1, extended)
            || !read(cursor, native_instruction_bits(), native))
            return false;
        result.instruction = decode_native(native);
        if (extended != 0) {
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
            result.schedule = IcuMacroSchedule{
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

        std::uint64_t shape_value = 0, target = 0;
        if (!read(cursor, 2, shape_value) || !read(cursor, 2, target))
            return false;
        result.schedule = IcuMacroSchedule{
            0, 1, 1, 0, 1, 1, 0,
            static_cast<IcuInductionTarget>(target)};
        return read_compact_schedule(
            cursor, static_cast<Shape>(shape_value), result.schedule);
    }

    static bool read_compact_schedule(
        Cursor& cursor, Shape shape, IcuMacroSchedule& schedule)
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

    std::optional<Delta> read_delta()
    {
        Cursor cursor(reservoir_);
        std::uint64_t bit = 0;
        std::size_t symbol = 0;
        if (!read(cursor, 1, bit)) return std::nullopt;
        if (bit == 0) {
            symbol = 0;
        } else {
            if (!read(cursor, 1, bit)) return std::nullopt;
            if (bit == 0) {
                symbol = 1;
            } else {
                if (!read(cursor, 1, bit)) return std::nullopt;
                if (bit == 0) {
                    if (!read(cursor, 1, bit)) return std::nullopt;
                    symbol = bit != 0 ? 3 : 2;
                } else {
                    if (!read(cursor, 1, bit)) return std::nullopt;
                    if (bit == 0) {
                        if (!read(cursor, 1, bit)) return std::nullopt;
                        symbol = bit != 0 ? 5 : 4;
                    } else {
                        if (!read(cursor, 1, bit)) return std::nullopt;
                        symbol = bit != 0 ? 7 : 6;
                    }
                }
            }
        }

        Delta result;
        if (symbol < dictionary_count_) {
            result = dictionary_[symbol];
        } else {
            if (symbol != 7)
                throw StaticScheduleError("Macro delta references a missing dictionary entry");
            std::uint64_t wide = 0, start = 0, operand = 0;
            if (!read(cursor, 1, wide)) return std::nullopt;
            if (wide == 0) {
                if (!read(cursor, kCompactStartDeltaBits, start)
                    || !read(cursor, kCompactOperandDeltaBits, operand))
                    return std::nullopt;
                result = Delta{
                    static_cast<std::uint32_t>(start),
                    static_cast<std::int32_t>(signed_value(
                        operand, kCompactOperandDeltaBits))};
            } else {
                if (!read(cursor, 32, start) || !read(cursor, 32, operand))
                    return std::nullopt;
                result = Delta{
                    static_cast<std::uint32_t>(start),
                    static_cast<std::int32_t>(operand)};
            }
        }
        consume(cursor.consumed());
        return result;
    }

    std::optional<DecodedContext> read_record()
    {
        if (decoded_count_ != 0) {
            const auto delta = read_delta();
            if (!delta.has_value()) return wait_for_bits();
            current_start_cycle_ += delta->start_cycle;
            current_operand_ += delta->operand;
        }

        if (current_start_cycle_ > std::numeric_limits<std::uint32_t>::max())
            throw StaticScheduleError("decoded Macro start cycle is out of range");
        const auto width = operand_bits();
        if (current_operand_ < 0
            || (width != 0
                && static_cast<std::uint64_t>(current_operand_)
                    >= (std::uint64_t{1} << width)))
            throw StaticScheduleError("decoded Macro operand is out of range");

        auto result = DecodedContext{
            decoded_count_, current_template_.schedule,
            current_template_.instruction};
        result.schedule.start_cycle =
            static_cast<std::size_t>(current_start_cycle_);
        set_operand(result.instruction,
            static_cast<std::size_t>(current_operand_));
        validate_context(result);

        ++decoded_count_;
        ++statistics_.decoded_contexts;
        if (--run_remaining_ == 0)
            state_ = decoded_count_ == command_count_
                ? State::Done : State::RunHeader;
        else if (decoded_count_ == command_count_)
            throw StaticScheduleError("Macro command count ends inside a run");
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
                throw StaticScheduleError("MXM dequant Macro has an operand");
        }
    }

    void validate_context(const DecodedContext& context) const
    {
        const auto& schedule = context.schedule;
        if (schedule.inner_count == 0 || schedule.outer_count == 0
            || schedule.inner_interval == 0 || schedule.outer_interval == 0
            || (schedule.outer_count > 1
                && schedule.outer_interval
                    <= (schedule.inner_count - 1)
                        * schedule.inner_interval))
            throw StaticScheduleError("decoded Macro has an invalid iteration space");

        if constexpr (std::is_same_v<FuncInstruction, MemInstruction>) {
            if (schedule.induction_target != IcuInductionTarget::MemAddress)
                throw StaticScheduleError("decoded MEM Macro has an invalid induction target");
        } else if constexpr (std::is_same_v<FuncInstruction,
                                 MxmControlInstruction>) {
            if (kind_ == IcuMacroQueueKind::MxmLoad) {
                if (context.instruction.opcode != MxmControlOpcode::IW
                    || (schedule.induction_target != IcuInductionTarget::None
                        && schedule.induction_target
                            != IcuInductionTarget::MxmWeightColumn))
                    throw StaticScheduleError("decoded MXM load Macro is invalid");
            } else if (context.instruction.opcode == MxmControlOpcode::IW
                || (schedule.induction_target != IcuInductionTarget::None
                    && schedule.induction_target
                        != IcuInductionTarget::MxmAccumulatorAddress)) {
                throw StaticScheduleError("decoded MXM compute Macro is invalid");
            }
        } else {
            if (schedule.induction_target != IcuInductionTarget::None
                || schedule.inner_stride != 0 || schedule.outer_stride != 0)
                throw StaticScheduleError("decoded MXM dequant Macro is invalid");
        }
    }

    IcuMacroQueueKind kind_;
    std::vector<RawWord> image_;
    std::optional<PendingFetch> pending_fetch_{};
    std::size_t next_fetch_address_{0};
    std::deque<std::uint8_t> reservoir_{};
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
    std::size_t run_remaining_{0};
    std::size_t decoded_count_{0};
    IcuMacroDecoderStatistics statistics_{};
};

} // namespace ftlpu
