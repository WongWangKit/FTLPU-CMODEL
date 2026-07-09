#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/mem/tile_array.hpp"
#include "ftlpu/mxm/mxm.hpp"
#include "ftlpu/vxm/slice.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <ostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ftlpu {

class InstructionControlUnit {
public:
    static constexpr std::size_t kVxmQueues = VxmSlice::kAluQueues;
    static constexpr std::size_t kMemQueues = hw::kSliceColumns;
    static constexpr std::size_t kMxmQueues = 2;

    struct Repeat {
        std::size_t count{0};
        std::size_t interval{1};
        std::int64_t address_stride{0};
    };

    void reset()
    {
        for (auto& queue : vxm_queues_) {
            queue.reset();
        }
        for (auto& queue : mem_queues_) {
            queue.reset();
        }
        for (auto& queue : mxm_load_queues_) {
            queue.reset();
        }
        for (auto& queue : mxm_compute_queues_) {
            queue.reset();
        }
        for (auto& queue : mxm_output_queues_) {
            queue.reset();
        }
        cycle_ = 0;
    }

    void enqueue_nop(std::size_t cycles)
    {
        for (auto& queue : vxm_queues_) {
            queue.push_nop(cycles);
        }
        for (auto& queue : mem_queues_) {
            queue.push_nop(cycles);
        }
        for (auto& queue : mxm_load_queues_) {
            queue.push_nop(cycles);
        }
        for (auto& queue : mxm_compute_queues_) {
            queue.push_nop(cycles);
        }
        for (auto& queue : mxm_output_queues_) {
            queue.push_nop(cycles);
        }
    }

    void enqueue_vxm(std::size_t alu, VxmLaneAluInstruction instruction)
    {
        check_vxm_queue(alu);
        vxm_queues_[alu].push_instruction(instruction);
    }

    void enqueue_vxm_nop(std::size_t alu, std::size_t cycles)
    {
        check_vxm_queue(alu);
        vxm_queues_[alu].push_nop(cycles);
    }

    void enqueue_vxm_repeat(std::size_t alu, std::size_t count, std::size_t interval = 1)
    {
        check_vxm_queue(alu);
        vxm_queues_[alu].push_repeat(Repeat {count, interval, 0});
    }

    void enqueue_mem(std::size_t column, MemInstruction instruction)
    {
        check_mem_queue(column);
        mem_queues_[column].push_instruction(instruction);
    }

    void enqueue_mem_nop(std::size_t column, std::size_t cycles)
    {
        check_mem_queue(column);
        mem_queues_[column].push_nop(cycles);
    }

    void enqueue_mem_repeat(
        std::size_t column,
        std::size_t count,
        std::size_t interval = 1,
        std::int64_t address_stride = 0)
    {
        check_mem_queue(column);
        mem_queues_[column].push_repeat(Repeat {count, interval, address_stride});
    }

    void enqueue_mxm(std::size_t mxm, MxmControlInstruction instruction)
    {
        check_mxm_queue(mxm);
        if (instruction.opcode == MxmControlOpcode::Output) {
            mxm_output_queues_[mxm].push_instruction(instruction);
        } else if (instruction.opcode == MxmControlOpcode::Compute) {
            mxm_compute_queues_[mxm].push_instruction(instruction);
        } else {
            mxm_load_queues_[mxm].push_instruction(instruction);
        }
    }

    void enqueue_mxm_nop(std::size_t mxm, std::size_t cycles)
    {
        check_mxm_queue(mxm);
        mxm_load_queues_[mxm].push_nop(cycles);
        mxm_compute_queues_[mxm].push_nop(cycles);
    }

    void enqueue_mxm_load_nop(std::size_t mxm, std::size_t cycles)
    {
        check_mxm_queue(mxm);
        mxm_load_queues_[mxm].push_nop(cycles);
    }

    void enqueue_mxm_compute_nop(std::size_t mxm, std::size_t cycles)
    {
        check_mxm_queue(mxm);
        mxm_compute_queues_[mxm].push_nop(cycles);
    }

    void enqueue_mxm_repeat(std::size_t mxm, std::size_t count, std::size_t interval = 1)
    {
        check_mxm_queue(mxm);
        mxm_compute_queues_[mxm].push_repeat(Repeat {count, interval, 0});
    }

    void enqueue_mxm_load_repeat(std::size_t mxm, std::size_t count, std::size_t interval = 1)
    {
        check_mxm_queue(mxm);
        mxm_load_queues_[mxm].push_repeat(Repeat {count, interval, 0});
    }

    void enqueue_mxm_compute_repeat(std::size_t mxm, std::size_t count, std::size_t interval = 1)
    {
        check_mxm_queue(mxm);
        mxm_compute_queues_[mxm].push_repeat(Repeat {count, interval, 0});
    }

    void enqueue_mxm_output_nop(std::size_t mxm, std::size_t cycles)
    {
        check_mxm_queue(mxm);
        mxm_output_queues_[mxm].push_nop(cycles);
    }

    void enqueue_mxm_output_repeat(std::size_t mxm, std::size_t count, std::size_t interval = 1)
    {
        check_mxm_queue(mxm);
        mxm_output_queues_[mxm].push_repeat(Repeat {count, interval, 0});
    }

    void dispatch_vxm(VxmSlice& vxm, std::ostream* os = nullptr)
    {
        log_cycle_header(os);
        bool any = false;
        for (std::size_t alu = 0; alu < kVxmQueues; ++alu) {
            const auto instruction = vxm_queues_[alu].dispatch_next();
            if (!instruction.has_value()) {
                continue;
            }
            vxm.issue_south(alu, *instruction);
            any = true;
            if (os != nullptr) {
                *os << "  ICU -> VXM alu" << alu << ' ' << describe_vxm(*instruction) << '\n';
            }
        }
        log_dispatch_idle(os, any);
        ++cycle_;
    }

    void dispatch(
        TileArrayModel& mem,
        VxmSlice& vxm,
        std::array<Mxm, kMxmQueues>& mxms,
        std::ostream* os = nullptr)
    {
        log_cycle_header(os);

        bool any = false;
        for (std::size_t alu = 0; alu < kVxmQueues; ++alu) {
            const auto instruction = vxm_queues_[alu].dispatch_next();
            if (!instruction.has_value()) {
                continue;
            }
            vxm.issue_south(alu, *instruction);
            any = true;
            if (os != nullptr) {
                *os << "  ICU -> VXM alu" << alu << ' ' << describe_vxm(*instruction) << '\n';
            }
        }

        for (std::size_t column = 0; column < kMemQueues; ++column) {
            const auto instruction = mem_queues_[column].dispatch_next();
            if (!instruction.has_value()) {
                continue;
            }
            mem.enqueue_instruction(column, *instruction);
            any = true;
            if (os != nullptr) {
                *os << "  ICU -> MEM q" << column << ' ' << describe_mem(*instruction) << '\n';
            }
        }

        for (std::size_t mxm = 0; mxm < kMxmQueues; ++mxm) {
            const auto instruction = mxm_load_queues_[mxm].dispatch_next();
            if (!instruction.has_value()) {
                continue;
            }
            mxms[mxm].control().issue_south(*instruction);
            any = true;
            if (os != nullptr) {
                *os << "  ICU -> MXM" << mxm << ".load " << describe_mxm(*instruction) << '\n';
            }
        }

        for (std::size_t mxm = 0; mxm < kMxmQueues; ++mxm) {
            const auto instruction = mxm_compute_queues_[mxm].dispatch_next();
            if (!instruction.has_value()) {
                continue;
            }
            mxms[mxm].control().issue_south(*instruction);
            any = true;
            if (os != nullptr) {
                *os << "  ICU -> MXM" << mxm << ".compute " << describe_mxm(*instruction) << '\n';
            }
        }

        for (std::size_t mxm = 0; mxm < kMxmQueues; ++mxm) {
            const auto instruction = mxm_output_queues_[mxm].dispatch_next();
            if (!instruction.has_value()) {
                continue;
            }
            mxms[mxm].control().issue_south(*instruction);
            any = true;
            if (os != nullptr) {
                *os << "  ICU -> MXM" << mxm << ".output " << describe_mxm(*instruction) << '\n';
            }
        }

        log_dispatch_idle(os, any);
        ++cycle_;
    }

    std::size_t cycle() const
    {
        return cycle_;
    }

private:
    enum class QueueCommandKind {
        Instruction,
        Nop,
        Repeat,
    };

    template <typename Instruction>
    struct QueueCommand {
        QueueCommandKind kind{QueueCommandKind::Instruction};
        Instruction instruction{};
        Repeat repeat{};
        std::size_t nop_cycles{0};
    };

    template <typename Instruction>
    class DispatchQueue {
    public:
        void reset()
        {
            commands_.clear();
            last_instruction_.reset();
            repeat_instruction_.reset();
            repeat_remaining_ = 0;
            repeat_interval_ = 1;
            repeat_cooldown_ = 0;
            repeat_address_stride_ = 0;
            repeat_index_ = 0;
            nop_remaining_ = 0;
        }

        void push_instruction(Instruction instruction)
        {
            commands_.push_back(QueueCommand<Instruction> {QueueCommandKind::Instruction, instruction});
        }

        void push_nop(std::size_t cycles)
        {
            if (cycles == 0) {
                return;
            }
            auto command = QueueCommand<Instruction> {};
            command.kind = QueueCommandKind::Nop;
            command.nop_cycles = cycles;
            commands_.push_back(command);
        }

        void push_repeat(Repeat repeat)
        {
            if (repeat.count == 0) {
                return;
            }
            if (repeat.interval == 0) {
                throw std::invalid_argument("ICU repeat interval must be at least one cycle");
            }
            auto command = QueueCommand<Instruction> {};
            command.kind = QueueCommandKind::Repeat;
            command.repeat = repeat;
            commands_.push_back(command);
        }

        std::optional<Instruction> dispatch_next()
        {
            if (nop_remaining_ > 0) {
                --nop_remaining_;
                return std::nullopt;
            }

            if (repeat_remaining_ > 0) {
                if (repeat_cooldown_ > 0) {
                    --repeat_cooldown_;
                    return std::nullopt;
                }
                auto instruction = apply_repeat_stride(*repeat_instruction_, repeat_address_stride_, repeat_index_);
                --repeat_remaining_;
                ++repeat_index_;
                if (repeat_remaining_ > 0) {
                    repeat_cooldown_ = repeat_interval_ - 1;
                }
                last_instruction_ = instruction;
                return instruction;
            }

            while (!commands_.empty()) {
                const auto command = commands_.front();
                commands_.pop_front();
                if (command.kind == QueueCommandKind::Instruction) {
                    last_instruction_ = command.instruction;
                    return command.instruction;
                }
                if (command.kind == QueueCommandKind::Nop) {
                    nop_remaining_ = command.nop_cycles - 1;
                    return std::nullopt;
                }
                if (!last_instruction_.has_value()) {
                    throw std::logic_error("ICU Repeat needs a previous instruction in the same queue");
                }
                repeat_instruction_ = *last_instruction_;
                repeat_remaining_ = command.repeat.count;
                repeat_interval_ = command.repeat.interval;
                repeat_cooldown_ = command.repeat.interval - 1;
                repeat_address_stride_ = command.repeat.address_stride;
                repeat_index_ = 1;
                if (repeat_cooldown_ > 0) {
                    --repeat_cooldown_;
                    return std::nullopt;
                }
                auto instruction = apply_repeat_stride(*repeat_instruction_, repeat_address_stride_, repeat_index_);
                --repeat_remaining_;
                ++repeat_index_;
                if (repeat_remaining_ > 0) {
                    repeat_cooldown_ = repeat_interval_ - 1;
                }
                last_instruction_ = instruction;
                return instruction;
            }

            return std::nullopt;
        }

        std::size_t queued_count() const
        {
            return commands_.size() + repeat_remaining_ + nop_remaining_;
        }

    private:
        std::deque<QueueCommand<Instruction>> commands_{};
        std::optional<Instruction> last_instruction_{};
        std::optional<Instruction> repeat_instruction_{};
        std::size_t repeat_remaining_{0};
        std::size_t repeat_interval_{1};
        std::size_t repeat_cooldown_{0};
        std::int64_t repeat_address_stride_{0};
        std::size_t repeat_index_{0};
        std::size_t nop_remaining_{0};
    };

    template <typename Instruction>
    static Instruction apply_repeat_stride(Instruction instruction, std::int64_t, std::size_t)
    {
        return instruction;
    }

    static MemInstruction apply_repeat_stride(
        MemInstruction instruction,
        std::int64_t address_stride,
        std::size_t repeat_index)
    {
        const auto delta = address_stride * static_cast<std::int64_t>(repeat_index);
        if (delta < 0 && instruction.address < static_cast<std::size_t>(-delta)) {
            throw std::out_of_range("ICU MEM Repeat address stride underflow");
        }
        instruction.address = static_cast<std::size_t>(static_cast<std::int64_t>(instruction.address) + delta);
        return instruction;
    }

    static void check_vxm_queue(std::size_t alu)
    {
        if (alu >= kVxmQueues) {
            throw std::out_of_range("ICU VXM queue is outside the 16 ALU queues");
        }
    }

    static void check_mem_queue(std::size_t column)
    {
        if (column >= kMemQueues) {
            throw std::out_of_range("ICU MEM queue is outside the 44 MEM queues");
        }
    }

    static void check_mxm_queue(std::size_t mxm)
    {
        if (mxm >= kMxmQueues) {
            throw std::out_of_range("ICU MXM queue is outside the two MXM queues");
        }
    }

    template <typename QueueArray>
    static std::size_t queued_instruction_count(const QueueArray& queues)
    {
        std::size_t count = 0;
        for (const auto& queue : queues) {
            count += queue.queued_count();
        }
        return count;
    }

    void log_cycle_header(std::ostream* os) const
    {
        if (os == nullptr) {
            return;
        }

        *os << "icu cycle " << cycle_ << '\n';
        *os << "  queues:"
            << " vxm=" << queued_instruction_count(vxm_queues_)
            << " mem=" << queued_instruction_count(mem_queues_)
            << " mxm_load=" << queued_instruction_count(mxm_load_queues_)
            << " mxm_compute=" << queued_instruction_count(mxm_compute_queues_)
            << " mxm_output=" << queued_instruction_count(mxm_output_queues_)
            << '\n';
    }

    static void log_dispatch_idle(std::ostream* os, bool any)
    {
        if (os != nullptr && !any) {
            *os << "  ICU dispatch idle\n";
        }
    }

    static const char* mem_opcode_name(MemOpcode opcode)
    {
        switch (opcode) {
        case MemOpcode::Read:
            return "Read";
        case MemOpcode::Write:
            return "Write";
        case MemOpcode::Gather:
            return "Gather";
        case MemOpcode::Scatter:
            return "Scatter";
        }
        return "?";
    }

    static std::string describe_mem(const MemInstruction& instruction)
    {
        std::ostringstream os;
        os << mem_opcode_name(instruction.opcode)
           << " address=" << instruction.address
           << " stream=" << instruction.stream;
        if (instruction.opcode == MemOpcode::Gather || instruction.opcode == MemOpcode::Scatter) {
            os << " map_stream=" << instruction.map_stream;
        }
        return os.str();
    }

    static std::string describe_mxm(const MxmControlInstruction& instruction)
    {
        std::ostringstream os;
        if (instruction.opcode == MxmControlOpcode::IW) {
            os << "IW col=" << instruction.supercell_column;
        } else if (instruction.opcode == MxmControlOpcode::LW) {
            os << "LW mask=0x" << std::hex << instruction.column_mask << std::dec;
        } else if (instruction.opcode == MxmControlOpcode::Compute) {
            os << "Compute";
        } else {
            os << "Output stream_base=" << instruction.stream_base;
        }
        return os.str();
    }

    static std::string describe_operand(const VxmLaneOperand& operand)
    {
        std::ostringstream os;
        switch (operand.kind) {
        case VxmLaneOperandKind::AluOutput:
            os << "alu" << operand.index;
            break;
        case VxmLaneOperandKind::StreamInt32:
            os << "stream[" << operand.index << ".." << (operand.index + 3) << "]";
            break;
        case VxmLaneOperandKind::Immediate:
            os << "imm(" << operand.immediate << ")";
            break;
        }
        return os.str();
    }

    static std::string describe_vxm(const VxmLaneAluInstruction& instruction)
    {
        std::ostringstream os;
        os << VxmLane::opcode_name(instruction.opcode)
           << " lhs=" << describe_operand(instruction.lhs)
           << " rhs=" << describe_operand(instruction.rhs);
        if (instruction.opcode == VxmAluOpcode::Cast) {
            os << " cast=" << (instruction.cast_target == VxmCastTarget::Float32 ? "fp32" : "int8");
        }
        os << " scale=" << instruction.scale
           << " zp=" << instruction.output_zero_point;
        if (instruction.output_stream.has_value()) {
            os << " out_stream=" << *instruction.output_stream;
        }
        return os.str();
    }

    std::array<DispatchQueue<VxmLaneAluInstruction>, kVxmQueues> vxm_queues_{};
    std::array<DispatchQueue<MemInstruction>, kMemQueues> mem_queues_{};
    std::array<DispatchQueue<MxmControlInstruction>, kMxmQueues> mxm_load_queues_{};
    std::array<DispatchQueue<MxmControlInstruction>, kMxmQueues> mxm_compute_queues_{};
    std::array<DispatchQueue<MxmControlInstruction>, kMxmQueues> mxm_output_queues_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
