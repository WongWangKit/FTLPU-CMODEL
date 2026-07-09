#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/mem/tile_array.hpp"
#include "ftlpu/mxm/mxm.hpp"
#include "ftlpu/vxm/slice.hpp"

#include <array>
#include <cstddef>
#include <deque>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace ftlpu {

class InstructionControlUnit {
public:
    static constexpr std::size_t kVxmQueues = VxmSlice::kAluQueues;
    static constexpr std::size_t kMemQueues = hw::kSliceColumns;
    static constexpr std::size_t kMxmQueues = 2;

    void reset()
    {
        for (auto& queue : vxm_queues_) {
            queue.clear();
        }
        for (auto& queue : mem_queues_) {
            queue.clear();
        }
        for (auto& queue : mxm_queues_) {
            queue.clear();
        }
        for (auto& queue : mxm_output_queues_) {
            queue.clear();
        }
        nop_cycles_ = 0;
        cycle_ = 0;
    }

    void enqueue_nop(std::size_t cycles)
    {
        nop_cycles_ += cycles;
    }

    void enqueue_vxm(std::size_t alu, VxmLaneAluInstruction instruction)
    {
        check_vxm_queue(alu);
        vxm_queues_[alu].push_back(instruction);
    }

    void enqueue_mem(std::size_t column, MemInstruction instruction)
    {
        check_mem_queue(column);
        mem_queues_[column].push_back(instruction);
    }

    void enqueue_mxm(std::size_t mxm, MxmControlInstruction instruction)
    {
        check_mxm_queue(mxm);
        if (instruction.opcode == MxmControlOpcode::Output) {
            mxm_output_queues_[mxm].push_back(instruction);
        } else {
            mxm_queues_[mxm].push_back(instruction);
        }
    }

    void dispatch_vxm(VxmSlice& vxm, std::ostream* os = nullptr)
    {
        log_cycle_header(os);
        if (consume_nop(os)) {
            ++cycle_;
            return;
        }
        bool any = false;
        for (std::size_t alu = 0; alu < kVxmQueues; ++alu) {
            if (vxm_queues_[alu].empty()) {
                continue;
            }
            vxm.issue_south(alu, vxm_queues_[alu].front());
            const auto instruction = vxm_queues_[alu].front();
            vxm_queues_[alu].pop_front();
            any = true;
            if (os != nullptr) {
                *os << "  ICU -> VXM alu" << alu << ' ' << describe_vxm(instruction) << '\n';
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
        if (consume_nop(os)) {
            ++cycle_;
            return;
        }

        bool any = false;
        for (std::size_t alu = 0; alu < kVxmQueues; ++alu) {
            if (vxm_queues_[alu].empty()) {
                continue;
            }
            const auto instruction = vxm_queues_[alu].front();
            vxm.issue_south(alu, instruction);
            vxm_queues_[alu].pop_front();
            any = true;
            if (os != nullptr) {
                *os << "  ICU -> VXM alu" << alu << ' ' << describe_vxm(instruction) << '\n';
            }
        }

        for (std::size_t column = 0; column < kMemQueues; ++column) {
            if (mem_queues_[column].empty()) {
                continue;
            }
            const auto instruction = mem_queues_[column].front();
            mem.enqueue_instruction(column, instruction);
            mem_queues_[column].pop_front();
            any = true;
            if (os != nullptr) {
                *os << "  ICU -> MEM q" << column << ' ' << describe_mem(instruction) << '\n';
            }
        }

        for (std::size_t mxm = 0; mxm < kMxmQueues; ++mxm) {
            if (mxm_queues_[mxm].empty()) {
                continue;
            }
            const auto instruction = mxm_queues_[mxm].front();
            mxms[mxm].control().issue_south(instruction);
            mxm_queues_[mxm].pop_front();
            any = true;
            if (os != nullptr) {
                *os << "  ICU -> MXM q" << mxm << ' ' << describe_mxm(instruction) << '\n';
            }
        }

        for (std::size_t mxm = 0; mxm < kMxmQueues; ++mxm) {
            if (mxm_output_queues_[mxm].empty()) {
                continue;
            }
            const auto instruction = mxm_output_queues_[mxm].front();
            mxms[mxm].control().issue_south(instruction);
            mxm_output_queues_[mxm].pop_front();
            any = true;
            if (os != nullptr) {
                *os << "  ICU -> MXM q" << mxm << ' ' << describe_mxm(instruction) << '\n';
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

    bool consume_nop(std::ostream* os)
    {
        if (nop_cycles_ == 0) {
            return false;
        }
        --nop_cycles_;
        if (os != nullptr) {
            *os << "  ICU NOP\n";
        }
        return true;
    }

    template <typename QueueArray>
    static std::size_t queued_instruction_count(const QueueArray& queues)
    {
        std::size_t count = 0;
        for (const auto& queue : queues) {
            count += queue.size();
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
            << " mxm=" << (queued_instruction_count(mxm_queues_) + queued_instruction_count(mxm_output_queues_))
            << " nop=" << nop_cycles_ << '\n';
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

    std::array<std::deque<VxmLaneAluInstruction>, kVxmQueues> vxm_queues_{};
    std::array<std::deque<MemInstruction>, kMemQueues> mem_queues_{};
    std::array<std::deque<MxmControlInstruction>, kMxmQueues> mxm_queues_{};
    std::array<std::deque<MxmControlInstruction>, kMxmQueues> mxm_output_queues_{};
    std::size_t nop_cycles_{0};
    std::size_t cycle_{0};
};

} // namespace ftlpu
