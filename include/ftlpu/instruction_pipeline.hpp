#pragma once

#include "ftlpu/hardware_params.hpp"
#include "ftlpu/mem_slice.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>

namespace ftlpu {

class NorthboundInstructionPipeline {
public:
    using Slot = std::optional<MemInstruction>;

    void reset()
    {
        for (auto& slot : rows_) {
            slot.reset();
        }
        cycle_ = 0;
    }

    std::size_t cycle() const
    {
        return cycle_;
    }

    const Slot& row(std::size_t row_index) const
    {
        if (row_index >= hw::kTileRows) {
            throw std::out_of_range("instruction pipeline row is outside the TSP grid");
        }
        return rows_[row_index];
    }

    void issue_south(MemInstruction instruction)
    {
        if (rows_[0].has_value()) {
            throw std::logic_error("south instruction input is occupied");
        }
        rows_[0] = instruction;
    }

    void tick(std::ostream& os)
    {
        print_cycle(os);

        for (std::size_t row = hw::kTileRows - 1; row > 0; --row) {
            rows_[row] = rows_[row - 1];
        }
        rows_[0].reset();
        ++cycle_;
    }

private:
    static const char* opcode_name(MemOpcode opcode)
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
        return "Unknown";
    }

    static void print_instruction(std::ostream& os, const MemInstruction& instruction)
    {
        os << opcode_name(instruction.opcode);
        switch (instruction.opcode) {
        case MemOpcode::Read:
        case MemOpcode::Write:
            os << " a=" << instruction.address << " s=" << instruction.stream;
            break;
        case MemOpcode::Gather:
        case MemOpcode::Scatter:
            os << " s=" << instruction.stream << " map=" << instruction.map_stream;
            break;
        }
    }

    void print_cycle(std::ostream& os) const
    {
        os << "cycle " << cycle_ << ":";

        bool any = false;
        for (std::size_t row = 0; row < hw::kTileRows; ++row) {
            if (!rows_[row].has_value()) {
                continue;
            }

            any = true;
            os << " row " << row << " executes ";
            print_instruction(os, *rows_[row]);
            os << ";";
        }

        if (!any) {
            os << " idle;";
        }
        os << '\n';
    }

    std::array<Slot, hw::kTileRows> rows_{};
    std::size_t cycle_{0};
};

} // namespace ftlpu
