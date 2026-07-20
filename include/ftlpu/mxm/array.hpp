#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/mxm/supercell.hpp"

#include <array>
#include <cstddef>
#include <ostream>
#include <stdexcept>

namespace ftlpu {

class MxmArray {
public:
    using Supercell = MxmSupercell;
    using InputVector = Supercell::InputVector;
    using Weight = Supercell::Weight;

    void reset()
    {
        for (auto& row : cells_) {
            for (auto& cell : row) {
                cell.reset();
            }
        }
    }

    Supercell& cell(std::size_t row, std::size_t column)
    {
        check_supercell(row, column);
        return cells_[row][column];
    }

    const Supercell& cell(std::size_t row, std::size_t column) const
    {
        check_supercell(row, column);
        return cells_[row][column];
    }

    Weight weight(
        std::size_t buffer,
        std::size_t supercell_row,
        std::size_t supercell_column,
        std::size_t row,
        std::size_t column) const
    {
        return cell(supercell_row, supercell_column).weight(buffer, row, column);
    }

    Weight weight(
        std::size_t supercell_row,
        std::size_t supercell_column,
        std::size_t row,
        std::size_t column) const
    {
        return weight(0, supercell_row, supercell_column, row, column);
    }

    Weight buffered_weight(
        std::size_t supercell_row,
        std::size_t supercell_column,
        std::size_t row,
        std::size_t column) const
    {
        return cell(supercell_row, supercell_column).weight(0, row, column);
    }

    Weight buffered_weight(
        std::size_t buffer,
        std::size_t supercell_row,
        std::size_t supercell_column,
        std::size_t row,
        std::size_t column) const
    {
        return cell(supercell_row, supercell_column).weight(buffer, row, column);
    }

    void tick_cell_iw_load(std::size_t row, std::size_t column, std::size_t buffer, InputVector input, std::ostream& os)
    {
        auto& target = cell(row, column);
        os << "mxm_array cell(" << row << "," << column << ") ";
        target.set_input(input);
        target.issue(MxmInstruction::IW(buffer));
        target.tick(os);
    }

private:
    static void check_supercell(std::size_t row, std::size_t column)
    {
        if (row >= hw::kMxmSupercellsPerPlane) {
            throw std::out_of_range("MXM supercell row is outside the configured array");
        }
        if (column >= hw::kMxmSupercellsPerPlane) {
            throw std::out_of_range("MXM supercell column is outside the configured array");
        }
    }

    std::array<std::array<Supercell, hw::kMxmSupercellsPerPlane>, hw::kMxmSupercellsPerPlane> cells_{};
};

} // namespace ftlpu
