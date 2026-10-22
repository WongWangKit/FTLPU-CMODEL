#pragma once

#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/system/tsp_slice_system.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

namespace ftlpu::vmodel_test {

using ScheduleRecord = std::array<std::uint32_t, 15>;

inline void set_record_bits(
    ScheduleRecord& record,
    std::size_t offset,
    std::uint64_t value,
    std::size_t width)
{
    for (std::size_t bit = 0; bit < width; ++bit)
        if ((value >> bit) & 1u)
            record[(offset + bit) / 32] |=
                std::uint32_t {1} << ((offset + bit) % 32);
}

inline ScheduleRecord command_record(
    std::uint8_t queue,
    std::uint32_t command)
{
    auto record = ScheduleRecord {};
    set_record_bits(record, 0, queue, 8);
    set_record_bits(record, 9, command, 32);
    return record;
}

inline ScheduleRecord instruction_record(
    std::uint8_t queue,
    std::uint64_t instruction,
    std::size_t width)
{
    auto record = ScheduleRecord {};
    set_record_bits(record, 0, queue, 8);
    set_record_bits(record, 8, 1, 1);
    set_record_bits(record, 41, instruction, width);
    return record;
}

inline ScheduleRecord vxm_record(
    std::uint8_t queue,
    const isa::EncodedVxmInstruction& instruction)
{
    auto record = ScheduleRecord {};
    set_record_bits(record, 0, queue, 8);
    set_record_bits(record, 8, 1, 1);
    for (std::size_t word = 0; word < instruction.words.size(); ++word)
        set_record_bits(record, 41 + word * 32, instruction.words[word], 32);
    return record;
}

inline std::uint8_t mxm_load_queue(std::size_t mxm)
{
    static constexpr std::array<std::uint8_t, 4> queues {104, 132, 105, 133};
    return queues.at(mxm);
}

inline std::uint8_t mxm_dequant_queue(std::size_t mxm)
{
    static constexpr std::array<std::uint8_t, 4> queues {106, 134, 107, 135};
    return queues.at(mxm);
}

inline std::uint8_t mxm_compute_queue(std::size_t mxm)
{
    static constexpr std::array<std::uint8_t, 4> queues {108, 136, 109, 137};
    return queues.at(mxm);
}

class RtlSchedule {
public:
    explicit RtlSchedule(TspSliceSystem& system) : system_(system) {}

    void mem_at(std::size_t queue, std::size_t cycle, MemInstruction instruction)
    {
        advance(queue, cycle, [&](std::size_t gap) {
            system_.icu().enqueue_mem_nop(queue, gap);
        });
        system_.icu().enqueue_mem(queue, instruction);
        records_.push_back(instruction_record(
            static_cast<std::uint8_t>(queue),
            isa::encode_mem_instruction(instruction), 47));
    }

    void mxm_load_at(
        std::size_t mxm,
        std::size_t cycle,
        MxmControlInstruction instruction)
    {
        const auto queue = mxm_load_queue(mxm);
        advance(queue, cycle, [&](std::size_t gap) {
            system_.icu().enqueue_mxm_load_nop(mxm, gap);
        });
        system_.icu().enqueue_mxm(mxm, instruction);
        records_.push_back(instruction_record(
            queue, isa::encode_mxm_instruction(instruction), 48));
    }

    void mxm_dequant_at(
        std::size_t mxm,
        std::size_t cycle,
        MxmDequantInstruction instruction)
    {
        const auto queue = mxm_dequant_queue(mxm);
        advance(queue, cycle, [&](std::size_t gap) {
            system_.icu().enqueue_mxm_dequant_nop(mxm, gap);
        });
        system_.icu().enqueue_mxm_dequant(mxm, instruction);
        records_.push_back(instruction_record(
            queue, isa::encode_mxm_dequant_instruction(instruction), 16));
    }

    void mxm_compute_at(
        std::size_t mxm,
        std::size_t cycle,
        MxmControlInstruction instruction)
    {
        const auto queue = mxm_compute_queue(mxm);
        advance(queue, cycle, [&](std::size_t gap) {
            system_.icu().enqueue_mxm_compute_nop(mxm, gap);
        });
        system_.icu().enqueue_mxm(mxm, instruction);
        records_.push_back(instruction_record(
            queue, isa::encode_mxm_instruction(instruction), 48));
    }

    void vxm_at(
        std::size_t alu,
        std::size_t cycle,
        VxmLaneAluInstruction instruction)
    {
        const auto queue = static_cast<std::uint8_t>(112 + alu);
        advance(queue, cycle, [&](std::size_t gap) {
            system_.icu().enqueue_vxm_nop(alu, gap);
        });
        system_.icu().enqueue_vxm(alu, instruction);
        records_.push_back(vxm_record(
            queue, isa::encode_vxm_instruction(instruction)));
    }

    void write(const char* path) const
    {
        auto output = std::ofstream(path, std::ios::trunc);
        if (!output) throw std::runtime_error("cannot open RTL schedule output");
        output << std::hex << std::setfill('0');
        for (const auto& record : records_) {
            for (std::size_t word = record.size(); word-- > 0;)
                output << std::setw(8) << record[word];
            output << '\n';
        }
    }

    std::size_t size() const { return records_.size(); }

private:
    template <typename EnqueueNop>
    void advance(std::size_t queue, std::size_t cycle, EnqueueNop enqueue_nop)
    {
        if (cycle < cursors_[queue])
            throw std::logic_error("RTL schedule queue overlap");
        const auto gap = cycle - cursors_[queue];
        if (gap != 0) {
            enqueue_nop(gap);
            records_.push_back(command_record(
                static_cast<std::uint8_t>(queue), isa::encode_icu_nop(gap)));
        }
        cursors_[queue] = cycle + 1;
    }

    TspSliceSystem& system_;
    std::array<std::size_t, 138> cursors_ {};
    std::vector<ScheduleRecord> records_;
};

} // namespace ftlpu::vmodel_test
