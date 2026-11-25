#pragma once

#include "ftlpu/core/hemisphere.hpp"
#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/mem/slice.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ftlpu::test::raw_macro_phase1_space {

struct OperationSpec {
    std::size_t issue_cycle{};
    Hemisphere hemisphere{Hemisphere::East};
    std::size_t mem_slice{};
    std::size_t bank{};
    MemInstruction instruction{};

    isa::EncodedMemInstruction machine_word() const
    {
        return isa::encode_mem_instruction(instruction);
    }
};

struct CaseSpec {
    std::string id{};
    std::vector<OperationSpec> operations{};
};

struct ProfileSpec {
    std::string id{};
    std::size_t rtl_active_mem_slices{};
    std::size_t rtl_physical_mem_slices{};
    std::size_t rtl_active_rows{};
    std::vector<CaseSpec> cases{};
};

inline OperationSpec read(std::size_t cycle, Hemisphere hemisphere,
    std::size_t mem_slice, std::size_t bank, std::size_t address,
    StreamId stream)
{
    return {cycle, hemisphere, mem_slice, bank,
        MemInstruction::Read(address, stream)};
}

inline OperationSpec write(std::size_t cycle, Hemisphere hemisphere,
    std::size_t mem_slice, std::size_t bank, std::size_t address,
    StreamId stream)
{
    return {cycle, hemisphere, mem_slice, bank,
        MemInstruction::Write(address, stream)};
}

inline std::vector<CaseSpec> tile_baseline_cases()
{
    const auto legacy_read = [](std::size_t cycle, std::size_t address,
                                 StreamId stream) {
        return read(cycle, Hemisphere::East, 0, 0, address, stream);
    };
    const auto legacy_write = [](std::size_t cycle, std::size_t address,
                                  StreamId stream) {
        return write(cycle, Hemisphere::East, 0, 0, address, stream);
    };
    return {
        {"mem_read_single", {legacy_read(0, 32, StreamId::East(1))}},
        {"mem_write_single", {legacy_write(0, 48, StreamId::East(2))}},
        {"mem_read_consecutive", {
            legacy_read(0, 64, StreamId::East(3)),
            legacy_read(1, 65, StreamId::East(3)),
            legacy_read(2, 66, StreamId::East(3)),
        }},
        {"mem_write_consecutive", {
            legacy_write(0, 80, StreamId::East(4)),
            legacy_write(1, 81, StreamId::East(4)),
            legacy_write(2, 82, StreamId::East(4)),
        }},
        {"mem_read_write_mixed", {
            legacy_read(0, 96, StreamId::East(5)),
            legacy_write(1, 97, StreamId::East(6)),
            legacy_read(2, 98, StreamId::East(7)),
            legacy_write(3, 99, StreamId::East(8)),
        }},
    };
}

inline std::vector<CaseSpec> bank_cases()
{
    return {
        {"bank_read_write_parallel", {
            read(0, Hemisphere::East, 0, 0, 16, StreamId::East(1)),
            write(0, Hemisphere::East, 0, 1, 17, StreamId::East(2)),
        }},
        {"bank_dual_read", {
            read(0, Hemisphere::East, 0, 0, 20, StreamId::East(3)),
            read(0, Hemisphere::East, 0, 1, 21, StreamId::East(4)),
        }},
        {"bank_dual_write", {
            write(0, Hemisphere::East, 0, 0, 22, StreamId::East(5)),
            write(0, Hemisphere::East, 0, 1, 23, StreamId::East(6)),
        }},
    };
}

inline std::vector<CaseSpec> slice_cases()
{
    return {
        {"slice_same_group_parallel", {
            read(0, Hemisphere::East, 0, 0, 30, StreamId::East(7)),
            read(0, Hemisphere::East, 3, 0, 31, StreamId::East(8)),
        }},
        {"slice_cross_group_parallel", {
            read(0, Hemisphere::East, 3, 0, 32, StreamId::East(9)),
            read(0, Hemisphere::East, 4, 0, 33, StreamId::East(10)),
        }},
    };
}

inline std::vector<CaseSpec> full_reduced_cases()
{
    std::vector<OperationSpec> sentinels;
    for (const auto hemisphere : {Hemisphere::East, Hemisphere::West}) {
        for (const auto mem_slice : {std::size_t{0}, std::size_t{25},
                 std::size_t{51}}) {
            for (std::size_t bank = 0; bank < 2; ++bank) {
                const auto local_bank = (mem_slice % 4) * 2 + bank;
                sentinels.push_back(read(0, hemisphere, mem_slice, bank,
                    48 + mem_slice + bank, StreamId::East(local_bank)));
            }
        }
    }
    return {
        {"dual_hemisphere_same_identity", {
            read(0, Hemisphere::East, 25, 1, 40, StreamId::East(11)),
            read(0, Hemisphere::West, 25, 1, 40, StreamId::East(11)),
        }},
        {"coordinate_sentinels", std::move(sentinels)},
        {"sparse_full_parallel", {
            read(0, Hemisphere::East, 0, 0, 41, StreamId::East(12)),
            write(0, Hemisphere::East, 0, 1, 42, StreamId::East(13)),
            read(0, Hemisphere::East, 51, 1, 43, StreamId::East(14)),
            read(0, Hemisphere::West, 0, 0, 41, StreamId::East(12)),
            write(0, Hemisphere::West, 51, 1, 42, StreamId::East(13)),
        }},
    };
}

inline std::vector<CaseSpec> queue_census_cases()
{
    std::vector<OperationSpec> operations;
    operations.reserve(hw::kHemispheres * hw::kMemSliceColumns
        * hw::kMemBanksPerSlice);
    for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
        const auto hemisphere = static_cast<Hemisphere>(side);
        for (std::size_t mem_slice = 0;
             mem_slice < hw::kMemSliceColumns; ++mem_slice) {
            for (std::size_t bank = 0;
                 bank < hw::kMemBanksPerSlice; ++bank) {
                const auto local_bank = (mem_slice % hw::kMemSlicesPerGroup)
                    * hw::kMemBanksPerSlice + bank;
                operations.push_back(read(local_bank, hemisphere, mem_slice,
                    bank, 32 + mem_slice / hw::kMemSlicesPerGroup,
                    StreamId::East(local_bank)));
            }
        }
    }
    return {{"all_208_bank_queue_census", std::move(operations)}};
}

inline std::vector<CaseSpec> full_formal_cases()
{
    std::vector<OperationSpec> operations;
    for (const auto hemisphere : {Hemisphere::East, Hemisphere::West}) {
        for (const auto mem_slice : {std::size_t{0}, std::size_t{51}}) {
            for (std::size_t bank = 0; bank < 2; ++bank) {
                const auto stream = StreamId::East(
                    (mem_slice % 4) * 2 + bank);
                operations.push_back(read(0, hemisphere, mem_slice, bank,
                    0, stream));
                operations.push_back(read(1, hemisphere, mem_slice, bank,
                    hw::kSramDepthRows - 1, stream));
            }
        }
    }
    return {{"depth_0_8191_sentinels", std::move(operations)}};
}

inline std::vector<ProfileSpec> profiles()
{
    return {
        {"tile_baseline", 1, 4, hw::kSramDepthRows,
            tile_baseline_cases()},
        {"bank_f4_d128", 4, 4, 128, bank_cases()},
        {"slice_f8_d128", 8, 8, 128, slice_cases()},
        {"full_reduced_f52_d128", 52, 52, 128, full_reduced_cases()},
        {"queue_census_f52_d128", 52, 52, 128, queue_census_cases()},
        {"full_formal_f52_d8192", 52, 52, hw::kSramDepthRows,
            full_formal_cases()},
    };
}

inline std::size_t queue_id(const OperationSpec& operation)
{
    return hemisphere_index(operation.hemisphere)
            * hw::kMemSliceColumns * hw::kMemBanksPerSlice
        + operation.mem_slice * hw::kMemBanksPerSlice + operation.bank;
}

inline std::uint8_t spatial_salt(const OperationSpec& operation)
{
    return static_cast<std::uint8_t>(
        hemisphere_index(operation.hemisphere) * 0x61
        + operation.mem_slice * 0x13 + operation.bank * 0x2b);
}

inline std::uint8_t read_preload_byte(
    const OperationSpec& operation, std::size_t tile, std::size_t lane)
{
    return static_cast<std::uint8_t>(
        operation.instruction.address * 3 + tile * 17 + lane * 5 + 0x21
        + spatial_salt(operation));
}

inline std::uint8_t write_stimulus_byte(std::size_t case_index,
    std::size_t operation_index, const OperationSpec& operation,
    std::size_t tile, std::size_t lane)
{
    return static_cast<std::uint8_t>(
        0x80 + case_index * 19 + operation_index * 23
        + tile * 11 + lane * 3 + spatial_salt(operation));
}

} // namespace ftlpu::test::raw_macro_phase1_space
