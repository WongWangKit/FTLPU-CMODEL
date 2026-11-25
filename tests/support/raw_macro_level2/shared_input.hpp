#pragma once

#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/mem/slice.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ftlpu::test::raw_macro_level2 {

struct OperationSpec {
    std::size_t issue_cycle{};
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

inline std::vector<CaseSpec> cases()
{
    const auto read = [](std::size_t cycle, std::size_t address,
                          StreamId stream) {
        return OperationSpec {cycle, 0, 0,
            MemInstruction::Read(address, stream)};
    };
    const auto write = [](std::size_t cycle, std::size_t address,
                           StreamId stream) {
        return OperationSpec {cycle, 0, 0,
            MemInstruction::Write(address, stream)};
    };

    return {
        {"mem_read_single", {
            read(0, 32, StreamId::East(1)),
        }},
        {"mem_write_single", {
            write(0, 48, StreamId::East(2)),
        }},
        {"mem_read_consecutive", {
            read(0, 64, StreamId::East(3)),
            read(1, 65, StreamId::East(3)),
            read(2, 66, StreamId::East(3)),
        }},
        {"mem_write_consecutive", {
            write(0, 80, StreamId::East(4)),
            write(1, 81, StreamId::East(4)),
            write(2, 82, StreamId::East(4)),
        }},
        {"mem_read_write_mixed", {
            read(0, 96, StreamId::East(5)),
            write(1, 97, StreamId::East(6)),
            read(2, 98, StreamId::East(7)),
            write(3, 99, StreamId::East(8)),
        }},
    };
}

inline std::uint8_t read_preload_byte(
    std::size_t address, std::size_t tile, std::size_t lane)
{
    return static_cast<std::uint8_t>(
        (address * 3 + tile * 17 + lane * 5 + 0x21) & 0xffu);
}

inline std::uint8_t write_stimulus_byte(
    std::size_t case_index,
    std::size_t operation_index,
    std::size_t tile,
    std::size_t lane)
{
    return static_cast<std::uint8_t>(
        (0x80 + case_index * 19 + operation_index * 23
            + tile * 11 + lane * 3) & 0xffu);
}

} // namespace ftlpu::test::raw_macro_level2
