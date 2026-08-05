#pragma once

#include "ftlpu/core/instruction_packet.hpp"
#include "ftlpu/dma/descriptor.hpp"
#include "ftlpu/icu/location.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ftlpu {

struct ProgramImageHeader {
    static constexpr std::uint32_t kMagic = 0x55504c46u; // "FLPU", little endian
    std::uint32_t magic{kMagic};
    std::uint16_t format_version{1};
    std::string workload{};
    std::string metadata{};
};

// Host-side compiler output for one distributed local i-MEM bank.
struct ProgramSection {
    IcuLocation target{};
    std::vector<isa::EncodedInstructionPacket> packets{};
    std::optional<std::size_t> entry_packet{};
    std::string metadata{};

    void validate() const
    {
        if (packets.empty()) {
            throw std::invalid_argument("program section must contain at least one packet");
        }
        if (entry_packet.has_value() && *entry_packet >= packets.size()) {
            throw std::out_of_range("program section entry packet is outside the section");
        }
    }
};

struct ProgramDataSection {
    std::string name{};
    DmaPurpose purpose{DmaPurpose::General};
    MemGlobalAddress24 memory_address{};
    std::vector<std::uint8_t> bytes{};
    std::vector<std::size_t> shape{};
    std::string metadata{};

    void validate() const
    {
        if (name.empty()) {
            throw std::invalid_argument("program data section must have a name");
        }
        if (bytes.empty()
            || bytes.size() % hw::kPhysicalVectorBytes != 0) {
            throw std::invalid_argument(
                "program data section must contain whole stream vectors");
        }
        if (!memory_address.slice_byte_address().word_aligned()) {
            throw std::invalid_argument(
                "program data section destination must be vector-word aligned");
        }
    }
};

struct ProgramEntryPoint {
    IcuLocation target{};
    std::size_t section_index{0};
    std::size_t imem_base{0};
    std::size_t start_cycle{0};
};

class ProgramImage {
public:
    ProgramImage() = default;

    explicit ProgramImage(ProgramImageHeader header)
        : header_(std::move(header))
    {
        if (header_.magic != ProgramImageHeader::kMagic
            || header_.format_version == 0) {
            throw std::invalid_argument("invalid ProgramImage header");
        }
    }

    ProgramSection& add_section(ProgramSection section)
    {
        section.validate();
        sections_.push_back(std::move(section));
        return sections_.back();
    }

    ProgramDataSection& add_data_section(ProgramDataSection section)
    {
        section.validate();
        data_sections_.push_back(std::move(section));
        return data_sections_.back();
    }

    ProgramEntryPoint& add_entry_point(ProgramEntryPoint entry)
    {
        if (entry.section_index >= sections_.size()) {
            throw std::out_of_range(
                "ProgramImage entry point references an unknown section");
        }
        if (sections_[entry.section_index].target != entry.target) {
            throw std::invalid_argument(
                "ProgramImage entry target does not match its program section");
        }
        entry_points_.push_back(std::move(entry));
        return entry_points_.back();
    }

    const ProgramImageHeader& header() const noexcept { return header_; }

    const std::vector<ProgramSection>& sections() const noexcept
    {
        return sections_;
    }

    const std::vector<ProgramDataSection>& data_sections() const noexcept
    {
        return data_sections_;
    }

    const std::vector<ProgramEntryPoint>& entry_points() const noexcept
    {
        return entry_points_;
    }

    bool empty() const noexcept
    {
        return sections_.empty() && data_sections_.empty();
    }

private:
    ProgramImageHeader header_{};
    std::vector<ProgramSection> sections_{};
    std::vector<ProgramDataSection> data_sections_{};
    std::vector<ProgramEntryPoint> entry_points_{};
};

} // namespace ftlpu
