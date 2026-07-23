#pragma once

#include "ftlpu/core/instruction_packet.hpp"
#include "ftlpu/icu/fetch.hpp"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ftlpu {

// Host-side program description.  It is not an ICU-private backing store;
// execution still reaches an ICU only through DMA, MEM Read and an SR stream.
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

class ProgramImage {
public:
    ProgramSection& add_section(ProgramSection section)
    {
        section.validate();
        sections_.push_back(std::move(section));
        return sections_.back();
    }

    const std::vector<ProgramSection>& sections() const noexcept
    {
        return sections_;
    }

    bool empty() const noexcept { return sections_.empty(); }

private:
    std::vector<ProgramSection> sections_{};
};

} // namespace ftlpu
