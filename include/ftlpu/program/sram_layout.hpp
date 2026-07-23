#pragma once

#include "ftlpu/core/hardware_params.hpp"
#include "ftlpu/dma/descriptor.hpp"
#include "ftlpu/program/packet_encoder.hpp"
#include "ftlpu/program/program_image.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ftlpu {

struct ProgramBlockPlacement {
    IcuLocation target{};
    std::size_t section_index{0};
    std::size_t block_index_in_section{0};
    std::size_t host_offset_bytes{0};
    MemGlobalAddress24 memory_address{};
};

class ProgramSramLayout {
public:
    static ProgramSramLayout Build(
        const ProgramImage& image,
        MemGlobalAddress24 base_address)
    {
        if (image.empty()) {
            throw std::invalid_argument("cannot lay out an empty program image");
        }
        if (!base_address.slice_byte_address().word_aligned()) {
            throw std::invalid_argument("program SRAM base must be vector-word aligned");
        }

        ProgramSramLayout result;
        auto next = std::optional<MemLocalWordAddress13> {
            base_address.slice_byte_address().local_word_address()};

        for (std::size_t section_index = 0;
             section_index < image.sections().size();
             ++section_index) {
            const auto& section = image.sections()[section_index];
            section.validate();
            const auto block_count =
                (section.packets.size() + hw::kIcuFetchPackets - 1)
                / hw::kIcuFetchPackets;
            for (std::size_t block = 0; block < block_count; ++block) {
                if (!next.has_value()) {
                    throw std::out_of_range("program image exceeds the selected MEM slice capacity");
                }
                auto local = align_block_inside_bank(*next);
                const auto address = MemGlobalAddress24::FromFields(
                    base_address.hemisphere(),
                    base_address.mem_slice(),
                    local.slice_byte_address());
                const auto host_offset = result.host_bytes_.size();

                for (std::size_t packet = 0; packet < hw::kIcuFetchPackets; ++packet) {
                    const auto section_packet = block * hw::kIcuFetchPackets + packet;
                    const auto encoded = section_packet < section.packets.size()
                        ? section.packets[section_packet]
                        : program::padding_nop_packet();
                    result.host_bytes_.insert(
                        result.host_bytes_.end(),
                        encoded.bytes.begin(),
                        encoded.bytes.end());
                }
                result.placements_.push_back(ProgramBlockPlacement {
                    section.target,
                    section_index,
                    block,
                    host_offset,
                    address,
                });
                next = next_block(local);
            }
        }
        return result;
    }

    const std::vector<std::uint8_t>& host_bytes() const noexcept
    {
        return host_bytes_;
    }

    const std::vector<ProgramBlockPlacement>& placements() const noexcept
    {
        return placements_;
    }

    std::vector<DmaDescriptor> make_dma_descriptors(HostBufferId buffer) const
    {
        if (!buffer.valid()) {
            throw std::invalid_argument("program DMA descriptors need a valid Host buffer");
        }
        std::vector<DmaDescriptor> descriptors;
        descriptors.reserve(placements_.size());
        for (const auto& placement : placements_) {
            const auto local = placement.memory_address.slice_byte_address()
                                   .local_word_address();
            if (local.word() + hw::kIcuFetchVectorCount
                > hw::kSramWordsPerBank) {
                throw std::logic_error("program DMA descriptor would cross an SRAM bank");
            }
            descriptors.push_back(DmaDescriptor {
                DmaDirection::HostToMemory,
                DmaPurpose::Program,
                buffer,
                placement.host_offset_bytes,
                placement.memory_address,
                hw::kIcuFetchVectorCount,
            });
        }
        return descriptors;
    }

private:
    static MemLocalWordAddress13 align_block_inside_bank(
        MemLocalWordAddress13 candidate)
    {
        if (candidate.word() + hw::kIcuFetchVectorCount
            <= hw::kSramWordsPerBank) {
            return candidate;
        }
        if (candidate.bank() + 1 >= hw::kSramBanksPerTileBlock) {
            throw std::out_of_range("program block cannot cross past the final SRAM bank");
        }
        return MemLocalWordAddress13::FromFields(candidate.bank() + 1, 0);
    }

    static std::optional<MemLocalWordAddress13> next_block(
        MemLocalWordAddress13 current)
    {
        const auto next_word = current.word() + hw::kIcuFetchVectorCount;
        if (next_word < hw::kSramWordsPerBank) {
            return MemLocalWordAddress13::FromFields(current.bank(), next_word);
        }
        if (next_word == hw::kSramWordsPerBank
            && current.bank() + 1 < hw::kSramBanksPerTileBlock) {
            return MemLocalWordAddress13::FromFields(current.bank() + 1, 0);
        }
        return std::nullopt;
    }

    std::vector<std::uint8_t> host_bytes_{};
    std::vector<ProgramBlockPlacement> placements_{};
};

} // namespace ftlpu
