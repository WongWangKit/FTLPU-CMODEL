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

struct ProgramDataPlacement {
    std::size_t section_index{0};
    std::size_t host_offset_bytes{0};
    MemGlobalAddress24 memory_address{};
    std::size_t vector_count{0};
    DmaPurpose purpose{DmaPurpose::General};
};

class ProgramSramLayout {
public:
    static ProgramSramLayout Build(
        const ProgramImage& image,
        MemGlobalAddress24 base_address)
    {
        if (image.sections().empty()) {
            throw std::invalid_argument(
                "program SRAM layout needs at least one instruction section");
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

        append_data_sections(result, image);
        validate_non_overlapping_device_regions(result);
        return result;
    }

    // Places every instruction section at an explicit SRAM base. This is used
    // for autonomous MEM ICUs because their reset fetch can address only their
    // own local SRAM slice. Blocks within one section remain contiguous.
    static ProgramSramLayout BuildAtSectionBases(
        const ProgramImage& image,
        const std::vector<MemGlobalAddress24>& section_bases)
    {
        if (image.sections().empty()) {
            throw std::invalid_argument(
                "program SRAM layout needs at least one instruction section");
        }
        if (section_bases.size() != image.sections().size()) {
            throw std::invalid_argument(
                "explicit program layout needs one base per instruction section");
        }

        ProgramSramLayout result;
        for (std::size_t section_index = 0;
             section_index < image.sections().size();
             ++section_index) {
            const auto& section = image.sections()[section_index];
            const auto base = section_bases[section_index];
            section.validate();
            if (!base.slice_byte_address().word_aligned()) {
                throw std::invalid_argument(
                    "program section SRAM base must be vector-word aligned");
            }
            auto next = std::optional<MemLocalWordAddress13> {
                base.slice_byte_address().local_word_address()};
            const auto block_count =
                (section.packets.size() + hw::kIcuFetchPackets - 1)
                / hw::kIcuFetchPackets;
            for (std::size_t block = 0; block < block_count; ++block) {
                if (!next.has_value()) {
                    throw std::out_of_range(
                        "program section exceeds its selected MEM slice capacity");
                }
                const auto local = align_block_inside_bank(*next);
                const auto address = MemGlobalAddress24::FromFields(
                    base.hemisphere(), base.mem_slice(), local.slice_byte_address());
                const auto host_offset = result.host_bytes_.size();
                append_encoded_block(result.host_bytes_, section, block);
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
        append_data_sections(result, image);
        validate_non_overlapping_device_regions(result);
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

    const std::vector<ProgramDataPlacement>& data_placements() const noexcept
    {
        return data_placements_;
    }

    std::vector<DmaDescriptor> make_dma_descriptors(HostBufferId buffer) const
    {
        if (!buffer.valid()) {
            throw std::invalid_argument("program DMA descriptors need a valid Host buffer");
        }
        std::vector<DmaDescriptor> descriptors;
        descriptors.reserve(placements_.size() + data_placements_.size());
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
        for (const auto& placement : data_placements_) {
            descriptors.push_back(DmaDescriptor {
                DmaDirection::HostToMemory,
                placement.purpose,
                buffer,
                placement.host_offset_bytes,
                placement.memory_address,
                placement.vector_count,
            });
        }
        return descriptors;
    }

private:
    static void append_encoded_block(
        std::vector<std::uint8_t>& host_bytes,
        const ProgramSection& section,
        std::size_t block)
    {
        for (std::size_t packet = 0; packet < hw::kIcuFetchPackets; ++packet) {
            const auto section_packet =
                block * hw::kIcuFetchPackets + packet;
            const auto encoded = section_packet < section.packets.size()
                ? section.packets[section_packet]
                : program::padding_nop_packet();
            host_bytes.insert(
                host_bytes.end(), encoded.bytes.begin(), encoded.bytes.end());
        }
    }

    static void append_data_sections(
        ProgramSramLayout& result,
        const ProgramImage& image)
    {
        for (std::size_t section_index = 0;
             section_index < image.data_sections().size();
             ++section_index) {
            const auto& data = image.data_sections()[section_index];
            data.validate();
            const auto host_offset = result.host_bytes_.size();
            result.host_bytes_.insert(
                result.host_bytes_.end(), data.bytes.begin(), data.bytes.end());
            result.data_placements_.push_back(ProgramDataPlacement {
                section_index,
                host_offset,
                data.memory_address,
                data.bytes.size() / hw::kPhysicalVectorBytes,
                data.purpose,
            });
        }
    }

    static void validate_non_overlapping_device_regions(
        const ProgramSramLayout& layout)
    {
        struct Region {
            std::size_t hemisphere{0};
            std::size_t mem_slice{0};
            std::size_t first_linear_word{0};
            std::size_t end_linear_word{0};
        };
        std::vector<Region> regions{};
        const auto add_region = [&](MemGlobalAddress24 address,
                                    std::size_t vector_count) {
            if (vector_count == 0) {
                throw std::logic_error(
                    "ProgramImage device region must not be empty");
            }
            const auto local =
                address.slice_byte_address().local_word_address();
            (void)local.advance_words(vector_count - 1);
            const auto first =
                local.bank() * hw::kSramWordsPerBank + local.word();
            const auto end = first + vector_count;
            for (const auto& region : regions) {
                if (region.hemisphere == address.hemisphere()
                    && region.mem_slice == address.mem_slice()
                    && first < region.end_linear_word
                    && region.first_linear_word < end) {
                    throw std::invalid_argument(
                        "ProgramImage instruction/data SRAM regions overlap");
                }
            }
            regions.push_back(Region {
                address.hemisphere(),
                address.mem_slice(),
                first,
                end,
            });
        };
        for (const auto& block : layout.placements_) {
            add_region(block.memory_address, hw::kIcuFetchVectorCount);
        }
        for (const auto& data : layout.data_placements_) {
            add_region(data.memory_address, data.vector_count);
        }
    }

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
    std::vector<ProgramDataPlacement> data_placements_{};
};

} // namespace ftlpu
