#include "ftlpu/dma/dma.hpp"

#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

template <typename Fn>
bool throws(Fn&& fn)
{
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

std::vector<std::uint8_t> make_host_vectors(std::size_t vector_count)
{
    std::vector<std::uint8_t> bytes(
        vector_count * ftlpu::hw::kPhysicalVectorBytes);
    for (std::size_t vector = 0; vector < vector_count; ++vector) {
        for (std::size_t byte = 0;
             byte < ftlpu::hw::kPhysicalVectorBytes;
             ++byte) {
            bytes[vector * ftlpu::hw::kPhysicalVectorBytes + byte]
                = static_cast<std::uint8_t>((vector * 97 + byte) & 0xff);
        }
    }
    return bytes;
}

ftlpu::StreamPayloadVector320 expected_vector(
    const std::vector<std::uint8_t>& host,
    std::size_t vector_index)
{
    ftlpu::StreamPayloadVector320 result{};
    const auto base = vector_index * ftlpu::hw::kPhysicalVectorBytes;
    for (std::size_t tile = 0; tile < ftlpu::hw::kTileRows; ++tile) {
        for (std::size_t lane = 0; lane < ftlpu::hw::kLanesPerTile; ++lane) {
            result[tile][lane] =
                host[base + tile * ftlpu::hw::kLanesPerTile + lane];
        }
    }
    return result;
}

} // namespace

int main()
{
    constexpr std::size_t kHemisphere = 1;
    constexpr std::size_t kMemSlice = 7;
    constexpr auto kBaseWord =
        ftlpu::MemLocalWordAddress13::FromFields(1, 100);
    const auto base_address = ftlpu::MemGlobalAddress24::FromFields(
        kHemisphere, kMemSlice, kBaseWord.slice_byte_address());

    auto mem = std::make_unique<ftlpu::MemArrayModel>();
    ftlpu::GlobalMemoryAddressSpace global_memory;
    global_memory.bind_hemisphere(kHemisphere, *mem);

    ftlpu::HostMemorySpace host_memory;
    const auto source_bytes = make_host_vectors(2);
    const auto source = host_memory.register_buffer(source_bytes);
    const auto output = host_memory.allocate_buffer(ftlpu::hw::kPhysicalVectorBytes);

    ftlpu::DmaEngine dma(host_memory, global_memory);
    dma.enqueue(ftlpu::DmaDescriptor {
        ftlpu::DmaTransferId(17),
        ftlpu::DmaDirection::HostToMemory,
        ftlpu::DmaPurpose::InputTensor,
        source,
        0,
        base_address,
        2,
    });

    assert(dma.tick());
    assert(dma.cycle() == 1);
    assert(!dma.idle());
    assert(mem->read_sram_vector(kMemSlice, kBaseWord)
        == expected_vector(source_bytes, 0));
    assert(mem->read_sram_vector(
        kMemSlice,
        ftlpu::MemLocalWordAddress13::FromFields(0, 100))
        == ftlpu::StreamPayloadVector320 {});
    assert(mem->read_sram_vector(kMemSlice, kBaseWord.next_word())
        == ftlpu::StreamPayloadVector320 {});
    assert(dma.last_beat().has_value());
    assert(dma.last_beat()->vector_index == 0);
    assert(dma.last_beat()->memory_address == base_address);

    assert(dma.tick());
    assert(dma.cycle() == 2);
    assert(dma.idle());
    assert(mem->read_sram_vector(kMemSlice, kBaseWord.next_word())
        == expected_vector(source_bytes, 1));
    assert(dma.completions().size() == 1);
    assert(dma.completions().front().id == ftlpu::DmaTransferId(17));
    assert(dma.completions().front().vector_count == 2);

    // Exercise the reverse path reserved for future output tensors.
    const auto second_address = ftlpu::MemGlobalAddress24::FromFields(
        kHemisphere,
        kMemSlice,
        kBaseWord.next_word().slice_byte_address());
    dma.enqueue(ftlpu::DmaDescriptor {
        ftlpu::DmaTransferId(18),
        ftlpu::DmaDirection::MemoryToHost,
        ftlpu::DmaPurpose::OutputTensor,
        output,
        0,
        second_address,
        1,
    });
    assert(dma.tick());
    assert(host_memory.buffer(output)
        == std::vector<std::uint8_t>(
            source_bytes.begin() + ftlpu::hw::kPhysicalVectorBytes,
            source_bytes.end()));

    assert(!dma.tick());
    assert(dma.cycle() == 4);

    assert(throws([&] {
        dma.enqueue(ftlpu::DmaDescriptor {
            ftlpu::DmaTransferId(19),
            ftlpu::DmaDirection::HostToMemory,
            ftlpu::DmaPurpose::Model,
            source,
            0,
            ftlpu::MemGlobalAddress24::FromFields(
                kHemisphere,
                kMemSlice,
                ftlpu::MemLocalWordAddress13::FromFields(0, 4095)
                    .slice_byte_address()),
            2,
        });
    }));

    assert(throws([&] {
        dma.enqueue(ftlpu::DmaDescriptor {
            ftlpu::DmaTransferId(20),
            ftlpu::DmaDirection::HostToMemory,
            ftlpu::DmaPurpose::General,
            source,
            0,
            ftlpu::MemGlobalAddress24::FromFields(
                kHemisphere,
                kMemSlice,
                ftlpu::MemSliceByteAddress17::FromFields(1, 100, 1)),
            1,
        });
    }));

    return 0;
}
