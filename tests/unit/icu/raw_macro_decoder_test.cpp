#include "ftlpu/icu/distributed_queue.hpp"
#include "ftlpu/system/icu.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

template <typename Queue>
std::size_t prime(Queue& queue)
{
    std::size_t cycles = 0;
    while (!queue.raw_macro_ready()) {
        queue.prefetch_raw_macro();
        if (++cycles > 64)
            throw std::runtime_error("raw Macro frontend did not become ready");
    }
    return cycles;
}

const std::vector<ftlpu::IcuRawImemWord96> kMemImage{
    {{{0x00000027u, 0x00000000u, 0x00000000u}}},
    {{{0x02000021u, 0x00000180u, 0x00303200u}}},
    {{{0x00000400u, 0x00000000u, 0x00000000u}}},
};

const std::vector<ftlpu::IcuRawImemWord128> kMxmLoadImage{
    {{{0x00000017u, 0x00000000u, 0x00000000u, 0x00000000u}}},
    {{{0x00000028u, 0x04000008u, 0x00000000u, 0x00000000u}}},
};

const std::vector<ftlpu::IcuRawImemWord128> kMxmComputeImage{
    {{{0x00000017u, 0x00000000u, 0x00000000u, 0x00000000u}}},
    {{{0x00000030u, 0x00040320u, 0x40004000u, 0x00000060u}}},
};

const std::vector<ftlpu::IcuRawImemWord128> kMxmDequantImage{
    {{{0x00000017u, 0x00000000u, 0x00000000u, 0x00000000u}}},
    {{{0x00000010u, 0x0007f000u, 0x00000000u, 0x00000000u}}},
};

const std::vector<ftlpu::IcuRawImemWord96> kMemExtendedImage{
    {{{0x00000017u, 0x00000000u, 0x00000000u}}},
    {{{0x000000a0u, 0x00627ff8u, 0x40041ffeu}}},
    {{{0x000c0000u, 0xfff80000u, 0x000bffffu}}},
    {{{0x38800000u, 0x01000001u, 0x00040000u}}},
};

const std::vector<ftlpu::IcuRawImemWord96> kMemCompactEscapeImage{
    {{{0x00000027u, 0x00000000u, 0x00000000u}}},
    {{{0x00000008u, 0x00030050u, 0x0011f020u}}},
    {{{0x00000100u, 0x00000000u, 0x00000000u}}},
};

const std::vector<ftlpu::IcuRawImemWord96> kMemWideEscapeImage{
    {{{0x00000027u, 0x00000000u, 0x00000000u}}},
    {{{0x00000008u, 0x00030050u, 0x2cfff020u}}},
    {{{0x00040131u, 0x00000000u, 0x00000000u}}},
};

std::vector<ftlpu::DecodedIcuMacroContext<ftlpu::MemInstruction>>
decode_mem_image(const std::vector<ftlpu::IcuRawImemWord96>& image)
{
    ftlpu::IcuMacroV1Decoder<ftlpu::MemInstruction, 96> decoder(
        ftlpu::IcuMacroQueueKind::Mem, image);
    std::vector<ftlpu::DecodedIcuMacroContext<ftlpu::MemInstruction>> result;
    for (std::size_t cycle = 0; !decoder.done(); ++cycle) {
        if (cycle > 64)
            throw std::runtime_error("raw MEM image did not finish decoding");
        if (auto context = decoder.tick(true))
            result.push_back(std::move(*context));
    }
    return result;
}

} // namespace

int main()
try {
    using namespace ftlpu;

    const auto extended = decode_mem_image(kMemExtendedImage);
    require(extended.size() == 1,
        "extended MEM image produced the wrong context count");
    require(extended[0].instruction.opcode == MemOpcode::Read
            && extended[0].instruction.address == 4095
            && extended[0].instruction.stream == 3,
        "extended MEM template reconstructed the wrong instruction");
    require(extended[0].schedule.start_cycle == 20
            && extended[0].schedule.inner_count == 4097
            && extended[0].schedule.inner_interval == 3
            && extended[0].schedule.inner_stride == -2
            && extended[0].schedule.outer_count == 2
            && extended[0].schedule.outer_interval == 20000
            && extended[0].schedule.outer_stride == 64,
        "extended MEM template reconstructed the wrong schedule");

    const auto compactEscape = decode_mem_image(kMemCompactEscapeImage);
    require(compactEscape.size() == 2
            && compactEscape[0].schedule.start_cycle == 1
            && compactEscape[1].schedule.start_cycle == 5
            && compactEscape[0].instruction.address == 10
            && compactEscape[1].instruction.address == 11,
        "compact delta escape decoded incorrectly");

    const auto wideEscape = decode_mem_image(kMemWideEscapeImage);
    require(wideEscape.size() == 2
            && wideEscape[0].schedule.start_cycle == 1
            && wideEscape[1].schedule.start_cycle == 5000000
            && wideEscape[0].instruction.address == 10
            && wideEscape[1].instruction.address == 11,
        "wide delta escape decoded incorrectly");

    require(kMemImage[0].bit(0) && kMemImage[0].bit(1)
            && kMemImage[0].bit(2) && !kMemImage[0].bit(3)
            && !kMemImage[0].bit(4) && kMemImage[0].bit(5),
        "raw word does not map lane-0 bit 0 to the first stream bit");
    auto invalidControlImage = kMemImage;
    invalidControlImage[0].lanes[0] |= 0x10000000u;
    bool rejectedReservedControl = false;
    try {
        IcuMacroV1Decoder<MemInstruction, 96> invalid(
            IcuMacroQueueKind::Mem, invalidControlImage);
        for (std::size_t cycle = 0; cycle < 3; ++cycle)
            static_cast<void>(invalid.tick(true));
    } catch (const StaticScheduleError&) {
        rejectedReservedControl = true;
    }
    require(rejectedReservedControl,
        "raw Macro decoder accepted reserved control-word bits");

    // The MEM image is a hard-coded physical golden vector. It contains two
    // singleton reads with one dictionary transition and crosses a 96-bit
    // payload-word boundary:
    //   cycle 3: Read(address=100, stream=2)
    //   cycle 7: Read(address=101, stream=2)
    using ConstrainedMemQueue =
        DistributedIcuQueue<MemInstruction, 96, 16, 4, 1, 1>;
    ConstrainedMemQueue constrained;
    constrained.load_raw_macro(IcuMacroQueueKind::Mem, kMemImage);
    const auto constrainedPrimeCycles = prime(constrained);
    require(constrainedPrimeCycles >= kMemImage.size(),
        "raw MEM decoder fetched with infinite bandwidth");

    std::vector<std::pair<std::size_t, std::size_t>> memIssues;
    for (std::size_t cycle = 0; cycle <= 7; ++cycle) {
        if (const auto instruction = constrained.tick())
            memIssues.emplace_back(cycle, instruction->address);
    }
    require(memIssues
            == std::vector<std::pair<std::size_t, std::size_t>>{
                {3, 100}, {7, 101}},
        "raw MEM Macro image decoded or issued incorrectly");
    require(constrained.done(), "raw MEM Macro queue did not complete");
    require(constrained.macro_decoder_statistics().fetched_words == 3,
        "raw MEM decoder fetched the wrong number of physical words");
    require(constrained.macro_decoder_statistics().decoded_contexts == 2,
        "raw MEM decoder produced the wrong number of contexts");
    require(constrained.macro_decoder_statistics().context_stall_cycles >= 2,
        "raw MEM admission did not model context-full backpressure");
    require(constrained.macro_decoder_statistics().ddb_entries_committed != 0
            && constrained.macro_decoder_statistics().descriptor_count != 0,
        "raw MEM frontend bypassed the decoded descriptor buffer");
    require(constrained.macro_decoder_statistics().payload_bits_consumed != 0
            && constrained.macro_decoder_statistics().decoder_active_cycles
                != 0,
        "raw MEM frontend did not expose cycle-accurate parser work");
    require(constrained.peak_active_macros() == 1,
        "raw MEM decoder exceeded the finite context RAM");

    InstructionControlUnit icu;
    for (std::size_t address = 0; address < kMemImage.size(); ++address)
        icu.write_mem_raw_macro_imem(0, address, kMemImage[address]);
    icu.configure_mem_raw_macro_imem(0, kMemImage.size());

    for (std::size_t address = 0; address < kMxmLoadImage.size(); ++address)
        icu.write_mxm_load_raw_macro_imem(
            0, address, kMxmLoadImage[address]);
    icu.configure_mxm_load_raw_macro_imem(0, kMxmLoadImage.size());

    for (std::size_t address = 0; address < kMxmComputeImage.size(); ++address)
        icu.write_mxm_compute_raw_macro_imem(
            0, address, kMxmComputeImage[address]);
    icu.configure_mxm_compute_raw_macro_imem(0, kMxmComputeImage.size());

    for (std::size_t address = 0; address < kMxmDequantImage.size(); ++address)
        icu.write_mxm_dequant_raw_macro_imem(
            0, address, kMxmDequantImage[address]);
    icu.configure_mxm_dequant_raw_macro_imem(0, kMxmDequantImage.size());

    const auto primeCycles = icu.prime_raw_macro_frontends();
    require(primeCycles >= 3,
        "system raw Macro frontend did not model fixed-width fetch cycles");

    std::vector<std::pair<std::size_t, std::size_t>> systemMemIssues;
    std::optional<MxmControlInstruction> loadIssue;
    std::optional<MxmControlInstruction> computeIssue;
    std::optional<MxmDequantInstruction> dequantIssue;
    for (std::size_t cycle = 0; cycle <= 7; ++cycle) {
        if (const auto instruction = icu.mem_iq(0).tick())
            systemMemIssues.emplace_back(cycle, instruction->address);
        if (const auto instruction = icu.mxm_load_iq(0).tick()) {
            require(cycle == 5, "raw MXM load issued on the wrong cycle");
            loadIssue = instruction;
        }
        if (const auto instruction = icu.mxm_compute_iq(0).tick()) {
            require(cycle == 6, "raw MXM compute issued on the wrong cycle");
            computeIssue = instruction;
        }
        if (const auto instruction = icu.mxm_dequant_iq(0).tick()) {
            require(cycle == 2, "raw MXM dequant issued on the wrong cycle");
            dequantIssue = instruction;
        }
    }

    require(systemMemIssues
            == std::vector<std::pair<std::size_t, std::size_t>>{
                {3, 100}, {7, 101}},
        "system raw MEM write API changed the decoded schedule");
    require(loadIssue.has_value()
            && loadIssue->opcode == MxmControlOpcode::IW
            && loadIssue->weight_column == 1,
        "raw MXM load decoder reconstructed the wrong instruction");
    require(computeIssue.has_value()
            && computeIssue->opcode == MxmControlOpcode::Compute
            && computeIssue->accumulator_address == 100,
        "raw MXM compute decoder reconstructed the wrong instruction");
    require(dequantIssue.has_value()
            && dequantIssue->scale_bf16 == 0x3f80,
        "raw MXM dequant decoder reconstructed the wrong instruction");

    const auto statistics = icu.frontend_statistics();
    require(statistics.imem_entries == 9,
        "raw Macro statistics did not count physical i-MEM words");
    require(statistics.fetched_entries == 9,
        "raw Macro statistics reported the wrong fetch count");
    require(statistics.decoded_macro_contexts == 5,
        "raw Macro statistics reported the wrong decoded-context count");
    require(statistics.peak_macro_reservoir_bits != 0,
        "raw Macro statistics did not expose reservoir occupancy");

    std::cout << "icu_raw_macro_decoder_test passed: prime_cycles="
              << primeCycles
              << " mem_context_stalls="
              << constrained.macro_decoder_statistics().context_stall_cycles
              << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << "icu_raw_macro_decoder_test failed: "
              << error.what() << '\n';
    return 1;
}
