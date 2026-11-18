#include "ftlpu/icu/distributed_queue.hpp"
#include "ftlpu/system/icu.hpp"

#include <algorithm>
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
    {{{0x00000025u, 0x00000000u, 0x00000000u}}},
    {{{0x02000021u, 0x00000180u, 0x00303200u}}},
    {{{0x00000400u, 0x00000000u, 0x00000000u}}},
};

const std::vector<ftlpu::IcuRawImemWord128> kMxmLoadImage{
    {{{0x00000015u, 0x00000000u, 0x00000000u, 0x00000000u}}},
    {{{0x00000028u, 0x04000008u, 0x00000000u, 0x00000000u}}},
};

const std::vector<ftlpu::IcuRawImemWord128> kMxmComputeImage{
    {{{0x00000015u, 0x00000000u, 0x00000000u, 0x00000000u}}},
    {{{0x00000030u, 0x00040320u, 0x40004000u, 0x00000060u}}},
};

const std::vector<ftlpu::IcuRawImemWord128> kMxmDequantImage{
    {{{0x00000015u, 0x00000000u, 0x00000000u, 0x00000000u}}},
    {{{0x00000010u, 0x0007f000u, 0x00000000u, 0x00000000u}}},
};

const std::vector<ftlpu::IcuRawImemWord96> kMemExtendedImage{
    {{{0x00000015u, 0x00000000u, 0x00000000u}}},
    {{{0x000000a0u, 0x00627ff8u, 0x40041ffeu}}},
    {{{0x000c0000u, 0xfff80000u, 0x000bffffu}}},
    {{{0x38800000u, 0x01000001u, 0x00040000u}}},
};

const std::vector<ftlpu::IcuRawImemWord96> kMemCompactEscapeImage{
    {{{0x00000025u, 0x00000000u, 0x00000000u}}},
    {{{0x00000008u, 0x00030050u, 0x0011f020u}}},
    {{{0x00000100u, 0x00000000u, 0x00000000u}}},
};

const std::vector<ftlpu::IcuRawImemWord96> kMemWideEscapeImage{
    {{{0x00000025u, 0x00000000u, 0x00000000u}}},
    {{{0x00000008u, 0x00030050u, 0x2cfff020u}}},
    {{{0x00040131u, 0x00000000u, 0x00000000u}}},
};

std::vector<ftlpu::IcuRawImemWord96> make_mem_repeat_image()
{
    // Test-side assembler for one compact MEM Macro:
    // Read addresses 100, 101, 102, 103 at cycles 0, 1, 2, 3.
    std::vector<ftlpu::IcuRawImemWord96> result(1);
    result[0].lanes[0] = 5u | (1u << 4); // MacroPacked v1, one descriptor; bit 1 is reserved.
    std::size_t offset = 96;
    const auto bits = [&](std::uint64_t value, unsigned width) {
        for (unsigned bit = 0; bit < width; ++bit, ++offset) {
            if (result.size() <= offset / 96) result.emplace_back();
            result[offset / 96].lanes[(offset % 96) / 32]
                |= std::uint32_t((value >> bit) & 1u) << (offset % 32);
        }
    };
    bits(0, 3);   // No delta dictionary.
    bits(0, 32);  // Initial absolute start cycle.
    bits(100, 13);// Initial MEM address operand.
    bits(0, 1);   // One-template run.
    bits(0, 1);   // Compact template.
    bits(1, 2);   // Inner1D schedule.
    bits(0, 1);   // Read, not Write.
    bits(0, 6);   // East stream 0.
    bits(0, 1);   // Preserve is irrelevant for Read.
    bits(3, 11);  // inner_count = 4.
    bits(0, 8);   // inner_interval = 1.
    bits(1, 10);  // address stride = 1.
    return result;
}

void assign_mem_stream_bits(std::vector<ftlpu::IcuRawImemWord96>& image,
    std::size_t offset, std::uint64_t value, unsigned width)
{
    for (unsigned bit = 0; bit < width; ++bit) {
        const auto position = offset + bit;
        const auto mask = std::uint32_t{1} << (position % 32);
        auto& lane = image[position / 96].lanes[(position % 96) / 32];
        if (((value >> bit) & 1u) != 0)
            lane |= mask;
        else
            lane &= ~mask;
    }
}

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

template <typename Instruction, std::size_t WordBits>
ftlpu::IcuMacroDecoderStatistics decode_statistics(
    ftlpu::IcuMacroQueueKind kind,
    const std::vector<ftlpu::IcuRawImemWord<WordBits>>& image)
{
    ftlpu::IcuMacroV1Decoder<Instruction, WordBits> decoder(kind, image);
    for (std::size_t cycle = 0; !decoder.done(); ++cycle) {
        if (cycle > 128)
            throw std::runtime_error(
                "raw Macro image did not finish timing decode");
        static_cast<void>(decoder.tick(true));
    }
    return decoder.statistics();
}

} // namespace

int main()
try {
    using namespace ftlpu;
    // Shift the golden stream's initial absolute start from 3 to 0. Priming
    // must install its first context before the global issue clock starts.
    auto zeroStartImage = kMemImage;
    for (std::size_t bit = 0; bit < 32; ++bit) {
        const auto offset = 96 + 3 + 36 + bit;
        zeroStartImage[offset / 96].lanes[(offset % 96) / 32]
            &= ~(std::uint32_t{1} << (offset % 32));
    }
    InstructionControlUnit::MemIcu zeroStart;
    zeroStart.load_raw_macro(IcuMacroQueueKind::Mem, zeroStartImage);
    prime(zeroStart);
    require(zeroStart.tick().has_value(),
        "cold-start priming did not make cycle-zero Macro issue visible");

    InstructionControlUnit::MemIcu consecutiveMem;
    consecutiveMem.load_raw_macro(
        IcuMacroQueueKind::Mem, make_mem_repeat_image());
    prime(consecutiveMem);
    require(consecutiveMem.issued_count() == 0
            && consecutiveMem.timed_issue_writes() == 0
            && consecutiveMem.timed_issue_reads() == 0,
        "MEM PRIME executed or pre-generated an FU instruction");
    for (std::size_t cycle = 0; cycle < 4; ++cycle) {
        const auto instruction = consecutiveMem.tick();
        require(instruction.has_value()
                && instruction->opcode == MemOpcode::Read
                && instruction->address == 100 + cycle,
            "direct MEM Macro issue missed a consecutive cycle");
    }
    require(!consecutiveMem.tick().has_value()
            && consecutiveMem.issued_count() == 4
            && consecutiveMem.timed_issue_occupancy() == 0
            && consecutiveMem.peak_timed_issue_occupancy() == 0,
        "direct MEM Macro issue did not retire exactly four operations");

    InstructionControlUnit::MemIcu canonicalLoad;
    const auto repeatImage = make_mem_repeat_image();
    for (std::size_t address = 0; address < repeatImage.size(); ++address) {
        std::array<std::uint32_t,4> loadData{};
        std::copy(repeatImage[address].lanes.begin(),
            repeatImage[address].lanes.end(),loadData.begin());
        canonicalLoad.load_mem_program_word(address,loadData,0x7,
            address+1==repeatImage.size());
        const auto& load = canonicalLoad.mem_local_imem_load_port_signals();
        require(load.load_valid_i && load.load_ready_o && load.load_fire
                && load.load_imem_address_i==address
                && load.load_strb_i==0x7,
            "MEM Program load adapter lost canonical load pins");
    }
    canonicalLoad.prime_mem_program(
        IcuMacroQueueKind::Mem,repeatImage.size());
    require(canonicalLoad.mem_local_imem_load_port_signals().prime_fire,
        "MEM Program PRIME did not fire through the canonical adapter");
    prime(canonicalLoad);
    const auto canonicalIssue = canonicalLoad.dispatch_mem_issue(true);
    const auto& issuePins = canonicalLoad.mem_issue_port_signals();
    require(canonicalIssue.has_value() && issuePins.mem_issue_valid_o
            && issuePins.mem_issue_ready_i && issuePins.mem_issue_fire
            && issuePins.mem_issue_instruction_o.has_value()
            && issuePins.mem_issue_cycle_o==0 && !issuePins.fault_o,
        "direct MEM Scheduler issue pins do not describe the accepted issue");

    InstructionControlUnit::MemIcu backpressured;
    backpressured.load_raw_macro(
        IcuMacroQueueKind::Mem,make_mem_repeat_image());
    prime(backpressured);
    auto sawBackpressureFault=false;
    try { static_cast<void>(backpressured.dispatch_mem_issue(false)); }
    catch (const StaticScheduleError&) { sawBackpressureFault=true; }
    const auto& blockedPins=backpressured.mem_issue_port_signals();
    require(sawBackpressureFault && blockedPins.mem_issue_valid_o
            && !blockedPins.mem_issue_ready_i && !blockedPins.mem_issue_fire
            && blockedPins.fault_o
            && blockedPins.fault_code_o==static_cast<std::uint8_t>(
                MemMacroSchedulerFault::IssueBackpressure)
            && backpressured.issued_count()==0,
        "due-cycle MEM backpressure was retried or lost its terminal fault");

    const auto memDdbLayout =
        IcuMacroV1Decoder<MemInstruction, 96>(
            IcuMacroQueueKind::Mem, kMemImage).ddb_layout();
    const auto loadDdbLayout =
        IcuMacroV1Decoder<MxmControlInstruction, 128>(
            IcuMacroQueueKind::MxmLoad, kMxmLoadImage).ddb_layout();
    const auto computeDdbLayout =
        IcuMacroV1Decoder<MxmControlInstruction, 128>(
            IcuMacroQueueKind::MxmCompute, kMxmComputeImage).ddb_layout();
    const auto dequantDdbLayout =
        IcuMacroV1Decoder<MxmDequantInstruction, 128>(
            IcuMacroQueueKind::MxmDequant, kMxmDequantImage).ddb_layout();
    require(memDdbLayout.entry_bits == 0,
        "RTL-equivalent MEM decoder still exposes the retired generic DDB");
    require(loadDdbLayout.entry_bits == 710
            && computeDdbLayout.entry_bits == 741
            && dequantDdbLayout.entry_bits == 630,
        "MXM DDB physical layouts have the wrong bit widths");

    using PipelinedFetchDecoder = IcuMacroV1Decoder<
        MemInstruction, 96, 2, 3, 64, 8, 8, 8, 1>;
    using NonPipelinedFetchDecoder = IcuMacroV1Decoder<
        MemInstruction, 96, 2, 3, 64, 8, 8, 8, 2>;
    static_assert(PipelinedFetchDecoder::imem_read_latency == 2);
    static_assert(
        PipelinedFetchDecoder::imem_request_initiation_interval == 1);
    PipelinedFetchDecoder pipelinedFetch(
        IcuMacroQueueKind::Mem, kMemImage);
    NonPipelinedFetchDecoder nonPipelinedFetch(
        IcuMacroQueueKind::Mem, kMemImage);
    PipelinedFetchDecoder pinNamedFetch(
        IcuMacroQueueKind::Mem,kMemImage);
    static_cast<void>(pinNamedFetch.tick(true));
    const auto descriptorStart=pinNamedFetch.imem_read_port_signals();
    require(pinNamedFetch.fetch_descriptor_started()
            && pinNamedFetch.fetch_descriptor_active()
            && !descriptorStart.imem_read_req_valid_o
            && !descriptorStart.imem_read_req_fire,
        "fetch descriptor start did not retain its registered boundary");

    for (std::size_t address = 0; address < kMemImage.size(); ++address) {
        static_cast<void>(pinNamedFetch.tick(true));
        const auto& pins=pinNamedFetch.imem_read_port_signals();
        require(pins.imem_read_req_valid_o
                && pins.imem_read_req_ready_i
                && pins.imem_read_req_fire
                && pins.imem_read_req_address_o == address,
            "descriptor fetch did not issue consecutive local iMEM reads");
        require(!pins.imem_read_rsp_fire,
            "local iMEM response arrived before all pipelined requests fired");
    }
    require(!pinNamedFetch.control_latched()
            && pinNamedFetch.pending_fetch_level()
                    + pinNamedFetch.response_fifo_level() >= 2,
        "payload fetches waited for control decode or lost outstanding reads");

    static_cast<void>(pinNamedFetch.tick(true));
    const auto firstRead=pinNamedFetch.imem_read_port_signals();
    require(firstRead.imem_read_rsp_valid_i
            && firstRead.imem_read_rsp_ready_o
            && firstRead.imem_read_rsp_fire
            && firstRead.imem_read_rsp_data_i.has_value()
            && !pinNamedFetch.fetch_complete(),
        "local iMEM request-to-response latency or completion boundary changed");
    for (std::size_t cycle = 0;
         cycle < kMemImage.size() && !pinNamedFetch.fetch_complete(); ++cycle)
        static_cast<void>(pinNamedFetch.tick(true));
    require(pinNamedFetch.fetch_complete()
            && !pinNamedFetch.fetch_descriptor_active(),
        "descriptor fetch did not complete on the last accepted response");

    DistributedIcuQueue<MemInstruction, 96, 16, 4, 2, 4>
        fetchCompleteReady;
    fetchCompleteReady.load_raw_macro(IcuMacroQueueKind::Mem, kMemImage);
    for (std::size_t cycle = 0; !fetchCompleteReady.raw_macro_ready(); ++cycle) {
        if (cycle > 64)
            throw std::runtime_error(
                "fetch-complete readiness probe did not finish");
        fetchCompleteReady.prefetch_raw_macro();
        if (!fetchCompleteReady.macro_fetch_complete())
            require(!fetchCompleteReady.raw_macro_ready(),
                "frontend became ready before descriptor fetch completed");
    }
    require(fetchCompleteReady.macro_fetch_complete(),
        "frontend ready did not retain descriptor fetch completion");
    std::size_t pipelinedContexts = 0;
    std::size_t nonPipelinedContexts = 0;
    for (std::size_t cycle = 0; cycle < 6; ++cycle) {
        if (pipelinedFetch.tick(true).has_value()) ++pipelinedContexts;
        if (nonPipelinedFetch.tick(true).has_value())
            ++nonPipelinedContexts;
    }
    require(pipelinedFetch.statistics().response_fifo_writes == 3
            && nonPipelinedFetch.statistics().response_fifo_writes == 2,
        "Macro i-MEM read latency is still coupled to request II");
    for (std::size_t cycle = 6;
         (!pipelinedFetch.done() || !nonPipelinedFetch.done()); ++cycle) {
        if (cycle > 128)
            throw std::runtime_error(
                "parameterized Macro fetch pipelines did not drain");
        if (!pipelinedFetch.done()
            && pipelinedFetch.tick(true).has_value())
            ++pipelinedContexts;
        if (!nonPipelinedFetch.done()
            && nonPipelinedFetch.tick(true).has_value())
            ++nonPipelinedContexts;
    }
    require(pipelinedContexts == 2 && nonPipelinedContexts == 2,
        "Macro fetch II changed decoded context semantics");

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
    const auto extendedTiming = decode_statistics<MemInstruction, 96>(
        IcuMacroQueueKind::Mem, kMemExtendedImage);
    require(extendedTiming.decoder_active_cycles
                - extendedTiming.decoder_starvation_cycles
            == 9,
        "extended MEM template did not preserve the RTL header/payload/validate stages");

    const auto compactMxmTiming =
        decode_statistics<MxmControlInstruction, 128>(
            IcuMacroQueueKind::MxmCompute, kMxmComputeImage);
    require(compactMxmTiming.decoder_active_cycles
                - compactMxmTiming.decoder_starvation_cycles
            == 6,
        "compact MXM template did not use prefix/decode stages");

    using RtlMemDecoder = IcuMacroV1Decoder<
        MemInstruction, 96, 1, 3, 64, 1, 8, 1>;
    RtlMemDecoder heldContext(IcuMacroQueueKind::Mem, kMemImage);
    for (std::size_t cycle = 0;
         cycle < 64 && heldContext.rtl_mem_state_code() != 10; ++cycle)
        static_cast<void>(heldContext.tick(false));
    require(heldContext.rtl_mem_state_code() == 10
            && heldContext.pending_decoded_contexts() == 1
            && heldContext.ddb_occupancy() == 0,
        "MEM decoder did not reach the RTL OUTPUT_CONTEXT holding state");
    const auto heldBits = heldContext.reservoir_occupancy_bits();
    for (unsigned cycle = 0; cycle < 3; ++cycle) {
        require(!heldContext.tick(false).has_value()
                && heldContext.rtl_mem_state_code() == 10
                && heldContext.reservoir_occupancy_bits() == heldBits,
            "OUTPUT_CONTEXT changed parser state under backpressure");
    }
    const auto releasedContext = heldContext.tick(true);
    require(releasedContext.has_value()
            && releasedContext->instruction.address == 100
            && heldContext.rtl_mem_state_code() == 6
            && heldContext.statistics().context_stall_cycles == 3,
        "OUTPUT_CONTEXT did not fire exactly once when ready returned");

    using TwoWordReservoirDecoder = IcuMacroV1Decoder<
        MemInstruction, 96, 1, 2, 64, 8, 8, 8>;
    static_assert(TwoWordReservoirDecoder::reservoir_words == 2);
    TwoWordReservoirDecoder twoWordReservoir(
        IcuMacroQueueKind::Mem, kMemImage);
    std::size_t twoWordContexts = 0;
    for (std::size_t cycle = 0; !twoWordReservoir.done(); ++cycle) {
        if (cycle > 64)
            throw std::runtime_error(
                "two-word-reservoir Macro decoder did not finish");
        if (twoWordReservoir.tick(true).has_value()) ++twoWordContexts;
    }
    require(twoWordContexts == 2
            && twoWordReservoir.statistics().peak_reservoir_bits <= 2 * 96,
        "two-word Macro reservoir changed decode semantics or overflowed");

    IcuMacroV1Decoder<MemInstruction, 96> fifoBackpressure(
        IcuMacroQueueKind::Mem, kMemExtendedImage);
    for (std::size_t cycle = 0;
         cycle < 64 && fifoBackpressure.rtl_mem_state_code() != 10; ++cycle)
        static_cast<void>(fifoBackpressure.tick(
            cycle, 1, 1, std::optional<std::size_t>{20}));
    require(fifoBackpressure.rtl_mem_state_code() == 10
            && !fifoBackpressure.tick(
                64, 1, 1, std::optional<std::size_t>{20}).has_value(),
        "full Context FIFO did not backpressure the MEM decoder");

    const auto compactEscape = decode_mem_image(kMemCompactEscapeImage);
    require(compactEscape.size() == 2
            && compactEscape[0].schedule.start_cycle == 1
            && compactEscape[1].schedule.start_cycle == 5
            && compactEscape[0].instruction.address == 10
            && compactEscape[1].instruction.address == 11,
        "compact delta escape decoded incorrectly");
    const auto compactEscapeTiming = decode_statistics<MemInstruction, 96>(
        IcuMacroQueueKind::Mem, kMemCompactEscapeImage);

    const auto wideEscape = decode_mem_image(kMemWideEscapeImage);
    require(wideEscape.size() == 2
            && wideEscape[0].schedule.start_cycle == 1
            && wideEscape[1].schedule.start_cycle == 5000000
            && wideEscape[0].instruction.address == 10
            && wideEscape[1].instruction.address == 11,
        "wide delta escape decoded incorrectly");
    const auto wideEscapeTiming = decode_statistics<MemInstruction, 96>(
        IcuMacroQueueKind::Mem, kMemWideEscapeImage);
    const auto compactEscapeService =
        compactEscapeTiming.decoder_active_cycles
        - compactEscapeTiming.decoder_starvation_cycles;
    const auto wideEscapeService = wideEscapeTiming.decoder_active_cycles
        - wideEscapeTiming.decoder_starvation_cycles;
    require(wideEscapeService == compactEscapeService,
        "RTL ESCAPE payload state did not consume narrow/wide fields in one state cycle");

    require(kMemImage[0].bit(0) && !kMemImage[0].bit(1)
            && kMemImage[0].bit(2) && !kMemImage[0].bit(3)
            && !kMemImage[0].bit(4) && kMemImage[0].bit(5),
        "raw word does not map lane-0 bit 0 to the first stream bit");
    auto reservedBitOneImage = kMemImage;
    reservedBitOneImage[0].lanes[0] |= 0x2u;
    bool rejectedReservedBitOne = false;
    try {
        IcuMacroV1Decoder<MemInstruction, 96> invalid(
            IcuMacroQueueKind::Mem, reservedBitOneImage);
        for (std::size_t cycle = 0; cycle < 6; ++cycle)
            static_cast<void>(invalid.tick(true));
    } catch (const StaticScheduleError&) {
        rejectedReservedBitOne = true;
    }
    require(rejectedReservedBitOne,
        "raw Macro decoder accepted reserved control bit 1");
    auto invalidControlImage = kMemImage;
    invalidControlImage[0].lanes[0] |= 0x10000000u;
    bool rejectedReservedControl = false;
    try {
        IcuMacroV1Decoder<MemInstruction, 96> invalid(
            IcuMacroQueueKind::Mem, invalidControlImage);
        // Descriptor start, two-cycle SRAM read, response and word FIFO
        // boundaries place the control word at the decoder on cycle six.
        for (std::size_t cycle = 0; cycle < 6; ++cycle)
            static_cast<void>(invalid.tick(true));
    } catch (const StaticScheduleError&) {
        rejectedReservedControl = true;
    }
    require(rejectedReservedControl,
        "raw Macro decoder accepted reserved control-word bits");

    const auto requireMemFault = [&](
        const std::vector<IcuRawImemWord96>& image,
        std::uint8_t expectedCode, const char* message) {
        IcuMacroV1Decoder<MemInstruction, 96> decoder(
            IcuMacroQueueKind::Mem, image);
        bool rejected = false;
        try {
            for (std::size_t cycle = 0; cycle < 128 && !decoder.done(); ++cycle)
                static_cast<void>(decoder.tick(true));
        } catch (const StaticScheduleError&) {
            rejected = true;
        }
        require(rejected && decoder.rtl_mem_fault()
                && decoder.rtl_mem_state_code() == 11
                && decoder.rtl_mem_fault_code() == expectedCode,
            message);
    };

    auto illegalNativeImage = kMemExtendedImage;
    // Stream bit 145 is the Extended flag; native[28] is bit 174.
    assign_mem_stream_bits(illegalNativeImage, 174, 1, 1);
    requireMemFault(illegalNativeImage, 6,
        "MEM decoder did not retain RTL FAULT_NATIVE semantics");

    auto illegalScheduleImage = kMemExtendedImage;
    // Extended schedule target occupies stream bits 371:370. Target zero is
    // illegal for the MEM address induction path.
    assign_mem_stream_bits(illegalScheduleImage, 370, 0, 2);
    requireMemFault(illegalScheduleImage, 7,
        "MEM decoder did not retain RTL FAULT_SCHEDULE semantics");

    auto illegalDeltaImage = kMemImage;
    // Dictionary[0] operand begins at stream bit 121. +8191 overflows the
    // legal 13-bit MEM address after the first Context.
    assign_mem_stream_bits(illegalDeltaImage, 121, 0x1fff, 14);
    requireMemFault(illegalDeltaImage, 8,
        "MEM decoder did not retain RTL FAULT_DELTA semantics");

    auto truncatedImage = kMemImage;
    truncatedImage.pop_back();
    requireMemFault(truncatedImage, 3,
        "MEM decoder did not retain RTL FAULT_TRUNCATED semantics");

    // The MEM image is a hard-coded physical golden vector. It contains two
    // singleton reads with one dictionary transition and crosses a 96-bit
    // payload-word boundary:
    //   cycle 3: Read(address=100, stream=2)
    //   cycle 7: Read(address=101, stream=2)
    using ConstrainedMemQueue =
        DistributedIcuQueue<MemInstruction, 96, 16, 4, 1, 4>;
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
    require(constrained.queued_count() == 0,
        "completed raw MEM Macro queue retained duplicate work accounting");
    require(constrained.macro_decoder_statistics().fetched_words == 3,
        "raw MEM decoder fetched the wrong number of physical words");
    require(constrained.macro_decoder_statistics().decoded_contexts == 2,
        "raw MEM decoder produced the wrong number of contexts");
    require(constrained.macro_decoder_statistics().descriptor_count != 0
            && constrained.macro_decoder_statistics().ddb_entries_committed == 0,
        "raw MEM frontend did not use the RTL decoder or retained the retired DDB");
    require(constrained.macro_decoder_statistics().payload_bits_consumed != 0
            && constrained.macro_decoder_statistics().decoder_active_cycles
                != 0,
        "raw MEM frontend did not expose cycle-accurate parser work");
    require(constrained.peak_active_macros() == 2,
        "raw MEM scheduler did not retain both primed Contexts");

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
    require(icu.mem_iq(0).timed_issue_writes() == 0
            && icu.mem_iq(0).timed_issue_reads() == 0
            && icu.mxm_load_iq(0).timed_issue_writes() == 1
            && icu.mxm_load_iq(0).timed_issue_reads() == 1
            && icu.mxm_compute_iq(0).timed_issue_writes() == 1
            && icu.mxm_compute_iq(0).timed_issue_reads() == 1
            && icu.mxm_dequant_iq(0).timed_issue_writes() == 1
            && icu.mxm_dequant_iq(0).timed_issue_reads() == 1,
        "MEM direct issue or MXM timed issue accounting changed");
    require(icu.mem_iq(0).peak_timed_issue_occupancy() == 0
            && icu.mem_iq(0).timed_issue_occupancy() == 0,
        "MEM unexpectedly used the removed timed issue queue");

    const auto statistics = icu.frontend_statistics();
    require(statistics.imem_entries == 9,
        "raw Macro statistics did not count physical i-MEM words");
    require(statistics.fetched_entries == 9,
        "raw Macro statistics reported the wrong fetch count");
    require(statistics.decoded_macro_contexts == 5,
        "raw Macro statistics reported the wrong decoded-context count");
    require(statistics.peak_macro_reservoir_bits != 0,
        "raw Macro statistics did not expose reservoir occupancy");
    require(statistics.macro_fetched_words == 9
            && statistics.macro_decoder_active_cycles != 0,
        "raw Macro aggregate statistics omitted fetch or decoder activity");
    require(statistics.context_fifo_writes == 5
            && statistics.context_fifo_reads == 5
            && statistics.timed_issue_writes == 3
            && statistics.timed_issue_reads == 3,
        "Macro Context/Timed FIFO aggregate accounting is not conserved");
    require(statistics.peak_imem_response_fifo_occupancy != 0
            && statistics.peak_macro_word_fifo_occupancy != 0
            && statistics.peak_context_fifo_occupancy != 0
            && statistics.peak_timed_issue_occupancy != 0,
        "Macro frontend FIFO occupancy is not observable");

    const auto queueStatistics = icu.macro_queue_frontend_statistics();
    require(queueStatistics.size() == 4,
        "raw Macro statistics did not report every configured queue");
    const auto findQueue = [&](IcuMacroQueueKind kind) {
        return std::find_if(queueStatistics.begin(), queueStatistics.end(),
            [&](const IcuMacroQueueFrontendStatistics& queue) {
                return queue.kind == kind && queue.queue_index == 0;
            });
    };
    const auto memQueue = findQueue(IcuMacroQueueKind::Mem);
    const auto loadQueue = findQueue(IcuMacroQueueKind::MxmLoad);
    const auto computeQueue = findQueue(IcuMacroQueueKind::MxmCompute);
    const auto dequantQueue = findQueue(IcuMacroQueueKind::MxmDequant);
    require(memQueue != queueStatistics.end()
            && loadQueue != queueStatistics.end()
            && computeQueue != queueStatistics.end()
            && dequantQueue != queueStatistics.end(),
        "raw Macro per-queue statistics lost a queue identity");
    require(memQueue->decoder.decoded_contexts == 2
            && loadQueue->decoder.decoded_contexts == 1
            && computeQueue->decoder.decoded_contexts == 1
            && dequantQueue->decoder.decoded_contexts == 1,
        "raw Macro per-queue statistics reported wrong context counts");
    require(memQueue->active_context_bits == 256
            && loadQueue->active_context_bits == 256
            && computeQueue->active_context_bits == 256
            && dequantQueue->active_context_bits == 256,
        "raw Macro per-queue statistics reported the wrong context width");
    require(memQueue->context_fifo_depth == 4
            && memQueue->timed_issue_depth == 0
            && memQueue->timed_issue_lookahead == 0
            && loadQueue->timed_issue_depth == 16
            && loadQueue->timed_issue_lookahead == 2,
        "raw Macro per-queue statistics lost MEM/MXM issue geometry");
    require(memQueue->ddb_entry_bits == 0
            && loadQueue->ddb_entry_bits == 710
            && computeQueue->ddb_entry_bits == 741
            && dequantQueue->ddb_entry_bits == 630,
        "raw Macro per-queue statistics reported wrong DDB layouts");

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
