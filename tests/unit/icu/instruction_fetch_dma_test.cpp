#include "ftlpu/icu/instruction_fetch_dma.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace ftlpu {
struct InstructionFetchDmaTestAccess {
    static void corrupt_front_response_id(InstructionFetchDma& dma) {
        if(dma.state_.responses.empty())
            throw std::logic_error("no response to corrupt");
        dma.state_.responses.front().dma_id^=1;
    }
    static void mark_front_response_error(InstructionFetchDma& dma) {
        if(dma.state_.responses.empty())
            throw std::logic_error("no response to mark");
        dma.state_.responses.front().error=true;
    }
};
}

using namespace ftlpu;

namespace {
void require(bool value,const char* message)
{if(!value)throw std::runtime_error(message);}

Ddr4Config ddr_config()
{
    Ddr4Config config{};
    config.beat_bytes=32;
    config.read_latency_cycles=2;
    config.read_latency_jitter_cycles=0;
    config.peak_bandwidth_bytes_per_second=config.lpu_clock_hz*32;
    config.transfer_channels=1;
    return config;
}

std::vector<std::uint8_t> bytes(std::size_t count,std::uint8_t seed)
{
    std::vector<std::uint8_t> result(count);
    for(std::size_t i=0;i<count;++i)
        result[i]=static_cast<std::uint8_t>(seed+i*37+(i>>1)*11);
    return result;
}

void stage(Ddr4Model& ddr,std::uint64_t address,
    const std::vector<std::uint8_t>& source)
{
    for(std::size_t offset=0;offset<source.size();offset+=32) {
        C2cVector vector{};
        for(std::size_t b=0;b<32;++b)
            vector.payload[b/hw::kLanesPerTile][b%hw::kLanesPerTile]=
                offset+b<source.size()?source[offset+b]:0xa5;
        ddr.initialize_vector(address+offset,vector);
    }
}

struct ProgramFixture {
    InstructionFetchProgram program{};
    std::vector<std::uint8_t> source{};
};

std::vector<PhysicalInstructionWord> run(
    std::vector<ProgramFixture> fixtures,
    InstructionFetchDmaConfig dmaConfig={},
    std::size_t releaseCycle=0,
    InstructionFetchDmaStatistics* statistics=nullptr)
{
    Ddr4Model ddr(ddr_config());
    for(std::size_t i=0;i<fixtures.size();++i) {
        fixtures[i].program.base_address=0x1003+i*0x4000;
        stage(ddr,fixtures[i].program.base_address,fixtures[i].source);
    }
    InstructionFetchDma dma(ddr,dmaConfig);
    std::vector<PhysicalInstructionWord> output;
    std::size_t next=0;
    for(std::size_t cycle=0;cycle<100000;++cycle) {
        if(next<fixtures.size()&&dma.can_accept_program())
            dma.start(fixtures[next++].program);
        if(cycle>=releaseCycle&&dma.word_ready()) {
            output.push_back(dma.front_word());
            dma.pop_word();
        }
        dma.tick();
        ddr.tick();
        if(next==fixtures.size()&&dma.drained()&&ddr.drained())break;
        require(cycle+1<100000,"instruction DMA scenario timed out");
    }
    if(statistics)*statistics=dma.statistics();
    return output;
}

void compare(const std::vector<PhysicalInstructionWord>& words,
    const std::vector<ProgramFixture>& fixtures)
{
    std::size_t output=0;
    for(const auto& fixture:fixtures) {
        const auto wordBytes=physical_instruction_word_bytes(
            fixture.program.word_width);
        for(std::size_t word=0;word<fixture.program.word_count;++word) {
            require(output<words.size(),"instruction word was lost");
            require(words[output].index==word
                &&words[output].byte_count==wordBytes,
                "instruction word metadata is wrong");
            for(std::size_t b=0;b<wordBytes;++b)
                require(words[output].bytes[b]
                    ==fixture.source[word*wordBytes+b],
                    "instruction bytes were lost, duplicated, or reordered");
            ++output;
        }
    }
    require(output==words.size(),"unexpected instruction word was emitted");
}

ProgramFixture fixture(std::size_t words,
    PhysicalInstructionWordWidth width,std::uint8_t seed)
{
    ProgramFixture result{};
    result.program.word_count=words;
    result.program.word_width=width;
    result.source=bytes(words*physical_instruction_word_bytes(width),seed);
    return result;
}

void single_and_multi_outstanding()
{
    auto single=fixture(11,PhysicalInstructionWordWidth::Bits96,0x21);
    auto singleCopy=std::vector<ProgramFixture>{single};
    compare(run(singleCopy),singleCopy);

    auto fixtures=std::vector<ProgramFixture>{
        fixture(40,PhysicalInstructionWordWidth::Bits96,0x31),
        fixture(20,PhysicalInstructionWordWidth::Bits128,0x52),
        fixture(24,PhysicalInstructionWordWidth::Bits64,0x73)};
    InstructionFetchDmaStatistics stats{};
    const auto output=run(fixtures,{},0,&stats);
    compare(output,fixtures);
    require(stats.command_handshakes==3
        &&stats.outstanding_peak>=2,
        "multiple instruction commands were not outstanding");
    require(stats.ar_handshakes==stats.r_handshakes,
        "DDR request/response count did not balance");
}

void response_fifo_backpressure_and_recovery()
{
    InstructionFetchDmaConfig config{};
    config.response_fifo_depth=2;
    auto fixtures=std::vector<ProgramFixture>{
        fixture(80,PhysicalInstructionWordWidth::Bits96,0x19)};
    InstructionFetchDmaStatistics stats{};
    const auto output=run(fixtures,config,120,&stats);
    compare(output,fixtures);
    require(stats.response_fifo_peak==2
        &&stats.response_fifo_full_cycles>0
        &&stats.queue_backpressure_cycles>0,
        "downstream stall did not fill and backpressure the response FIFO");
}

void id_error_and_reset_paths()
{
    auto exercise=[](bool responseError) {
        Ddr4Model ddr(ddr_config());
        auto item=fixture(24,PhysicalInstructionWordWidth::Bits96,0x44);
        item.program.base_address=0x9003;
        stage(ddr,item.program.base_address,item.source);
        InstructionFetchDma dma(ddr);
        dma.start(item.program);
        for(std::size_t cycle=0;dma.response_fifo_occupancy()==0;++cycle) {
            require(cycle<1000,"response FIFO never became non-empty");
            dma.tick();ddr.tick();
        }
        if(responseError)
            InstructionFetchDmaTestAccess::mark_front_response_error(dma);
        else
            InstructionFetchDmaTestAccess::corrupt_front_response_id(dma);
        dma.tick();ddr.tick();
        require(dma.fault()&&dma.fault_code()==(responseError?2:1),
            "instruction DMA did not latch the expected response fault");
        dma.reset();
        for(std::size_t cycle=0;!dma.drained()||!ddr.drained();++cycle) {
            require(cycle<1000,"instruction DMA reset did not drain orphan DDR responses");
            dma.tick();ddr.tick();
        }
    };
    exercise(false);
    exercise(true);
}
}

int main() try
{
    single_and_multi_outstanding();
    response_fifo_backpressure_and_recovery();
    id_error_and_reset_paths();
    std::cout<<"instruction_fetch_dma_test passed\n";
    return 0;
} catch(const std::exception& error) {
    std::cerr<<"instruction_fetch_dma_test failed: "
        <<error.what()<<'\n';
    return 1;
}
