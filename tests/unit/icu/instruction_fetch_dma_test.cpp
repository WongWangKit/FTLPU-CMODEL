#include "ftlpu/icu/instruction_fetch_dma.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

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

void protocol_step(InstructionDmaProtocolCore& core,
    InstructionDmaProtocolInputs inputs)
{
    core.drive(std::move(inputs));
    core.low();
    core.validate_posedge();
    core.posedge();
    core.high();
}

InstructionDmaProtocolInputs command(std::uint8_t id)
{
    InstructionDmaProtocolInputs inputs{};
    inputs.command_valid=true;
    inputs.command_address=0x1000+id*0x100;
    inputs.command_length=64;
    inputs.command_id=id;
    inputs.request_ready=true;
    return inputs;
}

InstructionDmaProtocolInputs response(std::uint8_t id,bool error=false,
    bool reassemblerReady=false)
{
    InstructionDmaProtocolInputs inputs{};
    inputs.request_ready=true;
    inputs.response_valid=true;
    inputs.response.id=id;
    inputs.response.keep=0xffffffffU;
    inputs.response.last=true;
    inputs.response.error=error;
    inputs.reassembler_ready=reassemblerReady;
    return inputs;
}

void protocol_fault_and_reset_paths()
{
    InstructionDmaProtocolCore core;
    protocol_step(core,command(7));
    protocol_step(core,command(8));
    require(core.outstanding()==2,"protocol command FIFO did not fill");

    auto mismatch=response(99,true);
    core.drive(mismatch);
    require(core.response_input_fire()&&core.response_id_invalid()
        &&core.response_error()&&core.fault()
        &&core.fault_code()==kInstructionDmaFaultId,
        "ID fault did not have RTL priority over response error");
    core.low();core.validate_posedge();core.posedge();
    require(core.fault()&&core.fault_code()==kInstructionDmaFaultId
        &&core.outstanding()==1&&core.response_fifo_occupancy()==1,
        "mismatched last response did not enqueue and dequeue the head ID");

    auto blocked=response(8,false,true);
    blocked.command_valid=true;
    blocked.command_id=9;
    blocked.command_length=32;
    core.drive(blocked);
    require(!core.command_ready()&&!core.request_valid()
        &&!core.response_ready()&&core.response_fire(),
        "sticky fault did not block new input while allowing RX drain");
    core.low();core.validate_posedge();core.posedge();
    require(core.outstanding()==1&&core.response_fifo_occupancy()==0
        &&!core.idle(),"fault drain or stranded outstanding state is wrong");

    core.reset();
    require(core.idle()&&!core.fault(),"protocol reset did not clear state");
    protocol_step(core,command(9));
    protocol_step(core,response(9,true));
    require(core.fault()&&core.fault_code()==kInstructionDmaFaultResponse
        &&core.outstanding()==0&&core.response_fifo_occupancy()==1,
        "matching error response did not latch the RTL response fault");
    InstructionDmaProtocolInputs drain{};
    drain.reassembler_ready=true;
    protocol_step(core,drain);
    require(core.idle()&&core.fault(),
        "response fault did not preserve sticky fault after RX drain");

    core.reset();
    protocol_step(core,response(55));
    require(core.fault()&&core.fault_code()==kInstructionDmaFaultId
        &&core.outstanding()==0&&core.response_fifo_occupancy()==1,
        "response without an outstanding command did not fault");
}
}

int main() try
{
    single_and_multi_outstanding();
    response_fifo_backpressure_and_recovery();
    protocol_fault_and_reset_paths();
    std::cout<<"instruction_fetch_dma_test passed\n";
    return 0;
} catch(const std::exception& error) {
    std::cerr<<"instruction_fetch_dma_test failed: "
        <<error.what()<<'\n';
    return 1;
}
