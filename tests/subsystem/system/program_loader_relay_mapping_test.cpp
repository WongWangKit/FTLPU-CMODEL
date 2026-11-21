#include "ftlpu/system/program_loader.hpp"

#include <array>
#include <iostream>
#include <stdexcept>

using namespace ftlpu;

namespace {
void require(bool value,const char* message)
{if(!value)throw std::runtime_error(message);}

void exhaustive_mem_mapping()
{
    constexpr auto banksPerRelay=hw::kProgramLoadMemBanksPerRelay;
    constexpr auto queuesPerHemisphere=
        InstructionControlUnit::kMemQueuesPerHemisphere;
    constexpr auto expectedRelayCount=
        (queuesPerHemisphere+banksPerRelay-1)/banksPerRelay;
    static_assert(queuesPerHemisphere==104);
    static_assert(banksPerRelay==16);
    static_assert(expectedRelayCount
        ==hw::config::kProgramLoadRelaysPerHemisphere);

    for(std::size_t queue=0;
        queue<InstructionControlUnit::kMemQueues;++queue) {
        ProgramImage image{};
        image.kind=IcuMacroQueueKind::Mem;
        image.queue_index=queue;
        image.byte_count=12;
        const auto [hemisphere,relay]=
            ProgramLoader::relay_destination_for_test(image);
        const auto expectedHemisphere=queue/queuesPerHemisphere;
        const auto local=queue%queuesPerHemisphere;
        const auto expectedRelay=local/banksPerRelay;
        require(hemisphere==expectedHemisphere,
            "MEM Queue mapped to the wrong hemisphere");
        require(relay==expectedRelay,
            "MEM Queue mapped to the wrong Relay");
    }

    for(std::size_t relay=0;relay<expectedRelayCount;++relay) {
        const auto first=relay*banksPerRelay;
        const auto last=std::min(queuesPerHemisphere,
            first+banksPerRelay)-1;
        std::cout<<"R"<<relay<<" local_mem_banks="
            <<first<<"-"<<last<<'\n';
    }
}

void mxm_uses_tail_relay()
{
    constexpr std::array kinds{
        IcuMacroQueueKind::MxmLoad,
        IcuMacroQueueKind::MxmDequant,
        IcuMacroQueueKind::MxmCompute};
    constexpr auto perHemisphere=
        InstructionControlUnit::kMxmQueuesPerHemisphere;
    constexpr auto tail=
        hw::config::kProgramLoadRelaysPerHemisphere-1;
    for(const auto kind:kinds)
        for(std::size_t queue=0;
            queue<InstructionControlUnit::kMxmQueues;++queue) {
            ProgramImage image{};
            image.kind=kind;
            image.queue_index=queue;
            image.byte_count=16;
            const auto [hemisphere,relay]=
                ProgramLoader::relay_destination_for_test(image);
            require(hemisphere==queue/perHemisphere,
                "MXM Queue mapped to the wrong hemisphere");
            require(relay==tail,
                "MXM Queue did not use the tail Relay");
        }
}
}

int main() try
{
    exhaustive_mem_mapping();
    mxm_uses_tail_relay();
    std::cout<<"program_loader_relay_mapping_test passed\n";
    return 0;
} catch(const std::exception& error) {
    std::cerr<<"program_loader_relay_mapping_test failed: "
        <<error.what()<<'\n';
    return 1;
}
