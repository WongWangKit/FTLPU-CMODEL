#include "ftlpu/system/program_distribution_transport.hpp"

#include <iostream>
#include <stdexcept>

using namespace ftlpu;

namespace {

void require(bool condition,const char* message)
{
    if(!condition) throw std::runtime_error(message);
}

ProgramDistributionRecord record(std::uint8_t seed,
    ProgramLoadScope scope=ProgramLoadScope::East)
{
    ProgramDistributionRecord value{};
    value.scope=scope;
    value.destination=0; // East, MEM, local Bank 0.
    value.strb=7;
    value.imem_address=seed;
    value.image_last=seed==3;
    for(std::size_t index=0;index<12;++index)
        value.data[index]=static_cast<std::uint8_t>(seed+index);
    return value;
}

ProgramDistributionHardwareInterfaceView step(
    ProgramDistributionTransport& transport,
    ProgramDistributionInputs inputs={})
{
    transport.drive(std::move(inputs));
    transport.low();
    const auto view=transport.hardware_interface_view();
    transport.posedge();
    return view;
}

void both_scope_and_reset()
{
    ProgramDistributionTransport transport;
    ProgramDistributionInputs inputs{};
    inputs.load_record_valid=true;
    inputs.load_record=record(1,ProgramLoadScope::Both);
    auto view=step(transport,inputs);
    require(view.load_record_input.fire&&view.elastic_level==0,
        "Distributor Elastic did not accept the BOTH record");

    view=step(transport);
    require(view.elastic_output.fire
            &&view.distributor_route_required[0]
            &&view.distributor_route_required[1]
            &&view.distributor_route_output[0].fire
            &&view.distributor_route_output[1].fire,
        "Distributor did not independently transfer BOTH branches");

    inputs={};
    inputs.mem_endpoint_ready[0][0]=false;
    inputs.mem_endpoint_ready[1][0]=true;
    view=step(transport,inputs);
    require(view.mem_endpoint_input[0][0].valid
            &&!view.mem_endpoint_input[0][0].fire
            &&view.mem_endpoint_input[1][0].fire,
        "BOTH Relay branches did not accept independent backpressure");
    const auto held=view.mem_endpoint_input[0][0].payload;

    inputs.mem_endpoint_ready[0][0]=true;
    view=step(transport,inputs);
    require(view.mem_endpoint_input[0][0].fire
            &&view.mem_endpoint_input[0][0].payload==held,
        "stalled Relay payload was not held until acceptance");

    inputs={};
    inputs.reset=true;
    (void)step(transport,inputs);
    require(transport.idle()&&!transport.fault(),
        "Program Distribution reset did not clear transport state");
}

void relay_full_forbids_replacement()
{
    ProgramDistributionTransport transport;
    for(std::uint8_t seed=1;seed<=3;++seed) {
        ProgramDistributionInputs inputs{};
        inputs.load_record_valid=true;
        inputs.load_record=record(seed);
        inputs.mem_endpoint_ready[0][0]=false;
        const auto view=step(transport,inputs);
        require(view.load_record_input.fire,
            "Elastic input unexpectedly rejected the directed record");
    }

    ProgramDistributionInputs release{};
    release.mem_endpoint_ready[0][0]=true;
    auto view=step(transport,release);
    require(view.relay_level[0][0]==2
            &&view.mem_endpoint_input[0][0].fire
            &&!view.relay_input[0][0].ready
            &&view.distributor_route_output[0].valid
            &&!view.distributor_route_output[0].fire,
        "full Relay incorrectly allowed same-cycle dequeue replacement");

    view=step(transport,release);
    require(view.distributor_route_output[0].fire
            &&view.relay_input[0][0].fire,
        "Relay did not accept the held record after capacity became visible");
}

} // namespace

int main() try
{
    both_scope_and_reset();
    relay_full_forbids_replacement();
    std::cout<<"program_distribution_transport_test passed\n";
    return 0;
} catch(const std::exception& error) {
    std::cerr<<"program_distribution_transport_test failed: "
        <<error.what()<<'\n';
    return 1;
}
