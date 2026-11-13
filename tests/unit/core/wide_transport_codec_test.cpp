#include "ftlpu/c2c/ddr4.hpp"
#include "ftlpu/c2c/link.hpp"
#include "ftlpu/icu/distributed_queue.hpp"
#include "ftlpu/mxm/accumulator.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}
}

int main() try
{
    using namespace ftlpu;
    const auto address = hw::kMxmAccumulatorRows - 1;
    const auto column = hw::kMxmSupercellsPerPlane - 1;
    const auto iw = MxmControlInstruction::IWDirect16(1, column);
    require(isa::decode_mxm_instruction(isa::encode_mxm_instruction(iw)).weight_column == column,
        "FU lost the highest MXM supercell column");
    const auto compute = MxmControlInstruction::Compute(1, 0, 0, address, 1,
        MxmAccumulatorDestination::Stream, MxmDataFormat::BFloat16, false,
        MxmAccumulatorOutputFormat::BFloat16);
    const auto decoded = isa::decode_mxm_instruction(isa::encode_mxm_instruction(compute));
    require(decoded.accumulator_address == address && !decoded.accumulator_clear
        && decoded.accumulator_output_format == MxmAccumulatorOutputFormat::BFloat16,
        "FU accumulator address overlapped stride or control fields");
    const auto read = MxmControlInstruction::AccumulatorRead(address,0,false);
    require(isa::decode_mxm_instruction(isa::encode_mxm_instruction(read)).accumulator_address == address,
        "FU accumulator-read lost its highest address");
    const auto streamAddress=address-(column+hw::kMxmIsColumns-1)/hw::kMxmSupercellsPerPlane;
    const auto stream = MxmControlInstruction::IsStreamCompute(0,0,
        MxmDataFormat::BFloat16,streamAddress,column,MxmAccumulatorDestination::Sram,false,MxmIsMode::Block);
    const auto decodedStream = isa::decode_mxm_instruction(isa::encode_mxm_instruction(stream));
    require(decodedStream.accumulator_address == streamAddress && decodedStream.weight_column == column
        && decodedStream.data_format == MxmDataFormat::BFloat16 && !decodedStream.accumulator_clear,
        "FU stream-compute fields overlapped after widening");
    // The Macro context carries the same native FU fields.
    using Codec = IcuActiveMacroContextCodec256<MxmControlInstruction>;
    const IcuMacroSchedule schedule{0,1,1,0,1,1,0,IcuInductionTarget::None};
    require(Codec::unpack(Codec::pack_initial(schedule,compute)).instruction.accumulator_address==address,
        "Macro context lost the highest accumulator address");
    MxmAccumulator accumulator;
    MxmAccumulator::Segment segment{};
    segment.fill(2.5f);
    accumulator.accumulate(address,column,segment);
    require(accumulator.read(address,column)==segment && accumulator.value(address,hw::kMxmColumns-1)==2.5f,
        "last context/column is not physically backed by accumulator storage");
    bool rejected = false;
    try { accumulator.accumulate(hw::kMxmAccumulatorRows, column, segment); }
    catch (const std::exception&) { rejected = true; }
    require(rejected, "accumulator accepted the first row beyond configured capacity");
    C2cVector vector;
    vector.vector_tag=91;
    for (std::size_t i=0;i<hw::kPhysicalVectorBytes;++i)
        vector.payload[i/hw::kLanesPerTile][i%hw::kLanesPerTile]=static_cast<std::uint8_t>(i+1);
    constexpr auto beatBytes = hw::kC2cBytesPerStreamPerCycle;
    C2cLink link({beatBytes,0,1});
    link.send(vector);
    const auto beats=(hw::kPhysicalVectorBytes+beatBytes-1)/beatBytes;
    for (std::size_t i=0;i<beats;++i) {
        link.tick();
        require(link.receive_ready()==(i+1==beats),"C2C exposed a partially assembled vector");
        require(!link.can_send(),"C2C released credit before the whole vector was consumed");
    }
    const auto received=link.pop_received();
    require(received.payload==vector.payload && received.vector_tag==91 && link.can_send(),
        "C2C reassembly or credit return failed");
    Ddr4Config config;
    config.beat_bytes=beatBytes;config.read_latency_cycles=0;config.write_latency_cycles=0;
    config.read_latency_jitter_cycles=0;config.write_latency_jitter_cycles=0;
    config.transfer_channels=1;
    config.peak_bandwidth_bytes_per_second=config.lpu_clock_hz*beatBytes;
    Ddr4Model ddr(config);
    ddr.initialize_vector(0x1000,vector);
    const auto request=ddr.request_read(0x1000);
    for (std::size_t i=0;i<beats;++i) {
        ddr.tick();
        require(ddr.read_completion_ready(request)==(i+1==beats),"DDR completed a partial vector");
        require(ddr.read_bytes_transferred()==std::min<std::size_t>((i+1)*beatBytes,hw::kPhysicalVectorBytes),
            "DDR byte accounting used link beats as logical vectors");
    }
    require(ddr.pop_read_completion(request).vector.payload==vector.payload,"DDR read lost upper vector lanes");
    // Multiple pending requests share a physical lane rather than creating
    // extra link bandwidth. Other lanes retain independent DDR capacity.
    config.transfer_channels=2;
    config.peak_bandwidth_bytes_per_second=config.lpu_clock_hz*2*beatBytes;
    Ddr4Model concurrent(config);
    concurrent.initialize_vector(0x1000,vector);
    concurrent.initialize_vector(0x1100,vector);
    const auto first=concurrent.request_read(0x1000,0);
    const auto second=concurrent.request_read(0x1100,0);
    for (std::size_t i=0;i<2*beats;++i) {
        const auto before=concurrent.read_bytes_transferred();
        concurrent.tick();
        require(concurrent.read_bytes_transferred()-before<=beatBytes,
            "pending requests multiplied one C2C lane's beat bandwidth");
        if (i+1<2*beats)
            require(!(concurrent.read_completion_ready(first)&&concurrent.read_completion_ready(second)),
                "two vectors on one C2C lane completed faster than physical link bandwidth");
    }
    require(concurrent.read_completion_ready(first)&&concurrent.read_completion_ready(second),
        "per-lane transport arbitration did not complete both requests");
    Ddr4Model independent(config);
    independent.initialize_vector(0x1000,vector);
    independent.initialize_vector(0x1100,vector);
    const auto left=independent.request_read(0x1000,0);
    const auto right=independent.request_read(0x1100,2);
    for (std::size_t i=0;i<beats;++i) independent.tick();
    require(independent.read_completion_ready(left)&&independent.read_completion_ready(right)
        && independent.read_bytes_transferred()==2*hw::kPhysicalVectorBytes,
        "per-lane arbitration unnecessarily serialized independent C2C lanes");
    std::cout << "wide_transport_codec_test passed: lanes=" << hw::kPhysicalVectorBytes
        << " address_bits=" << hw::kMxmAccumulatorAddressBits << " column_bits=" << hw::kMxmColumnBits
        << " compute_words=" << Codec::word_count << " c2c_beats=" << beats << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';return 1;
}
