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
    const auto loadRun = MxmLoadIcuInstruction::Load3D({0,{2,1,1},{1,1,1}},
        1,MxmIcuBufferMode::Fixed,column,{-static_cast<std::int64_t>(column),0,0},
        16,MxmWeightInputMode::Direct16);
    const auto decodedLoad = isa::decode_mxm_load_icu_3d_instruction(
        isa::encode_mxm_load_icu_3d_instruction(loadRun));
    require(decodedLoad.weight_column_base == column
        && decodedLoad.weight_column_strides[0] == -static_cast<std::int64_t>(column)
        && decodedLoad.weight_stream_base == 16 && decodedLoad.weight_input_mode == MxmWeightInputMode::Direct16,
        "ICU load fields overlapped when the column selector widened");
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
    const auto stream = MxmControlInstruction::DecodeStreamCompute(0,0,
        MxmDataFormat::BFloat16,address,column,MxmAccumulatorDestination::Stream,false,MxmDecodeLayout::Native4x4);
    const auto decodedStream = isa::decode_mxm_instruction(isa::encode_mxm_instruction(stream));
    require(decodedStream.accumulator_address == address && decodedStream.weight_column == column
        && decodedStream.data_format == MxmDataFormat::BFloat16 && !decodedStream.accumulator_clear,
        "FU stream-compute fields overlapped after widening");
    const auto half = static_cast<std::int64_t>(hw::kMxmAccumulatorRows / 2);
    auto run = MxmComputeIcuInstruction::Compute3D({0,{1,2,1},{1,3,1}},
        1,MxmIcuBufferMode::Fixed,0,0,half,{0,-half,0},1,MxmDataFormat::BFloat16,
        {MxmAccumulatorDestination::Sram,false,MxmAccumulatorOutputFormat::BFloat16});
    const auto packet = isa::encode_mxm_compute_icu_3d_instruction(run);
    const auto decodedRun = isa::decode_mxm_compute_icu_3d_instruction(packet);
    require(decodedRun.accumulator_address_base == static_cast<std::size_t>(half)
        && decodedRun.accumulator_address_strides[1] == -half,
        "ICU lost a high accumulator base or negative stride");
    run.accumulator_address_base = 0;
    run.accumulator_address_strides[1] = half;
    const auto positive = isa::decode_mxm_compute_icu_3d_instruction(
        isa::encode_mxm_compute_icu_3d_instruction(run));
    require(positive.accumulator_address_strides[1] == half,
        "ICU truncated a positive stride at the old signed boundary");
    using Queue = DistributedIcuQueue<MxmControlInstruction,128,64,16,1,1,IcuQueueRole::MxmCompute>;
    Queue queue;
    queue.push_mxm_compute_3d(run);
    std::vector<std::size_t> issued;
    for (std::size_t cycle=0; cycle<6; ++cycle)
        if (const auto instruction=queue.tick()) issued.push_back(instruction->accumulator_address);
    require(issued == std::vector<std::size_t>{0,static_cast<std::size_t>(half)},
        "wide ICU packet did not decode and issue both address domains");
    require(queue.peak_active_3d_contexts() == 1 && queue.active_3d_context_count() == 0,
        "wide ICU packet did not retire its sole active context");
    const auto readRun = MxmComputeIcuInstruction::AccumulatorRead3D({0,{1,1,1},{1,1,1}},0,address,{0,0,0});
    require(isa::decode_mxm_compute_icu_3d_instruction(
        isa::encode_mxm_compute_icu_3d_instruction(readRun)).accumulator_address_base == address,
        "ICU accumulator-read lost its highest address");
    const auto streamRun = MxmComputeIcuInstruction::DecodeStreamCompute3D(
        {0,{1,1,1},{1,1,1}},0,MxmIcuBufferMode::Fixed,0,MxmDataFormat::BFloat16,
        address,{0,0,0},column,MxmAccumulatorDestination::Stream,false,MxmDecodeLayout::Native4x4);
    const auto decodedStreamRun = isa::decode_mxm_compute_icu_3d_instruction(
        isa::encode_mxm_compute_icu_3d_instruction(streamRun));
    require(decodedStreamRun.accumulator_column == column && decodedStreamRun.accumulator_address_base == address,
        "ICU stream-compute lost its widened fields");
    MxmAccumulator accumulator;
    MxmAccumulator::Segment segment{};
    segment.fill(2.5f);
    accumulator.accumulate(address,column,segment);
    require(accumulator.read(address,column)==segment && accumulator.value(address,hw::kMxmColumns-1)==2.5f,
        "last context/column is not physically backed by accumulator storage");
    bool rejected = false;
    try { accumulator.accumulate(hw::kMxmAccumulatorRows, column, segment); }
    catch (const std::out_of_range&) { rejected = true; }
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
        << " compute_words=" << isa::EncodedMxmComputeIcu3DPacket::kWordCount << " c2c_beats=" << beats << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';return 1;
}
