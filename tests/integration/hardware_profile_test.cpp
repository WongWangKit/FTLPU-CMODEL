#include "ftlpu/system/tsp_slice_system.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
}

int main() try {
    using namespace ftlpu;
    auto system = std::make_unique<TspSliceSystem>();
    for (std::size_t i = 0; i < hw::kMxmCount; ++i)
        (void)system->mxm_unit(i);
    bool rejected = false;
    try { (void)system->mxm_unit(hw::kMxmCount); }
    catch (const std::out_of_range&) { rejected = true; }
    require(rejected, "physical MXM array is not bounded by the target");
    MxmAccumulator acc;
    MxmAccumulator::Segment values{};
    for (std::size_t i=0; i<values.size(); ++i) values[i] = float(i+1);
    const auto row = hw::kMxmAccumulatorRows-1;
    const auto col = hw::kMxmSupercellsPerPlane-1;
    acc.accumulate(row, col, values);
    acc.accumulate(row, col, values);
    auto result = acc.read(row, col);
    for (std::size_t i=0; i<values.size(); ++i)
        require(result[i] == 2*values[i], "last ACC segment accumulation failed");
    acc.clear_segment(row, col);
    for (float v : acc.read(row, col)) require(v == 0, "ACC clear failed");
    rejected = false;
    try { (void)acc.read(row+1, col); }
    catch (const std::out_of_range&) { rejected = true; }
    require(rejected, "ACC overflow was not rejected");
    Sram bank;
    bank.set_byte(hw::kSramDepthRows-1, hw::kSramRowBytes-1, 0xa5);
    require(bank.byte(hw::kSramDepthRows-1, hw::kSramRowBytes-1) == 0xa5,
        "last SRAM byte roundtrip failed");
    rejected = false;
    try { bank.set_byte(hw::kSramDepthRows, 0, 0); }
    catch (const std::out_of_range&) { rejected = true; }
    require(rejected, "SRAM overflow was not rejected");
    std::cout << "hardware_profile_test passed: lanes=" << hw::kPhysicalVectorBytes
        << " physical_mxms=" << hw::kMxmCount
        << " mem_bytes=" << hw::kPublicTotalSramBytes
        << " acc_bytes_per_mxm=" << hw::kMxmAccumulatorBytes
        << " c2c_beat_bytes=" << hw::kC2cBytesPerStreamPerCycle << '\n';
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
