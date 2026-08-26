#include "ftlpu/system/tsp_slice_system.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

template <typename Fn>
void require_out_of_range(Fn&& fn, const char* message)
{
    try {
        fn();
    } catch (const std::out_of_range&) {
        return;
    }
    throw std::logic_error(message);
}

} // namespace

int main()
try {
    ftlpu::SystemHardwareConfiguration compact;
    compact.sram_depth_rows = 1024;
    compact.mxms_per_hemisphere = 1;
    compact.mxm_weight_buffers = 1;
    compact.vxm_alus = 8;
    compact.mxm_local_dequant_enabled = false;
    compact.mxm_weight_activation_overlap_enabled = false;

    ftlpu::TspSliceSystem system(compact);
    const auto& active = system.hardware_configuration();
    if (active.sram_depth_rows != 1024
        || active.mxms_per_hemisphere != 1
        || active.mxm_weight_buffers != 1
        || active.vxm_alus != 8
        || active.mxm_local_dequant_enabled
        || active.mxm_weight_activation_overlap_enabled)
        throw std::logic_error("compact hardware configuration was not retained");

    system.initialize_mem_sram_lane_byte(0, 0, 1023, 0, 0x5a);
    if (system.read_mem_sram_lane_byte(0, 0, 1023, 0) != 0x5a)
        throw std::logic_error("configured SRAM range did not retain data");
    require_out_of_range(
        [&] { system.initialize_mem_sram_lane_byte(0, 0, 1024, 0, 0); },
        "configured SRAM depth was not enforced");

    (void)system.mxm_unit(0);
    (void)system.mxm_unit(2);
    require_out_of_range(
        [&] { (void)system.mxm_unit(1); },
        "disabled east-hemisphere MXM remained addressable");
    require_out_of_range(
        [&] { (void)system.mxm_unit(3); },
        "disabled west-hemisphere MXM remained addressable");

    auto full = ftlpu::SystemHardwareConfiguration {};
    system.configure_hardware(full);
    (void)system.mxm_unit(1);
    (void)system.mxm_unit(3);
    system.initialize_mem_sram_lane_byte(
        0, 0, ftlpu::hw::kSramDepthRows - 1, 0, 0xa5);

    std::cout << "hardware configuration test passed\n";
    return 0;
} catch (const std::exception& ex) {
    std::cerr << "hardware_configuration_test failed: " << ex.what() << '\n';
    return 1;
}
