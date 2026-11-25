#include "ftlpu/system/c2c_dma_system.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using namespace ftlpu;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

Ddr4Config ddr_config()
{
    Ddr4Config config;
    config.read_latency_cycles = 2;
    config.read_latency_jitter_cycles = 0;
    config.peak_bandwidth_bytes_per_second = config.lpu_clock_hz * 32;
    return config;
}

std::vector<std::uint8_t> single_context_bytes(
    std::uint32_t start_cycle,
    std::uint64_t native_instruction,
    std::size_t native_bits,
    std::size_t operand_bits)
{
    std::array<std::uint32_t, 8> lanes{};
    lanes[0] = 0x15u;
    std::size_t offset = 128;
    auto push = [&](std::uint64_t value, std::size_t width) {
        for (std::size_t bit = 0; bit < width; ++bit, ++offset)
            lanes[offset / 32] |= std::uint32_t((value >> bit) & 1u)
                << (offset % 32);
    };
    push(0, 3);
    push(start_cycle, 32);
    push(0, operand_bits);
    push(0, 1);
    push(0, 1);
    push(native_instruction, native_bits);
    push(0, 2);
    push(0, 2);

    std::vector<std::uint8_t> bytes(32);
    for (std::size_t index = 0; index < bytes.size(); ++index)
        bytes[index] = static_cast<std::uint8_t>(
            lanes[index / 4] >> (8 * (index % 4)));
    return bytes;
}

void stage(C2cDmaSystem& system, std::uint64_t address,
    const std::vector<std::uint8_t>& bytes)
{
    for (std::size_t offset = 0; offset < bytes.size(); offset += 32) {
        C2cVector vector{};
        for (std::size_t byte = 0; byte < 32; ++byte)
            vector.payload[byte / hw::kLanesPerTile]
                [byte % hw::kLanesPerTile] = bytes[offset + byte];
        system.ddr4().initialize_vector(address + offset, vector);
    }
}

} // namespace

int main()
{
    try {
        C2cDmaSystem system(ddr_config(), 8);
        const std::array<IcuMacroQueueKind, 5> kinds{
            IcuMacroQueueKind::MxmLoad,
            IcuMacroQueueKind::MxmCompute,
            IcuMacroQueueKind::MxmDequant,
            IcuMacroQueueKind::MxmParamLoad,
            IcuMacroQueueKind::MxmBackend};
        const std::array<std::uint32_t, 5> start_cycles{3, 4, 3, 6, 7};
        const std::array<std::uint64_t, 5> native{0x2, 0x7, 0x1, 0x1, 0x8};
        const std::array<std::size_t, 5> native_bits{2, 7, 1, 1, 13};
        const std::array<std::size_t, 5> operand_bits{1, 6, 1, 1, 10};

        std::vector<ProgramImage> images;
        for (std::size_t stream = 0; stream < kinds.size(); ++stream) {
            const auto bytes = single_context_bytes(start_cycles[stream],
                native[stream], native_bits[stream], operand_bits[stream]);
            const auto source = std::uint64_t{0x80000} + stream * 0x100;
            stage(system, source, bytes);
            images.push_back(ProgramImage{source, kinds[stream], 0,
                bytes.size(), IcuQueueMode::Macro});
        }

        auto& loader = system.program_loader();
        loader.submit(images, system.cycle());
        loader.start();
        loader.set_dependencies_ready(true);

        std::array<std::size_t, 5> issue_count{};
        for (std::size_t timeout = 0; timeout < 5000; ++timeout) {
            system.tick();
            auto& core = system.chip().mxm_unit(0);
            const auto& in = core.inputs();
            const auto& out = core.pre_posedge_outputs();
            const auto& icu = system.chip().icu();

            if (in.load_enqueue_valid_i) {
                require(out.load_enqueue_fire && out.load_launch_fire,
                    "MXM Load did not use the native direct Fire path");
                require(in.load_instruction_i == MxmLoadCommand{true, false},
                    "MXM Load command changed before MxmCore");
                require(icu.mxm_load_iq(0).macro_issue_port_signals()
                        .issue_cycle_o == 3,
                    "MXM Load Scheduler cycle changed");
                ++issue_count[0];
            }
            if (in.compute_enqueue_valid_i) {
                require(out.compute_enqueue_fire && out.compute_launch_fire,
                    "MXM Compute did not use the native direct Fire path");
                require(in.compute_instruction_i == MxmComputeCommand{3, true},
                    "MXM Compute command changed before MxmCore");
                require(icu.mxm_compute_iq(0).macro_issue_port_signals()
                        .issue_cycle_o == 4,
                    "MXM Compute Scheduler cycle changed");
                ++issue_count[1];
            }
            if (in.dequant_enqueue_valid_i) {
                require(out.dequant_enqueue_fire && out.dequant_launch_fire,
                    "MXM Dequant did not use the native direct Fire path");
                require(in.dequant_instruction_i == MxmDequantCommand{true},
                    "MXM Dequant command changed before MxmCore");
                require(icu.mxm_dequant_iq(0).macro_issue_port_signals()
                        .issue_cycle_o == 3,
                    "MXM Dequant Scheduler cycle changed");
                ++issue_count[2];
            }
            if (in.param_load_enqueue_valid_i) {
                require(out.param_load_enqueue_fire
                        && out.param_load_launch_fire,
                    "MXM Param Load did not use the native direct Fire path");
                require(in.param_load_instruction_i
                        == MxmParamLoadCommand{true},
                    "MXM Param Load command changed before MxmCore");
                require(icu.mxm_param_load_iq(0).macro_issue_port_signals()
                        .issue_cycle_o == 6,
                    "MXM Param Load Scheduler cycle changed");
                ++issue_count[3];
            }
            if (in.backend_enqueue_valid_i) {
                require(out.backend_enqueue_fire && out.backend_launch_fire,
                    "MXM Backend did not use the native direct Fire path");
                require(in.backend_instruction_i == MxmBackendCommand{
                        false, 2, MxmBackendOperation::DirectOut},
                    "MXM Backend command changed before MxmCore");
                require(icu.mxm_backend_iq(0).macro_issue_port_signals()
                        .issue_cycle_o == 7,
                    "MXM Backend Scheduler cycle changed");
                ++issue_count[4];
            }

            if (issue_count == std::array<std::size_t, 5>{1, 1, 1, 1, 1}) {
                require(core.compute_queue_level() == 0
                        && core.load_queue_level() == 0
                        && core.param_load_queue_level() == 0
                        && core.dequant_queue_level() == 0
                        && core.backend_queue_level() == 0,
                    "Scheduler-to-Core direct Fire retained an extra command");
                std::cout << "MXM_FULL_INSTRUCTION_PATH_PASS streams=5"
                          << " program_load=1 direct_fire=5\n";
                return 0;
            }
        }
        std::cerr << "timeout loader_state="
                  << static_cast<unsigned>(loader.state())
                  << " started=" << loader.started()
                  << " issue_counts=" << issue_count[0] << ','
                  << issue_count[1] << ',' << issue_count[2] << ','
                  << issue_count[3] << ',' << issue_count[4]
                  << " decoder_faults="
                  << system.chip().icu().mxm_load_iq(0).macro_decoder_fault()
                  << system.chip().icu().mxm_compute_iq(0).macro_decoder_fault()
                  << system.chip().icu().mxm_dequant_iq(0).macro_decoder_fault()
                  << system.chip().icu().mxm_param_load_iq(0).macro_decoder_fault()
                  << system.chip().icu().mxm_backend_iq(0).macro_decoder_fault()
                  << " cycles="
                  << system.chip().icu().mxm_load_iq(0).cycle() << ','
                  << system.chip().icu().mxm_compute_iq(0).cycle() << ','
                  << system.chip().icu().mxm_dequant_iq(0).cycle() << ','
                  << system.chip().icu().mxm_param_load_iq(0).cycle() << ','
                  << system.chip().icu().mxm_backend_iq(0).cycle() << '\n';
        loader.statistics().print(std::cerr);
        const auto& load_view = loader.program_load_hardware_interface_view();
        std::cerr << "program_fault=" << load_view.program_fault_o
                  << " code=" << static_cast<unsigned>(
                         load_view.program_fault_code_o)
                  << " load_active=" << load_view.load_request_active
                  << " pipeline_idle=" << load_view.load_pipeline_idle
                  << " distribution_idle=" << load_view.distribution_idle
                  << " word_fifo=" << load_view.r2.word_fifo_level_o << '\n';
        throw std::runtime_error(
            "five-stream Program Load did not reach all native MxmCore ports");
    } catch (const std::exception& error) {
        std::cerr << "mxm_full_instruction_path_test failed: "
                  << error.what() << '\n';
        return 1;
    }
}
