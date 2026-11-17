#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/c2c_weight_pager.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/weight_page_builder.hpp"
#include "ftlpu/system/c2c_dma_system.hpp"

#include <array>
#include <iostream>
#include <memory>

using namespace ftlpu;
namespace rt = ftlpu::software::runtime;

void print_state(const char* name, C2cDmaSystem& system,
    const rt::CModelRuntime& runtime, std::uint32_t event,
    const rt::BinaryProgram& program)
{
    std::cout << name
              << " cycle=" << system.cycle()
              << " runtime_physical=" << runtime.physical_cycles()
              << " runtime_logical=" << runtime.logical_cycles()
              << " runtime_done=" << runtime.kernel_done()
              << " loader_state=" << static_cast<unsigned>(system.program_loader().state())
              << " loader_done=" << system.program_loader().kernel_done()
              << " queues_done=" << system.chip().icu().program_queues_done()
              << " barriers_done=" << system.chip().icu().common_barriers_done()
              << " resources_drained=" << system.chip().program_resources_drained()
              << " ddr_drained=" << system.ddr4().drained()
              << " east_dma_idle=" << system.dma(Hemisphere::East).idle()
              << " west_dma_idle=" << system.dma(Hemisphere::West).idle()
              << " transport_launched=" << runtime.common_transport_launched(event)
              << " transport_committed=" << runtime.common_transport_committed(event)
              << '\n';
    system.program_loader().statistics().print(std::cout);
    for (std::size_t side = 0; side < hw::kHemispheres; ++side) {
        const auto hemisphere = static_cast<Hemisphere>(side);
        const auto& dma = system.chip().icu().c2c_dma_iq(hemisphere);
        const auto& rx = system.chip().icu().c2c_rx_iq(hemisphere);
        std::cout << name << " c2c_side=" << side
                  << " dma_done=" << dma.done()
                  << " dma_cycle=" << dma.cycle()
                  << " dma_issued=" << dma.issued_count()
                  << " dma_action=" << static_cast<unsigned>(dma.last_trace().action)
                  << " rx_done=" << rx.done()
                  << " rx_cycle=" << rx.cycle()
                  << " rx_issued=" << rx.issued_count()
                  << " rx_action=" << static_cast<unsigned>(rx.last_trace().action)
                  << '\n';
    }
    for (const auto& queue : program.transport_queues) {
        const auto& transport = system.chip().icu().mem_transport_iq(queue.index);
        const auto hemisphere = static_cast<Hemisphere>(
            queue.index / InstructionControlUnit::kMemQueuesPerHemisphere);
        const auto local = queue.index
            % InstructionControlUnit::kMemQueuesPerHemisphere;
        const auto committed = system.chip().mem_array(hemisphere)
            .transport_committed_segments(runtime.execution_generation(), event,
                local / hw::kMemBanksPerSlice,
                local % hw::kMemBanksPerSlice);
        std::cout << name << " mem_transport=" << queue.index
                  << " done=" << transport.done()
                  << " cycle=" << transport.cycle()
                  << " issued=" << transport.issued_count()
                  << " action=" << static_cast<unsigned>(transport.last_trace().action)
                  << " committed_segments=" << committed
                  << '\n';
    }
    for (const auto& queue : program.queues) {
        const auto emit = [&](const auto& device_queue) {
            std::cout << name
                      << " program_kind=" << static_cast<unsigned>(queue.kind)
                      << " index=" << queue.index
                      << " done=" << device_queue.done()
                      << " cycle=" << device_queue.cycle()
                      << " issued=" << device_queue.issued_count()
                      << " action="
                      << static_cast<unsigned>(device_queue.last_trace().action)
                      << '\n';
        };
        auto& icu = system.chip().icu();
        switch (queue.kind) {
        case rt::QueueKind::Mem: emit(icu.mem_iq(queue.index)); break;
        case rt::QueueKind::MxmLoad: emit(icu.mxm_load_iq(queue.index)); break;
        case rt::QueueKind::MxmCompute: emit(icu.mxm_compute_iq(queue.index)); break;
        case rt::QueueKind::MxmDequant: emit(icu.mxm_dequant_iq(queue.index)); break;
        case rt::QueueKind::Vxm: emit(icu.vxm_iq(queue.index)); break;
        case rt::QueueKind::SxmTranspose:
            emit(icu.sxm_transpose_iq(static_cast<Hemisphere>(queue.index)));
            break;
        case rt::QueueKind::SxmPermute:
            emit(icu.sxm_permute_iq(static_cast<Hemisphere>(queue.index)));
            break;
        case rt::QueueKind::C2cDma:
            emit(icu.c2c_dma_iq(static_cast<Hemisphere>(queue.index)));
            break;
        case rt::QueueKind::C2cTx:
            emit(icu.c2c_tx_iq(static_cast<Hemisphere>(queue.index)));
            break;
        case rt::QueueKind::C2cRx:
            emit(icu.c2c_rx_iq(static_cast<Hemisphere>(queue.index)));
            break;
        }
    }
}

int main() try
{
    rt::BinaryBinding binding;
    binding.index = 0;
    binding.access = rt::BindingAccess::Input;
    binding.element_type = rt::BindingElementType::I8;
    binding.layout = rt::BindingLayout::W8A16AttentionWeightStriped;
    binding.shape = {32, 128};
    binding.byte_size = 4096;
    binding.slices = {20,21,22,23,24,25,26,27};
    binding.hemisphere_mask = 3;
    binding.bank = 1;
    binding.paged_weight = true;
    binding.page_count = 1;
    binding.page_rows = 8;
    binding.instruction_count = 8;
    binding.address_stride = 1;
    binding.page_granularity = 1;
    binding.page_role_group_count = 1;
    binding.page_items_per_slice_group = 1;
    binding.page_bank_count = 2;
    binding.page_storage_slices = binding.slices;
    binding.role = "weight";
    binding.name = "test.weight";

    std::vector<std::uint8_t> values(4096);
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = static_cast<std::uint8_t>(i * 13 + 19);
    const auto packed = rt::pack_weight_binding_page(
        binding, 0, values, rt::ExecutableHardwareConfig{});

    auto direct = std::make_unique<C2cDmaSystem>();
    auto system = std::make_unique<C2cDmaSystem>();
    rt::BinaryProgram program;
    program.common_barrier_protocol = true;
    program.max_cycle = 256;
    program.bindings = {binding};
    program.weight_page_uses = {{0,0,1,128,256,true}};
    program.timelines = {{"c2c.window.binding:0.page:0",32,128}};
    rt::C2cWeightPage page;
    page.bank = 1;
    for (const auto& segment : packed.segments)
        page.segments.push_back({static_cast<Hemisphere>(segment.hemisphere),
            segment.slice,1,segment.base_row,0,0x10000+segment.byte_offset,
            segment.vector_count});
    rt::C2cWeightPager pager(*system);
    pager.begin_schedule(program);
    const std::array<std::size_t,1> instances{0};
    pager.schedule(program,page,32,48,128,0x10000,0x30000,32,instances);
    pager.finalize_schedule(program);

    const auto stage = [&](C2cDmaSystem& device) {
        for (std::size_t offset = 0; offset < packed.data.size();
             offset += hw::kPhysicalVectorBytes) {
            C2cVector vector{};
            for (std::size_t b = 0; b < hw::kPhysicalVectorBytes; ++b)
                vector.payload[b / hw::kLanesPerTile][b % hw::kLanesPerTile]
                    = packed.data[offset + b];
            device.ddr4().initialize_vector(0x10000 + offset, vector);
        }
    };
    stage(*direct);
    stage(*system);

    rt::CModelRuntime reference(*direct), timed(*system);
    reference.load_direct(program);
    reference.upload_input(0, values);
    timed.load_timed(program);
    timed.upload_input(0, values);
    const auto event = program.weight_page_uses[0].transport_launch_event_id;

    const auto direct_done = reference.run_until_complete(10000);
    print_state("direct", *direct, reference, event, program);
    const auto timed_done = timed.run_until_complete(10000);
    print_state("timed", *system, timed, event, program);
    std::cout << "result direct_done=" << direct_done
              << " timed_done=" << timed_done << '\n';
    return direct_done && timed_done ? 0 : 2;
} catch (const std::exception& error) {
    std::cerr << "b1_barr_transport_diagnostic failed: " << error.what() << '\n';
    return 1;
}
