#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/cmodel_runtime.hpp"
#include "ftlpu/software/runtime/icu_program.hpp"
#include "ftlpu/system/c2c_dma_system.hpp"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace rt = ftlpu::software::runtime;

namespace {

constexpr std::uint64_t kImageBase = 0x1'0000'0000ULL;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main(int argc, char** argv) try
{
    if (argc != 2 && argc != 3)
        throw std::runtime_error(
            "usage: b1_binary_delivery_adapter program.ftlpu [--execute]");
    const bool execute = argc == 3 && std::string_view(argv[2]) == "--execute";
    if (argc == 3 && !execute)
        throw std::runtime_error("the only optional argument is --execute");

    const std::filesystem::path binary_path = argv[1];
    const auto program = rt::read_binary_program(binary_path);
    const auto expected_images = rt::prepare_program_images(program, kImageBase, true);

    auto system = std::make_unique<ftlpu::C2cDmaSystem>();
    rt::CModelRuntime runtime(*system);
    runtime.load_timed(program, kImageBase);

    auto& loader = system->program_loader();
    if (execute) {
        require(runtime.run_until_complete(10'000),
            "compiler Binary did not complete after timed delivery");
    } else {
        // Hold the execution dependency low so this mode observes delivery and
        // PRIME without allowing any FU issue.
        loader.set_dependencies_ready(false);
        loader.start();
        while (!loader.ready_to_start()) {
            require(system->cycle() < 1'000'000, "timed binary delivery timed out");
            system->tick();
        }
    }

    require(runtime.all_images_committed(), "not all program images committed");
    require(loader.statistics().frontend_ready.has_value(), "frontend did not become ready");
    if (!execute)
        require(!loader.statistics().program_start.has_value(),
            "delivery-only adapter unexpectedly started FU execution");
    require(expected_images.size() == loader.statistics().images.size(),
        "ProgramImage count changed during timed delivery");

    for (std::size_t index = 0; index < expected_images.size(); ++index) {
        const auto& expected = expected_images[index];
        require(system->chip().icu().read_program_image(expected.descriptor) == expected.bytes,
            "local iMEM readback differs from compiler binary image");
        const auto& image = loader.statistics().images[index];
        std::cout << "image=" << index
                  << " storage=" << static_cast<unsigned>(expected.descriptor.storage)
                  << " kind=" << static_cast<unsigned>(expected.descriptor.kind)
                  << " queue=" << expected.descriptor.queue_index
                  << " words=" << expected.descriptor.word_count()
                  << " effective_bytes=" << image.effective_bytes
                  << " dma_bytes=" << image.dma_bytes
                  << " source_done=" << image.source_done.value_or(0)
                  << " committed=" << image.image_committed.value_or(0) << '\n';
    }

    const auto& stats = loader.statistics();
    std::cout << "binary=" << binary_path.string() << '\n'
              << "queues=" << program.queues.size()
              << " transport_queues=" << program.transport_queues.size()
              << " max_cycle=" << program.max_cycle << '\n'
              << "request_accepted=" << stats.request_accepted.value_or(0)
              << " first_ddr_request=" << stats.first_ddr_request.value_or(0)
              << " first_data_return=" << stats.first_data_return.value_or(0)
              << " all_images_committed=" << stats.all_images_committed.value_or(0)
              << " frontend_ready=" << stats.frontend_ready.value_or(0) << '\n'
              << "effective_bytes=" << stats.effective_bytes
              << " dma_bytes=" << stats.dma_bytes
              << " written_words=" << stats.written_words << '\n'
              << "instruction_dma_ar=" << system->program_loader().dma().statistics().ar_handshakes;
    if (execute)
        std::cout << " program_start=" << stats.program_start.value_or(0)
                  << " status=EXECUTION_COMPLETE\n";
    else
        std::cout << " program_start=pending status=DELIVERY_FRONTEND_READY\n";

    if (execute) {
        const auto& final_stats = runtime.program_load_statistics();
        std::cout << "execution_status=COMPLETE"
                  << " physical_cycles=" << runtime.physical_cycles()
                  << " logical_cycles=" << runtime.logical_cycles()
                  << " program_start=" << final_stats.program_start.value_or(0)
                  << " kernel_done=" << final_stats.kernel_done_cycle.value_or(0)
                  << " mem_queue0_issued=" << system->chip().icu().mem_iq(0).issued_count()
                  << " queues_done=" << system->chip().icu().program_queues_done()
                  << " resources_drained=" << system->chip().program_resources_drained()
                  << " numerical_check=not_applicable" << '\n';
    }
    return 0;
} catch (const std::exception& exception) {
    std::cerr << "b1_binary_delivery_adapter failed: " << exception.what() << '\n';
    return 1;
}
