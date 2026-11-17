#include "ftlpu/software/runtime/binary.hpp"
#include "ftlpu/software/runtime/icu_program.hpp"
#include "ftlpu/software/runtime/macro_bitstream.hpp"
#include "ftlpu/system/c2c_dma_system.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>

using namespace ftlpu;
namespace rt = ftlpu::software::runtime;

int main()
{
    try {
        rt::BinaryProgram program;
        for (unsigned kind = 0; kind < 10; ++kind) {
            rt::QueueProgram queue{static_cast<rt::QueueKind>(kind), 0, {}};
            for (unsigned i = 0; i < 5; ++i)
                queue.commands.push_back(rt::encode_icu_control_raw_word(
                    IcuControlInstruction::Nop(3)));
            if (queue.kind == rt::QueueKind::C2cDma) {
                const auto packet = rt::encode_c2c_dma_icu_packet(
                    C2cIcuPacketCodec::encode(C2cDmaIcuInstruction::Load(
                        Hemisphere::East, 0, 0x8000, 2)));
                queue.commands.insert(
                    queue.commands.begin() + 1, packet.begin(), packet.end());
            }
            program.queues.push_back(std::move(queue));
        }
        program.transport_queues = {{rt::QueueKind::Mem, 1,
            {rt::encode_icu_control_raw_word(IcuControlInstruction::Nop(4)),
             rt::encode_icu_control_raw_word(IcuControlInstruction::Nop(2)),
             rt::encode_icu_control_raw_word(IcuControlInstruction::Nop(1))}}};

        const auto images = rt::prepare_program_images(program, 0x10000);
        auto system = std::make_unique<C2cDmaSystem>();
        std::vector<ProgramImage> descriptors;
        for (const auto& image : images) {
            descriptors.push_back(image.descriptor);
            for (std::size_t offset = 0; offset < image.bytes.size();
                 offset += hw::kPhysicalVectorBytes) {
                C2cVector vector{};
                for (std::size_t b = 0;
                     b < hw::kPhysicalVectorBytes && offset + b < image.bytes.size();
                     ++b)
                    vector.payload[b / hw::kLanesPerTile][b % hw::kLanesPerTile]
                        = image.bytes[offset + b];
                system->ddr4().initialize_vector(
                    image.descriptor.source_address + offset, vector);
            }
        }

        system->program_loader().submit(std::move(descriptors), 0);
        auto& loader = system->program_loader();
        loader.start();
        loader.set_dependencies_ready(true);
        while (!loader.ready_to_start()) {
            if (system->cycle() >= 5000)
                throw std::runtime_error("all-format load timed out");
            system->tick();
        }

        std::size_t expected_effective = 0;
        std::size_t expected_dma = 0;
        std::size_t expected_words = 0;
        for (std::size_t i = 0; i < images.size(); ++i) {
            const auto& image = images[i];
            expected_effective += image.bytes.size();
            expected_dma += image.descriptor.vector_count()
                * hw::kPhysicalVectorBytes;
            expected_words += image.descriptor.word_count();
            const auto& actual = loader.statistics().images.at(i);
            std::cout << "image=" << i
                      << " storage=" << static_cast<unsigned>(image.descriptor.storage)
                      << " kind=" << static_cast<unsigned>(image.descriptor.kind)
                      << " bytes=" << image.bytes.size()
                      << " words=" << image.descriptor.word_count()
                      << " vectors=" << image.descriptor.vector_count()
                      << " stats_effective=" << actual.effective_bytes
                      << " stats_dma=" << actual.dma_bytes
                      << " stats_words=" << actual.written_words
                      << " first_ddr="
                      << (actual.first_ddr_request
                              ? std::to_string(*actual.first_ddr_request)
                              : std::string("pending"))
                      << " source_done="
                      << (actual.source_done
                              ? std::to_string(*actual.source_done)
                              : std::string("pending"))
                      << " image_committed="
                      << (actual.image_committed
                              ? std::to_string(*actual.image_committed)
                              : std::string("pending")) << '\n';
        }

        const auto& stats = loader.statistics();
        std::cout << "expected effective=" << expected_effective
                  << " dma=" << expected_dma
                  << " words=" << expected_words << '\n';
        std::cout << "actual effective=" << stats.effective_bytes
                  << " dma=" << stats.dma_bytes
                  << " words=" << stats.written_words << '\n';
        std::cout << "instruction_dma ar_handshakes="
                  << loader.dma().statistics().ar_handshakes
                  << " bytes="
                  << loader.dma().statistics().ar_handshakes
                        * hw::kPhysicalVectorBytes << '\n';
        stats.print(std::cout);
        return stats.effective_bytes == expected_effective
                && stats.dma_bytes == expected_dma
                && stats.written_words == expected_words
            ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "b1_program_load_diagnostic failed: " << error.what() << '\n';
        return 1;
    }
}
