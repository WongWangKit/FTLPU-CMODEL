#include "ftlpu/icu/rtl_fifo.hpp"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace {

using ftlpu::RtlSynchronousFifo;

struct Stimulus {
    bool enqueue_valid;
    std::uint8_t enqueue_data;
    bool dequeue_ready;
};

constexpr std::array<Stimulus, 12> kStimuli{{
    {true, 10, false},
    {true, 20, false},
    {true, 30, false},
    {true, 30, true},
    {true, 40, true},
    {false, 0, true},
    {false, 0, true},
    {true, 50, true},
    {true, 60, false},
    {true, 70, true},
    {false, 0, true},
    {false, 0, true},
}};

std::uint32_t visible_data(const RtlSynchronousFifo<std::uint8_t>& fifo)
{
    return fifo.dequeue_valid() ? fifo.front() : 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "usage: icu_rtl_fifo_golden_trace <output.csv>\n";
        return 2;
    }

    std::ofstream output(argv[1]);
    if (!output) {
        std::cerr << "cannot open trace output: " << argv[1] << '\n';
        return 2;
    }

    output << "cycle,enqueue_valid,enqueue_data,dequeue_ready,"
              "enqueue_ready_pre,dequeue_valid_pre,enqueue_fire,dequeue_fire,"
              "dequeue_data_pre,level_pre,write_ptr_pre,read_ptr_pre,"
              "level_post,write_ptr_post,read_ptr_post,dequeue_valid_post,"
              "dequeue_data_post\n";

    RtlSynchronousFifo<std::uint8_t> fifo(2);
    for (std::size_t cycle = 0; cycle < kStimuli.size(); ++cycle) {
        const auto stimulus = kStimuli[cycle];
        fifo.drive(stimulus.enqueue_valid,
            stimulus.enqueue_valid
                ? std::optional<std::uint8_t>{stimulus.enqueue_data}
                : std::nullopt,
            stimulus.dequeue_ready);

        const auto enqueue_ready_pre = fifo.enqueue_ready();
        const auto dequeue_valid_pre = fifo.dequeue_valid();
        const auto enqueue_fire = fifo.enqueue_fire();
        const auto dequeue_fire = fifo.dequeue_fire();
        const auto dequeue_data_pre = visible_data(fifo);
        const auto level_pre = fifo.occupancy();
        const auto write_ptr_pre = fifo.write_pointer();
        const auto read_ptr_pre = fifo.read_pointer();

        fifo.low();
        fifo.validate_posedge();
        fifo.posedge();
        fifo.high();

        output << cycle << ',' << stimulus.enqueue_valid << ','
               << static_cast<unsigned>(stimulus.enqueue_data) << ','
               << stimulus.dequeue_ready << ',' << enqueue_ready_pre << ','
               << dequeue_valid_pre << ',' << enqueue_fire << ','
               << dequeue_fire << ',' << dequeue_data_pre << ',' << level_pre
               << ',' << write_ptr_pre << ',' << read_ptr_pre << ','
               << fifo.occupancy() << ',' << fifo.write_pointer() << ','
               << fifo.read_pointer() << ',' << fifo.dequeue_valid() << ','
               << visible_data(fifo) << '\n';
    }

    std::cout << "wrote " << kStimuli.size() << " CModel FIFO trace rows\n";
    return 0;
}
