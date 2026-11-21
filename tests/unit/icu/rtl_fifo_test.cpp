#include "ftlpu/icu/rtl_fifo.hpp"

#include <stdexcept>

namespace {

using namespace ftlpu;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

template<class T>
void cycle(RtlSynchronousFifo<T>& fifo, bool write,
    std::optional<T> data, bool read)
{
    fifo.drive(write, std::move(data), read);
    fifo.low();
    fifo.validate_posedge();
    fifo.posedge();
    fifo.high();
}

class RejectingModule final : public timing::ClockedModule {
public:
    void low() override { prepared = true; }
    void validate_posedge() const override
    {
        if (prepared) throw std::logic_error("intentional rejection");
    }
    void posedge() override { ++commits; }
    bool prepared{false};
    std::size_t commits{0};
};

void generic_fifo_contract()
{
    RtlSynchronousFifo<int> fifo(2);
    require(fifo.empty() && !fifo.dequeue_valid(), "reset FIFO not empty");

    fifo.drive(true, 10, false);
    require(fifo.enqueue_fire() && !fifo.dequeue_valid(),
        "empty FIFO unexpectedly bypassed enqueue data");
    fifo.low();
    require(fifo.empty(), "LOW exposed Next occupancy");
    fifo.posedge();
    require(fifo.occupancy() == 1 && fifo.front() == 10,
        "first enqueue did not commit at POSEDGE");

    cycle<int>(fifo, true, 20, false);
    require(fifo.full() && fifo.front() == 10, "FIFO did not fill in order");

    fifo.drive(true, 30, true);
    require(fifo.enqueue_ready() && fifo.enqueue_fire() && fifo.dequeue_fire(),
        "full replacement was not accepted");
    fifo.low(); fifo.posedge();
    require(fifo.full() && fifo.front() == 20,
        "simultaneous full read/write corrupted order or level");

    cycle<int>(fifo, false, std::nullopt, true);
    require(fifo.occupancy() == 1 && fifo.front() == 30,
        "replacement payload or pointer wrap was lost");
    cycle<int>(fifo, false, std::nullopt, true);
    require(fifo.empty(), "FIFO did not drain");

    const auto& stats = fifo.statistics();
    require(stats.accepted_writes == 3 && stats.accepted_reads == 3,
        "FIFO accepted-operation statistics are incorrect");
    require(stats.maximum_occupancy == 2 && stats.full_cycles != 0,
        "FIFO occupancy statistics are incorrect");
}

void relay_registered_ready_contract()
{
    RtlSynchronousFifo<int> relay(2, false);
    cycle<int>(relay, true, 1, false);
    cycle<int>(relay, true, 2, false);
    relay.drive(true, 3, true);
    require(!relay.enqueue_ready() && !relay.enqueue_fire()
            && relay.dequeue_fire(),
        "relay full ready depended on same-cycle dequeue");
    relay.low(); relay.posedge();
    require(relay.occupancy() == 1 && relay.front() == 2,
        "relay full-cycle dequeue behavior is incorrect");
    require(relay.statistics().rejected_writes == 1,
        "relay backpressure was not recorded");
}

void validation_barrier_contract()
{
    RtlSynchronousFifo<int> fifo(2);
    RejectingModule reject;
    timing::Clock clock;
    clock.register_module(fifo);
    clock.register_module(reject);
    fifo.drive(true, 7, false);
    fifo.low();
    reject.low();
    clock.complete_low();
    auto rejected = false;
    try { clock.validate_all(); }
    catch (const std::logic_error&) { rejected = true; }
    require(rejected, "validation barrier accepted rejecting module");
    require(fifo.empty() && reject.commits == 0,
        "FIFO committed before all participants validated");
}

} // namespace

int main()
{
    generic_fifo_contract();
    relay_registered_ready_contract();
    validation_barrier_contract();
    return 0;
}
