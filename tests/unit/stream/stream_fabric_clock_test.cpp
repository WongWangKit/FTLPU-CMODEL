#include "ftlpu/core/stream_fabric.hpp"
#include "ftlpu/core/timing.hpp"

#include <cassert>
#include <cstdint>

namespace {

using namespace ftlpu;

class DualFabricScenario {
public:
    explicit DualFabricScenario(bool reverse_commit_registration)
        : east_(2), west_(2)
    {
        if (reverse_commit_registration) {
            clock_.register_module(west_);
            clock_.register_module(east_);
        } else {
            clock_.register_module(east_);
            clock_.register_module(west_);
        }
        phases_.add_node("open both hemisphere fabrics", [this] {
            east_.low();
            west_.low();
            ++low_calls_;
        }, [this] {
            east_.high();
            west_.high();
            ++high_calls_;
        });
        phases_.add_node("stage east producer", [this] {
            east_.stage_write(0, 0, 0, StreamId::East(0),
                StreamCell::Valid(east_value_), "east test producer");
            ++stage_calls_;
        });
        phases_.add_node("stage west producer", [this] {
            west_.stage_write(0, 0, 0, StreamId::West(0),
                StreamCell::Valid(west_value_), "west test producer");
            ++stage_calls_;
        });
    }

    void run(std::uint8_t east_value, std::uint8_t west_value)
    {
        east_value_ = east_value;
        west_value_ = west_value;
        phases_.low();
        assert(!east_.cell(0, 0, 0, StreamId::East(0)).valid);
        assert(!west_.cell(0, 0, 0, StreamId::West(0)).valid);
        clock_.complete_low();
        clock_.validate_all();
        clock_.commit_all();

        const auto east_postedge = east_.cell(0, 0, 0, StreamId::East(0));
        const auto west_postedge = west_.cell(0, 0, 0, StreamId::West(0));
        phases_.high();
        assert(east_.cell(0, 0, 0, StreamId::East(0)).data == east_postedge.data);
        assert(west_.cell(0, 0, 0, StreamId::West(0)).data == west_postedge.data);
        clock_.complete_high();
        clock_.negedge_all();

        assert(east_postedge.valid && east_postedge.data == east_value);
        assert(west_postedge.valid && west_postedge.data == west_value);
        assert(east_.cycle() == clock_.cycle());
        assert(west_.cycle() == clock_.cycle());
        assert(low_calls_ == 1 && high_calls_ == 1 && stage_calls_ == 2);
    }

    const StreamRegisterFabric& east() const { return east_; }
    const StreamRegisterFabric& west() const { return west_; }

private:
    StreamRegisterFabric east_;
    StreamRegisterFabric west_;
    timing::Clock clock_{};
    timing::SystemPhase phases_{};
    std::uint8_t east_value_{};
    std::uint8_t west_value_{};
    std::size_t low_calls_{};
    std::size_t high_calls_{};
    std::size_t stage_calls_{};
};

} // namespace

int main()
{
    DualFabricScenario forward(false);
    DualFabricScenario reverse(true);
    forward.run(0x31, 0x72);
    reverse.run(0x31, 0x72);
    assert(forward.east().cell(0, 0, 0, ftlpu::StreamId::East(0)).data
        == reverse.east().cell(0, 0, 0, ftlpu::StreamId::East(0)).data);
    assert(forward.west().cell(0, 0, 0, ftlpu::StreamId::West(0)).data
        == reverse.west().cell(0, 0, 0, ftlpu::StreamId::West(0)).data);

    // Validation failure in the second module occurs before either module can
    // commit, proving the global barrier avoids a predictable partial edge.
    ftlpu::StreamRegisterFabric prepared(1);
    ftlpu::StreamRegisterFabric missing(1);
    ftlpu::timing::Clock barrier_clock;
    barrier_clock.register_module(prepared);
    barrier_clock.register_module(missing);
    prepared.low();
    prepared.stage_write(0, 0, 0, ftlpu::StreamId::East(0),
        ftlpu::StreamCell::Valid(9), "barrier test");
    barrier_clock.complete_low();
    bool rejected = false;
    try { barrier_clock.validate_all(); }
    catch (const std::logic_error&) { rejected = true; }
    assert(rejected && barrier_clock.faulted());
    assert(!prepared.cell(0, 0, 0, ftlpu::StreamId::East(0)).valid);
    return 0;
}

