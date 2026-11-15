#include "ftlpu/core/timing.hpp"
#include "ftlpu/sxm/slice.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

using namespace ftlpu;

struct Result {
    std::array<std::uint8_t, hw::kTileRows * hw::kLanesPerTile> bytes{};
    std::size_t cycle{};
};

void initialize(StreamRegisterFabric& fabric)
{
    const auto source = StreamId::East(0);
    for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
            fabric.initialize_cell(0, tile, lane, source,
                StreamCell::Valid(static_cast<std::uint8_t>(tile * 16 + lane)));
}

Result run_one_cycle(bool reverse_registration)
{
    StreamRegisterFabric fabric(2);
    SxmSlice sxm(SxmStreamPortMap::SameDirection(0, 1));
    timing::Clock clock;
    timing::SystemPhase phases;

    if (reverse_registration) {
        clock.register_module(sxm);
        clock.register_module(fabric);
    } else {
        clock.register_module(fabric);
        clock.register_module(sxm);
    }
    phases.add_node("fabric", [&] { fabric.low(); }, [&] { fabric.high(); });
    phases.add_node("sxm", [&] { sxm.evaluate(fabric); }, [&] { sxm.high(); });

    initialize(fabric);
    sxm.issue(SxmInstruction::Distribute(
        SxmStreamId {StreamId::East(0).packed()},
        SxmStreamId {StreamId::East(1).packed()},
        Distribute16::identity_map()));

    phases.low();
    clock.complete_low();
    assert(sxm.cycle() == 0);
    assert(fabric.cell(0, 0, 0, StreamId::East(0)).valid);
    assert(!fabric.cell(1, 0, 0, StreamId::East(1)).valid);

    clock.validate_all();
    clock.commit_all();
    assert(sxm.cycle() == 1);
    assert(!fabric.cell(0, 0, 0, StreamId::East(0)).valid);
    assert(fabric.cell(1, 0, 0, StreamId::East(1)).valid);

    phases.high();
    clock.complete_high();
    assert(sxm.cycle() == 1);
    clock.negedge_all();

    Result result{};
    result.cycle = sxm.cycle();
    for (std::size_t tile = 0; tile < hw::kTileRows; ++tile)
        for (std::size_t lane = 0; lane < hw::kLanesPerTile; ++lane)
            result.bytes[tile * hw::kLanesPerTile + lane] =
                fabric.cell(1, tile, lane, StreamId::East(1)).data;
    return result;
}

class RejectingModule final : public timing::ClockedModule {
public:
    void low() override { prepared_ = true; }
    void validate_posedge() const override
    {
        if (!prepared_) throw std::logic_error("rejector LOW missing");
        throw std::logic_error("intentional validation failure");
    }
    void posedge() override { committed_ = true; }
    bool committed() const noexcept { return committed_; }

private:
    bool prepared_{};
    bool committed_{};
};

void validation_barrier_preserves_current()
{
    StreamRegisterFabric fabric(2);
    SxmSlice sxm(SxmStreamPortMap::SameDirection(0, 1));
    RejectingModule rejector;
    timing::Clock clock;
    timing::SystemPhase phases;
    clock.register_module(fabric);
    clock.register_module(sxm);
    clock.register_module(rejector);
    phases.add_node("fabric", [&] { fabric.low(); });
    phases.add_node("sxm", [&] { sxm.evaluate(fabric); });
    phases.add_node("rejector", [&] { rejector.low(); });

    initialize(fabric);
    sxm.issue(SxmInstruction::Distribute(
        SxmStreamId {StreamId::East(0).packed()},
        SxmStreamId {StreamId::East(1).packed()},
        Distribute16::identity_map()));
    phases.low();
    clock.complete_low();
    bool rejected = false;
    try {
        clock.validate_all();
    } catch (const std::logic_error&) {
        rejected = true;
    }
    assert(rejected);
    assert(sxm.cycle() == 0);
    assert(!rejector.committed());
    assert(fabric.cell(0, 0, 0, StreamId::East(0)).valid);
    assert(!fabric.cell(1, 0, 0, StreamId::East(1)).valid);
}

} // namespace

int main()
{
    const auto normal = run_one_cycle(false);
    const auto reversed = run_one_cycle(true);
    assert(normal.cycle == 1 && reversed.cycle == 1);
    assert(normal.bytes == reversed.bytes);
    validation_barrier_preserves_current();
    std::cout << "SXM clocked Current/Next, registration-order and validation-barrier checks passed\n";
}
