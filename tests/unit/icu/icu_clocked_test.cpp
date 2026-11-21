#include "ftlpu/core/timing.hpp"
#include "ftlpu/icu/instruction.hpp"
#include "ftlpu/icu/location.hpp"
#include "ftlpu/system/icu.hpp"
#include "ftlpu/vxm/slice.hpp"

#include <cstddef>
#include <stdexcept>

namespace {

using namespace ftlpu;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

class RejectingModule final : public timing::ClockedModule {
public:
    void low() override { prepared = true; }
    void validate_posedge() const override
    {
        if (prepared) throw std::logic_error("intentional validation failure");
    }
    void posedge() override { ++commits; }

    bool prepared{};
    std::size_t commits{};
};

void complete_icu_cycle(InstructionControlUnit& icu)
{
    icu.advance_barrier_events();
    icu.validate_posedge();
    icu.posedge();
}

void test_dispatch_changes_current_only_at_posedge()
{
    auto icu = InstructionControlUnit {};
    auto vxm = VxmSlice {};
    icu.enqueue_vxm_nop(0, 1);
    icu.enable_clocked_mode();

    icu.dispatch_vxm(vxm);
    require(icu.cycle() == 0 && icu.vxm_iq(0).cycle() == 0,
        "ICU exposed queue or global Next state during LOW");
    complete_icu_cycle(icu);
    require(icu.cycle() == 1 && icu.vxm_iq(0).cycle() == 1,
        "ICU did not publish its prepared state at POSEDGE");

    icu.high();
    require(icu.cycle() == 1 && icu.vxm_iq(0).cycle() == 1,
        "ICU HIGH repeated queue progression");
}

void test_external_notification_is_sampled_into_next()
{
    auto icu = InstructionControlUnit {};
    auto vxm = VxmSlice {};
    icu.enqueue_control(
        IcuLocation::Vxm(0), IcuControlInstruction::Sync());
    icu.enable_clocked_mode();

    icu.dispatch_vxm(vxm);
    complete_icu_cycle(icu);
    require(icu.vxm_iq(0).blocked_on_sync(),
        "ICU Sync did not enter its wait state");

    icu.notify(IcuLocation::Vxm(0));
    require(icu.vxm_iq(0).blocked_on_sync(),
        "external notification changed ICU Current before LOW");
    icu.dispatch_vxm(vxm);
    icu.advance_barrier_events();
    require(icu.vxm_iq(0).blocked_on_sync(),
        "external notification changed ICU Current before POSEDGE");
    icu.validate_posedge();
    icu.posedge();
    require(icu.vxm_iq(0).done(),
        "sampled notification did not release Sync at POSEDGE");
}

void test_validation_barrier_prevents_partial_icu_commit()
{
    auto icu = InstructionControlUnit {};
    auto vxm = VxmSlice {};
    auto reject = RejectingModule {};
    auto clock = timing::Clock {};
    icu.enqueue_vxm_nop(0, 1);
    icu.enable_clocked_mode();

    // Register ICU first to prove validate-all completes before any commit.
    clock.register_module(icu);
    clock.register_module(reject);
    icu.dispatch_vxm(vxm);
    reject.low();
    clock.complete_low();

    auto rejected = false;
    try {
        clock.validate_all();
    } catch (const std::logic_error&) {
        rejected = true;
    }
    require(rejected, "Clock accepted an intentionally invalid barrier");
    require(icu.cycle() == 0 && icu.vxm_iq(0).cycle() == 0,
        "ICU committed before every Clock participant validated");
    require(reject.commits == 0,
        "Clock committed another participant after validation failed");
}

} // namespace

int main()
{
    test_dispatch_changes_current_only_at_posedge();
    test_external_notification_is_sampled_into_next();
    test_validation_barrier_prevents_partial_icu_commit();
    return 0;
}
