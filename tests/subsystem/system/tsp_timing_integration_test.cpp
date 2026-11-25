#include "ftlpu/system/tsp_slice_system.hpp"

#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

[[noreturn]] void fail_check(const char* expression, int line)
{
    throw std::runtime_error(
        "check failed at line " + std::to_string(line) + ": " + expression);
}

#undef assert
#define assert(expression) \
    ((expression) ? static_cast<void>(0) : fail_check(#expression, __LINE__))

} // namespace

int main()
{
    try {
        // Construct on the stack deliberately: MxmCore must keep bulk Current/
        // Next state and reset temporaries out of the TSP object's stack footprint.
        using Timing = ftlpu::MemArrayModel::ReadTiming;
        {
        auto system = std::make_unique<ftlpu::TspSliceSystem>();
        assert(system->mem_array(ftlpu::Hemisphere::East).read_timing()
            == Timing::RegisteredR1);
        assert(system->mem_array(ftlpu::Hemisphere::West).read_timing()
            == Timing::RegisteredR1);
        assert(system->clock_phase() == ftlpu::timing::ClockPhase::LOW);
        assert(system->phase_count() == 0 && system->cycle() == 0);

        system->tick(ftlpu::TspSliceSystem::LogSinks {});
        assert(system->clock_phase() == ftlpu::timing::ClockPhase::LOW);
        assert(system->phase_count() == 4 && system->cycle() == 1);
        assert(system->stream_fabric(ftlpu::Hemisphere::East).cycle() == 1);
        assert(system->stream_fabric(ftlpu::Hemisphere::West).cycle() == 1);

        system->tick(ftlpu::TspSliceSystem::LogSinks {});
        assert(system->phase_count() == 8 && system->cycle() == 2);
        assert(system->stream_fabric(ftlpu::Hemisphere::East).cycle()
            == system->stream_fabric(ftlpu::Hemisphere::West).cycle());

        auto mxm_inputs = system->mxm_unit(0).inputs();
        mxm_inputs.run_i = true;
        mxm_inputs.compute_enqueue_valid_i = true;
        mxm_inputs.compute_instruction_i = {3, true};
        system->mxm_unit(0).drive(mxm_inputs);
        system->tick(ftlpu::TspSliceSystem::LogSinks {});
        assert(system->mxm_unit(0).compute_queue_level() == 1);
        assert(system->mxm_unit(0).cycle() == system->cycle());

        system->reset_execution_state();
        assert(system->clock_phase() == ftlpu::timing::ClockPhase::LOW);
        assert(system->phase_count() == 0 && system->cycle() == 0);
        assert(system->stream_fabric(ftlpu::Hemisphere::East).cycle() == 0);
        assert(system->stream_fabric(ftlpu::Hemisphere::West).cycle() == 0);
        assert(system->mxm_unit(0).compute_queue_level() == 0);
        }
        {
        auto mainTimingSystem = std::make_unique<ftlpu::TspSliceSystem>(
            ftlpu::SystemHardwareConfiguration {},
            ftlpu::SxmTransposeArchitecture::LocalBlocks, Timing::Main);
        assert(mainTimingSystem->mem_array(ftlpu::Hemisphere::East).read_timing()
            == Timing::Main);
        assert(mainTimingSystem->mem_array(ftlpu::Hemisphere::West).read_timing()
            == Timing::Main);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tsp_timing_integration_test: " << error.what() << '\n';
        return 1;
    }
}

