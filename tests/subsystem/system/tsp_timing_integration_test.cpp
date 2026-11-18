#include "ftlpu/system/tsp_slice_system.hpp"

#include <cassert>

int main()
{
    using Timing = ftlpu::MemArrayModel::ReadTiming;
    {
        ftlpu::TspSliceSystem system;
        assert(system.mem_array(ftlpu::Hemisphere::East).read_timing()
            == Timing::RegisteredR1);
        assert(system.mem_array(ftlpu::Hemisphere::West).read_timing()
            == Timing::RegisteredR1);
        assert(system.clock_phase() == ftlpu::timing::ClockPhase::LOW);
        assert(system.phase_count() == 0 && system.cycle() == 0);

        system.tick(ftlpu::TspSliceSystem::LogSinks {});
        assert(system.clock_phase() == ftlpu::timing::ClockPhase::LOW);
        assert(system.phase_count() == 4 && system.cycle() == 1);
        assert(system.stream_fabric(ftlpu::Hemisphere::East).cycle() == 1);
        assert(system.stream_fabric(ftlpu::Hemisphere::West).cycle() == 1);

        system.tick(ftlpu::TspSliceSystem::LogSinks {});
        assert(system.phase_count() == 8 && system.cycle() == 2);
        assert(system.stream_fabric(ftlpu::Hemisphere::East).cycle()
            == system.stream_fabric(ftlpu::Hemisphere::West).cycle());

        system.reset_execution_state();
        assert(system.clock_phase() == ftlpu::timing::ClockPhase::LOW);
        assert(system.phase_count() == 0 && system.cycle() == 0);
        assert(system.stream_fabric(ftlpu::Hemisphere::East).cycle() == 0);
        assert(system.stream_fabric(ftlpu::Hemisphere::West).cycle() == 0);
    }
    {
        ftlpu::TspSliceSystem mainTimingSystem(
            {}, ftlpu::SxmTransposeArchitecture::LocalBlocks, Timing::Main);
        assert(mainTimingSystem.mem_array(ftlpu::Hemisphere::East).read_timing()
            == Timing::Main);
        assert(mainTimingSystem.mem_array(ftlpu::Hemisphere::West).read_timing()
            == Timing::Main);
    }
    return 0;
}

