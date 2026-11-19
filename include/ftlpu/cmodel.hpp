#pragma once
// Current hardware-model entry point. Include system/tsp_slice_system.hpp
// explicitly when running an archived whole-chip schedule with the legacy ISA.
#include "ftlpu/rtl/model.hpp"
namespace ftlpu {
using CModel = rtl::Model;
}
