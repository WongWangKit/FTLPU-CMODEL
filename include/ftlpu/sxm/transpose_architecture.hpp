#pragma once
#include <stdexcept>
#include <string_view>
namespace ftlpu {
enum class SxmTransposeArchitecture { Fifo };
// "wide" is a historical CLI spelling for a full-matrix stream operation.
// It selects the same FIFO datapath. A local/block executable needs recompilation.
inline SxmTransposeArchitecture sxm_architecture_from_name(const char* name) {
    if (!name || std::string_view(name)=="fifo" || std::string_view(name)=="wide")
        return SxmTransposeArchitecture::Fifo;
    throw std::invalid_argument("SXM architecture must be fifo");
}
}
