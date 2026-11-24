#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
namespace ftlpu {
// One active zero-based 96-bit MEM Macro image. Data storage belongs to the
// private iMEM; this tracks the three transactions, not a second image.
struct MemProgramEndpoint {
    enum class Error { None, Write, Commit, Prime };
    std::size_t accepted{}, committed{};
    bool last_seen{}, primed{};
    Error error{};
    bool write(std::size_t addr, unsigned strb, bool last, std::size_t count) {
        if (error!=Error::None) return false;
        if (primed || last_seen || strb!=7 || addr!=accepted || addr>=count ||
                last!=(addr+1==count)) { error=Error::Write; return false; }
        ++accepted; last_seen=last; return true;
    }
    bool commit(std::size_t addr) {
        if (error!=Error::None) return false;
        if (primed || addr!=committed || committed>=accepted) {error=Error::Commit;return false;}
        ++committed;return true;
    }
    bool prime(std::size_t count) {
        if (error!=Error::None) return false;
        if (primed || !last_seen || !count || committed!=count) {error=Error::Prime;return false;}
        primed=true;return true;
    }
};
struct MemProgramPortSignals {
    std::uint64_t mem_program_generation_o{};
    std::size_t mem_program_queue_o{}, mem_program_imem_address_o{}, mem_program_commit_address_i{};
    std::array<std::uint8_t,16> mem_program_data_o{};
    std::uint8_t mem_program_strb_o{};
    bool mem_program_error_i{}, mem_prime_ready_i{};
    bool mem_program_valid_o{}, mem_program_ready_i{}, mem_program_fire{}, mem_program_last_o{}, mem_program_commit_i{}, mem_prime_valid_o{}, mem_prime_fire{};
    MemProgramEndpoint::Error mem_program_fault_code_i{};
};
}
