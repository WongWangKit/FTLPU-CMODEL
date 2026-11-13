#pragma once
#include <array>
#include <cstdint>
namespace ftlpu::isa {
struct EncodedMemIcuWord { std::array<std::uint32_t,3> lanes{}; };
struct EncodedMxmLoadIcuWord { std::array<std::uint32_t,4> lanes{}; };
struct EncodedMxmDequantIcuWord { std::array<std::uint32_t,4> lanes{}; };
struct EncodedMxmComputeIcuWord { std::array<std::uint32_t,4> lanes{}; };
struct EncodedSxmIcuWord { std::array<std::uint32_t,3> lanes{}; };
} // namespace ftlpu::isa
