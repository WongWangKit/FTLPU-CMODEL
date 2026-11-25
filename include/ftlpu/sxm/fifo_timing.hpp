#pragma once
#include "ftlpu/core/hardware_params.hpp"

namespace ftlpu::sxm_fifo {
constexpr int lanes = hw::kLanesPerTile;
constexpr int superlanes = hw::kTileRows;
constexpr int width = lanes * superlanes;
constexpr int skew = superlanes - 1;
constexpr bool binary_exchange = (superlanes & (superlanes - 1)) == 0;
// A binary butterfly retains its original latency. An arbitrary-width line
// reverses each active contiguous interval in at most superlanes layers.
constexpr int exchange_stages = binary_exchange ? superlanes - 1 : superlanes;

// Packets entering an exchange wave from one matrix share source + target.
// Its participating wires form the interval [first,last]. Alternating adjacent
// swaps for the length of that interval reverse it, including for odd lengths.
constexpr int interval_exchange_port(int stage, int port, int sum, int count) {
    const int first = sum < count ? 0 : sum - (count - 1);
    const int last = sum < count ? sum : count - 1;
    if (stage >= last - first + 1) return port;
    const int neighbor = ((port + stage) % 2 == 0) ? port + 1 : port - 1;
    return neighbor < first || neighbor > last ? port : neighbor;
}
constexpr int input_delay(int sl) { return lanes * sl + skew - sl; }
constexpr int output_delay(int sl) { return lanes * (skew - sl) + sl; }
constexpr int output_offset = width + skew + exchange_stages;
constexpr int execution_cycles = output_offset + width + skew;
}
