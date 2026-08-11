#pragma once

#include "ftlpu/system/tsp_slice_system.hpp"

#include <cstddef>
#include <filesystem>
#include <vector>

namespace ftlpu::test::smollm2_layer {

constexpr std::size_t kPrefillLength = 128;
constexpr std::size_t kHidden = 576;
constexpr std::size_t kIntermediate = 1536;

struct PhaseResult {
    std::vector<float> output;
    std::vector<float> key_cache;
    std::vector<float> value_cache;
    std::size_t cycles{0};
};

PhaseResult run_prefill_attention(
    TspSliceSystem& system,
    const std::vector<float>& input,
    const std::filesystem::path& trace_path = {});

PhaseResult run_prefill_ffn(
    TspSliceSystem& system,
    const std::vector<float>& input,
    const std::filesystem::path& trace_path = {});

PhaseResult run_decode_attention(
    TspSliceSystem& system,
    const std::vector<float>& input,
    const std::filesystem::path& trace_path = {},
    const std::filesystem::path& log_dir = {},
    const std::vector<float>& prefill_keys = {},
    const std::vector<float>& prefill_values = {});

PhaseResult run_decode_ffn(
    TspSliceSystem& system,
    const std::vector<float>& input,
    const std::filesystem::path& trace_path = {});

} // namespace ftlpu::test::smollm2_layer
