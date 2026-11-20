#pragma once
#include "ftlpu/core/hardware_params.hpp"
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ftlpu {
struct ExpertWeightEntry {
  std::uint32_t first_segment{}, segment_count{};
};
struct ExpertWeightSegment {
  std::uint64_t ddr_address{};
  std::uint32_t vector_count{}, hemisphere{}, slice{}, bank{}, base_row{},
      fabric_stream{30};
};
// Stored in the existing finite Routing descriptor RAM, not in host pointers.
// Entries occupy two u32 words each; physical segments occupy eight words each.
struct ExpertWeightTableDescriptor {
  static constexpr std::size_t max_segments = 256, max_words = 4096;
  std::uint32_t workset_id{1}, experts{}, segments{};
  std::vector<std::uint32_t> words;
};
inline std::array<std::uint32_t, 8>
expert_weight_segment_words(const ExpertWeightSegment &s) {
  return {static_cast<std::uint32_t>(s.ddr_address),
          static_cast<std::uint32_t>(s.ddr_address >> 32),
          s.vector_count,
          s.hemisphere,
          s.slice,
          s.bank,
          s.base_row,
          s.fabric_stream};
}
inline ExpertWeightSegment
decode_expert_weight_segment(const std::array<std::uint32_t, 8> &w) {
  return {std::uint64_t(w[0]) | (std::uint64_t(w[1]) << 32),
          w[2],
          w[3],
          w[4],
          w[5],
          w[6],
          w[7]};
}
inline ExpertWeightSegment
expert_weight_segment(const ExpertWeightTableDescriptor &t, std::size_t index) {
  if (index >= t.segments)
    throw std::out_of_range("expert weight segment index");
  std::array<std::uint32_t, 8> w{};
  for (std::size_t i = 0; i < 8; ++i)
    w[i] = t.words.at(2 * t.experts + 8 * index + i);
  return decode_expert_weight_segment(w);
}
inline void
validate_expert_weight_table(const ExpertWeightTableDescriptor &t,
                             std::size_t depth = hw::kSramDepthRows) {
  if (!t.workset_id || !t.experts || t.experts > 1024 || !t.segments ||
      t.segments > ExpertWeightTableDescriptor::max_segments ||
      t.words.size() != 2 * t.experts + 8 * t.segments ||
      t.words.size() > ExpertWeightTableDescriptor::max_words)
    throw std::invalid_argument("expert weight table capacity/shape");
  for (std::size_t i = 0; i < t.segments; ++i) {
    const auto s = expert_weight_segment(t, i);
    if (!s.vector_count || s.vector_count > 65536 ||
        s.ddr_address % hw::kPhysicalVectorBytes ||
        s.ddr_address > UINT64_MAX - std::uint64_t(s.vector_count) *
                                         hw::kPhysicalVectorBytes ||
        s.hemisphere >= hw::kHemispheres || s.slice >= hw::kMemSliceColumns ||
        s.bank >= hw::kMemBanksPerSlice ||
        s.fabric_stream >= hw::kStreamsPerDirection ||
        std::uint64_t(s.base_row) + s.vector_count > depth)
      throw std::invalid_argument(
          "invalid expert weight segment address/layout");
  }
  for (std::size_t e = 0; e < t.experts; ++e) {
    const auto first = t.words[2 * e], count = t.words[2 * e + 1];
    if (!count || std::uint64_t(first) + count > t.segments)
      throw std::invalid_argument("invalid expert weight entry");
    for (std::size_t i = first; i < first + count; ++i)
      for (std::size_t j = first; j < i; ++j) {
        const auto a = expert_weight_segment(t, i),
                   b = expert_weight_segment(t, j);
        if (a.hemisphere == b.hemisphere && a.slice == b.slice &&
            a.bank == b.bank &&
            std::uint64_t(a.base_row) <
                std::uint64_t(b.base_row) + b.vector_count &&
            std::uint64_t(b.base_row) <
                std::uint64_t(a.base_row) + a.vector_count)
          throw std::invalid_argument("overlapping expert weight segments");
      }
  }
}
inline ExpertWeightTableDescriptor
make_expert_weight_table(std::uint32_t workset,
                         const std::vector<ExpertWeightEntry> &entries,
                         const std::vector<ExpertWeightSegment> &segments) {
  ExpertWeightTableDescriptor t;
  if (entries.size() > 1024 ||
      segments.size() > ExpertWeightTableDescriptor::max_segments)
    throw std::invalid_argument("expert weight table capacity");
  t.workset_id = workset;
  t.experts = static_cast<std::uint32_t>(entries.size());
  t.segments = static_cast<std::uint32_t>(segments.size());
  for (const auto &e : entries)
    t.words.insert(t.words.end(), {e.first_segment, e.segment_count});
  for (const auto &s : segments) {
    const auto w = expert_weight_segment_words(s);
    t.words.insert(t.words.end(), w.begin(), w.end());
  }
  return t;
}
} // namespace ftlpu
