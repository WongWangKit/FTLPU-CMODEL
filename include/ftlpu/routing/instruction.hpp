#pragma once

#include "ftlpu/core/instruction_codec.hpp"
#include "ftlpu/icu/macro_task_binding.hpp"
#include "ftlpu/routing/expert_weight_descriptor.hpp"
#include <array>
#include <bit>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ftlpu {

// Version 1 is a finite, single-context, area-oriented device implementation.
struct RoutingConfig {
  bool enabled{hw::config::kRoutingEnabled};
  static constexpr std::size_t max_experts = 1024, max_k = 16, max_tokens = 128;
  static constexpr std::size_t block_rows = 32, address_depth = 8;
  static constexpr std::size_t imem_depth = 256, iq_depth = 4,
                               descriptor_depth = 32;
  std::size_t control_latency{hw::config::kRoutingControlLatency};
  void validate() const {
    if (!control_latency || control_latency > 1024)
      throw std::invalid_argument("routing control latency");
  }
};

enum class RoutingElement : std::uint32_t {
  F32 = 1,
  U32 = 2,
  F16 = 3,
  BF16 = 4
};
struct RoutingPlane {
  std::uint32_t hemisphere{}, slice{}, bank{}, base_row{};
  friend bool operator==(const RoutingPlane &, const RoutingPlane &) = default;
};
struct RoutingTensorView {
  RoutingElement element{RoutingElement::F32};
  std::uint32_t rows{}, columns{}, token_stride{1}, feature_stride{};
  std::array<RoutingPlane, 4> planes{};
  std::size_t bytes() const {
    return element == RoutingElement::F16 || element == RoutingElement::BF16
               ? 2
               : 4;
  }
  bool present() const { return rows != 0; }
  std::size_t row(std::size_t r, std::size_t c, std::size_t p) const {
    if (r >= rows || c >= columns || p >= bytes())
      throw std::out_of_range("routing tensor coordinate");
    return planes[p].base_row + std::uint64_t(r) * token_stride +
           std::uint64_t(c / hw::kPhysicalVectorBytes) * feature_stride;
  }
  void validate(std::size_t depth = hw::kSramDepthRows) const {
    if (!rows || !columns || !token_stride || !feature_stride ||
        static_cast<unsigned>(element) < 1 ||
        static_cast<unsigned>(element) > 4)
      throw std::invalid_argument("invalid routing byte-planar view");
    const auto blocks =
        (std::uint64_t(columns) - 1) / hw::kPhysicalVectorBytes + 1;
    const auto divisor = std::gcd(token_stride, feature_stride);
    if (feature_stride / divisor < rows && token_stride / divisor < blocks)
      throw std::invalid_argument("routing view aliases token/feature rows");
    const auto last_token = std::uint64_t(rows - 1) * token_stride;
    const auto last_feature = (blocks - 1) * feature_stride;
    if (last_token >= depth || last_feature >= depth)
      throw std::out_of_range("routing view strides exceed SRAM");
    for (std::size_t p = 0; p < bytes(); ++p) {
      const auto &q = planes[p];
      if (q.hemisphere >= hw::kHemispheres || q.slice >= hw::kMemSliceColumns ||
          q.bank >= hw::kMemBanksPerSlice ||
          std::uint64_t(q.base_row) + last_token + last_feature >= depth)
        throw std::out_of_range("routing view exceeds physical SRAM");
      for (std::size_t j = 0; j < p; ++j)
        if (q.hemisphere == planes[j].hemisphere &&
            q.slice == planes[j].slice && q.bank == planes[j].bank)
          throw std::invalid_argument(
              "routing byte planes require distinct banks");
    }
  }
  friend bool operator==(const RoutingTensorView &,
                         const RoutingTensorView &) = default;
};
struct RouteSelectDescriptor {
  std::uint32_t context_id{1}, tokens{}, experts{}, k{};
  RoutingTensorView choice_scores{}, raw_scores{}, eligibility{},
      selected_ids{}, selected_scores{};
};
struct RoutePlanDescriptor {
  std::uint32_t context_id{1};
  // Absent means selected raw scores are already the final routing weights.
  RoutingTensorView weights{};
};
struct RouteAddressDescriptor {
  std::uint32_t context_id{1}, active_index{}, row_offset{}, max_rows{32};
  RoutingTensorView source{}, destination{};
};
struct RouteBlockInfo {
  std::uint32_t context_id{}, expert_id{}, expert_token_count{}, row_offset{},
      valid_rows{};
  bool more{};
  friend bool operator==(const RouteBlockInfo &,
                         const RouteBlockInfo &) = default;
};
struct RouteTaskDispatchDescriptor {
  std::uint32_t context_id{1}, address_template_descriptor_id{}, block_rows{32};
};
struct ExpertTaskIdentity {
  std::uint64_t generation{}, plan_epoch{}, task_id{};
  friend bool operator==(const ExpertTaskIdentity &,
                         const ExpertTaskIdentity &) = default;
};
struct ExpertTaskContext {
  ExpertTaskIdentity identity{};
  std::uint32_t route_context_id{}, active_index{}, expert_id{},
      expert_token_count{};
  std::uint32_t row_offset{}, valid_rows{}, assignment_base{};
  bool more{};
  friend bool operator==(const ExpertTaskContext &,
                         const ExpertTaskContext &) = default;
};
struct ExpertTemplateQueue {
  std::uint32_t kind{}, index{};
  std::vector<MacroTaskBinding> bindings;
  std::uint32_t weight_base{};
};
struct ExpertMacroTemplateDescriptor {
  RoutingTensorView output;
  std::uint32_t workspace_base{}, scale_bf16{0x3f80};
  std::vector<ExpertTemplateQueue> queues;
};
struct RouteCombineDescriptor {
  std::uint32_t context_id{1};
  RoutingTensorView source, destination;
};
inline void validate_route_combine(const RouteCombineDescriptor &d,
                                  std::size_t depth = hw::kSramDepthRows) {
  d.source.validate(depth); d.destination.validate(depth);
  if (!d.context_id || (d.source.element != RoutingElement::BF16 && d.source.element != RoutingElement::F16)
      || d.source.element != d.destination.element
      || d.source.columns != d.destination.columns || d.source.columns % hw::kPhysicalVectorBytes)
    throw std::invalid_argument("combine requires matching whole-vector FP16/BF16 views");
  for (std::size_t a=0;a<2;++a) for (std::size_t b=0;b<2;++b) {
    const auto x=d.source.planes[a], y=d.destination.planes[b];
    if (x.hemisphere==y.hemisphere && x.slice==y.slice && x.bank==y.bank)
      throw std::invalid_argument("combine source and destination must use distinct banks");
  }
}
inline void
validate_expert_macro_template(const ExpertMacroTemplateDescriptor &t,
                               std::size_t depth = hw::kSramDepthRows) {
  t.output.validate(depth);
  if (t.output.columns % hw::kPhysicalVectorBytes || t.queues.empty() ||
      t.queues.size() > 64)
    throw std::invalid_argument("expert macro template capacity/layout");
  for (std::size_t i = 0; i < t.queues.size(); ++i) {
    const auto &q = t.queues[i];
    const auto capacity =
        q.kind == 0
            ? hw::kHemispheres * hw::kMemSliceColumns * hw::kMemBanksPerSlice
        : q.kind <= 3 ? hw::kMxmCount
        : q.kind == 4 ? hw::kVxmColumnsPerDirection
                      : hw::kHemispheres;
    if (q.kind > 6 || q.index >= capacity)
      throw std::invalid_argument("expert macro queue identity");
    validate_macro_task_bindings(q.bindings);
    for (const auto &b : q.bindings)
      if ((q.kind == 4) !=
              (b.target == MacroBindingTarget::NativeRepeatCount) ||
          (q.kind > 4 && !q.bindings.empty()))
        throw std::invalid_argument("expert macro binding queue type");
    for (std::size_t j = 0; j < i; ++j)
      if (t.queues[j].kind == q.kind && t.queues[j].index == q.index)
        throw std::invalid_argument("duplicate expert macro queue");
  }
}
enum class RoutingOpcode : std::uint32_t {
  Select = 1,
  Plan = 2,
  Address = 3,
  Dispatch = 4,
  WeightTable = 5, // configuration record only; not an executable opcode
  ExpertTemplate = 6,
  Combine = 7
};
inline constexpr const char *routing_opcode_name(RoutingOpcode opcode) {
  switch (opcode) {
  case RoutingOpcode::Select:
    return "ROUTE_SELECT";
  case RoutingOpcode::Plan:
    return "ROUTE_PLAN";
  case RoutingOpcode::Address:
    return "ROUTE_ADDR";
  case RoutingOpcode::Dispatch:
    return "ROUTE_DISPATCH";
  case RoutingOpcode::WeightTable:
    return "WEIGHT_TABLE";
  case RoutingOpcode::ExpertTemplate:
    return "EXPERT_TEMPLATE";
  case RoutingOpcode::Combine:
    return "ROUTE_COMBINE";
  }
  return "INVALID";
}
struct RoutingDescriptor {
  RoutingOpcode opcode{RoutingOpcode::Select};
  RouteSelectDescriptor select{};
  RoutePlanDescriptor plan{};
  RouteAddressDescriptor address{};
  RouteTaskDispatchDescriptor dispatch{};
  ExpertWeightTableDescriptor weight_table{};
  ExpertMacroTemplateDescriptor expert_template{};
  RouteCombineDescriptor combine{};
};
struct RoutingInstruction {
  RoutingOpcode opcode{RoutingOpcode::Select};
  std::uint32_t descriptor{}, completion{};
};
inline void
validate_routing_descriptor_references(const std::vector<RoutingDescriptor> &ds,
                                       std::uint32_t capability) {
  if (capability > 5 || (!capability && !ds.empty()))
    throw std::invalid_argument("unsupported routing capability");
  bool table_seen = false, template_seen = false;
  for (const auto &d : ds) {
    if (d.opcode == RoutingOpcode::Combine) {
      if (capability < 5) throw std::invalid_argument("combine requires Routing v5");
      validate_route_combine(d.combine);
      bool template_match=false, select_match=false;
      for (const auto &x:ds) {
        if (x.opcode==RoutingOpcode::ExpertTemplate && x.expert_template.output==d.combine.source) template_match=true;
        if (x.opcode==RoutingOpcode::Select && x.select.context_id==d.combine.context_id) {
          if (d.combine.source.rows != std::uint64_t(x.select.tokens)*x.select.k || d.combine.destination.rows!=x.select.tokens)
            throw std::invalid_argument("combine shape must match current routing plan");
          select_match=true;
        }
      }
      if(!template_match || !select_match) throw std::invalid_argument("combine requires matching expert output and SELECT");
    }
    if (d.opcode == RoutingOpcode::ExpertTemplate) {
      if (capability < 4 || template_seen)
        throw std::invalid_argument(
            "expert template requires Routing v4; only one template supported");
      template_seen = true;
      validate_expert_macro_template(d.expert_template);
    }
    if (d.opcode == RoutingOpcode::WeightTable) {
      if (capability < 3 || table_seen)
        throw std::invalid_argument(
            "weight table requires Routing v3; only one table supported");
      table_seen = true;
      validate_expert_weight_table(d.weight_table);
    }
    if (d.opcode != RoutingOpcode::Dispatch)
      continue;
    if (capability < 2 ||
        d.dispatch.address_template_descriptor_id >= ds.size())
      throw std::invalid_argument(
          "dispatch requires Routing v2 and valid template");
    const auto &a = ds[d.dispatch.address_template_descriptor_id];
    if (a.opcode != RoutingOpcode::Address ||
        a.address.context_id != d.dispatch.context_id ||
        !d.dispatch.block_rows || d.dispatch.block_rows > a.address.max_rows)
      throw std::invalid_argument("invalid dispatch address template");
  }
  if (table_seen) {
    const ExpertWeightTableDescriptor *table = nullptr;
    for (const auto &d : ds)
      if (d.opcode == RoutingOpcode::WeightTable)
        table = &d.weight_table;
    for (const auto &d : ds)
      if (d.opcode == RoutingOpcode::Dispatch) {
        const auto &a = ds[d.dispatch.address_template_descriptor_id].address;
        for (const auto &s : ds)
          if (s.opcode == RoutingOpcode::Select &&
              s.select.context_id == d.dispatch.context_id &&
              s.select.experts != table->experts)
            throw std::invalid_argument("weight table expert count mismatch");
        for (std::size_t i = 0; i < table->segments; ++i) {
          const auto s = expert_weight_segment(*table, i);
          for (const auto *v : {&a.source, &a.destination})
            for (std::size_t p = 0; p < v->bytes(); ++p) {
              const auto plane = v->planes[p];
              if (s.hemisphere == plane.hemisphere && s.slice == plane.slice &&
                  s.bank == plane.bank)
                throw std::invalid_argument("weight banks must be distinct "
                                            "from activation/input banks");
            }
        }
      }
  }
  if (template_seen) {
    const ExpertMacroTemplateDescriptor *t = nullptr;
    const ExpertWeightTableDescriptor *w = nullptr;
    bool dispatch = false;
    for (const auto &d : ds) {
      if (d.opcode == RoutingOpcode::ExpertTemplate)
        t = &d.expert_template;
      if (d.opcode == RoutingOpcode::WeightTable)
        w = &d.weight_table;
      if (d.opcode == RoutingOpcode::Dispatch)
        dispatch = true;
    }
    if (!dispatch)
      throw std::invalid_argument("expert template requires DISPATCH");
    if (t->scale_bf16 > 65535)
      throw std::invalid_argument("expert scale field exceeds BF16");
    for (const auto &d : ds)
      if (d.opcode == RoutingOpcode::Dispatch) {
        const auto &a = ds[d.dispatch.address_template_descriptor_id].address;
        for (const auto &select : ds)
          if (select.opcode == RoutingOpcode::Select &&
              select.select.context_id == d.dispatch.context_id &&
              std::uint64_t(select.select.tokens) * select.select.k >
                  t->output.rows)
            throw std::invalid_argument(
                "expert output cannot hold assignment rows");
        for (std::size_t op = 0; op < t->output.bytes(); ++op) {
          const auto out = t->output.planes[op];
          for (const auto *v : {&a.source, &a.destination})
            for (std::size_t ip = 0; ip < v->bytes(); ++ip) {
              const auto in = v->planes[ip];
              if (out.hemisphere == in.hemisphere && out.slice == in.slice &&
                  out.bank == in.bank)
                throw std::invalid_argument(
                    "expert output bank aliases activation/input");
            }
          if (w)
            for (std::size_t i = 0; i < w->segments; ++i) {
              const auto seg = expert_weight_segment(*w, i);
              if (out.hemisphere == seg.hemisphere && out.slice == seg.slice &&
                  out.bank == seg.bank)
                throw std::invalid_argument(
                    "expert output aliases resident weight bank");
            }
        }
      }
    if (w) {
      const auto count = w->words[1], first = w->words[0];
      for (std::size_t e = 0; e < w->experts; ++e) {
        if (w->words[2 * e + 1] != count)
          throw std::invalid_argument(
              "expert template requires a fixed weight layout");
        for (std::size_t i = 0; i < count; ++i) {
          const auto a = expert_weight_segment(*w, first + i),
                     b = expert_weight_segment(*w, w->words[2 * e] + i);
          if (a.hemisphere != b.hemisphere || a.slice != b.slice ||
              a.bank != b.bank || a.base_row != b.base_row ||
              a.vector_count != b.vector_count)
            throw std::invalid_argument(
                "expert template weight queue/layout changes across experts");
        }
      }
      for (const auto &q : t->queues)
        for (const auto &binding : q.bindings)
          if (binding.parameter == MacroTaskParameter::WeightBase) {
            if (q.kind != 0)
              throw std::invalid_argument(
                  "weight base binding requires MEM queue");
            const auto local =
                q.index % (hw::kMemSliceColumns * hw::kMemBanksPerSlice);
            const auto h = q.index /
                           (hw::kMemSliceColumns * hw::kMemBanksPerSlice),
                       slice = local / hw::kMemBanksPerSlice,
                       bank = local % hw::kMemBanksPerSlice;
            unsigned matches = 0;
            for (std::size_t i = 0; i < count; ++i) {
              const auto a = expert_weight_segment(*w, first + i);
              if (a.hemisphere == h && a.slice == slice && a.bank == bank &&
                  a.base_row == q.weight_base)
                ++matches;
            }
            if (matches != 1)
              throw std::invalid_argument(
                  "ambiguous/mismatching template weight base");
          }
    }
  }
}
struct RoutingWord {
  std::array<std::uint32_t, 3> lanes{};
  friend bool operator==(const RoutingWord &, const RoutingWord &) = default;
};
inline RoutingWord encode_routing_instruction(RoutingInstruction i) {
  if (static_cast<unsigned>(i.opcode) < 1 ||
      (static_cast<unsigned>(i.opcode) > 4 && i.opcode != RoutingOpcode::Combine) ||
      i.descriptor >= RoutingConfig::descriptor_depth || !i.completion ||
      i.completion >= 0x80000000u)
    throw std::invalid_argument("invalid routing instruction");
  return {{{static_cast<std::uint32_t>(i.opcode) << 2, i.descriptor,
            i.completion}}};
}
inline RoutingInstruction decode_routing_instruction(RoutingWord w) {
  RoutingInstruction i{static_cast<RoutingOpcode>(w.lanes[0] >> 2), w.lanes[1],
                       w.lanes[2]};
  if (encode_routing_instruction(i) != w)
    throw std::invalid_argument("invalid routing reserved bits");
  return i;
}

// Explicit little-endian u32 records; never serialize C++ object padding.
inline void routing_pack_view(std::vector<std::uint32_t> &w,
                              const RoutingTensorView &v) {
  w.insert(w.end(), {static_cast<std::uint32_t>(v.element), v.rows, v.columns,
                     v.token_stride, v.feature_stride});
  for (auto p : v.planes)
    w.insert(w.end(), {p.hemisphere, p.slice, p.bank, p.base_row});
}
inline std::vector<std::uint32_t>
encode_routing_descriptors(const std::vector<RoutingDescriptor> &ds) {
  if (ds.size() > RoutingConfig::descriptor_depth)
    throw std::invalid_argument("routing descriptor capacity");
  std::vector<std::uint32_t> w{0x31554352u,
                               static_cast<std::uint32_t>(ds.size())};
  for (const auto &d : ds) {
    w.push_back(static_cast<std::uint32_t>(d.opcode));
    switch (d.opcode) {
    case RoutingOpcode::Select:
      w.insert(w.end(), {d.select.context_id, d.select.tokens, d.select.experts,
                         d.select.k});
      for (const auto *v : {&d.select.choice_scores, &d.select.raw_scores,
                            &d.select.eligibility, &d.select.selected_ids,
                            &d.select.selected_scores})
        routing_pack_view(w, *v);
      break;
    case RoutingOpcode::Plan:
      w.push_back(d.plan.context_id);
      routing_pack_view(w, d.plan.weights);
      break;
    case RoutingOpcode::Address:
      w.insert(w.end(), {d.address.context_id, d.address.active_index,
                         d.address.row_offset, d.address.max_rows});
      routing_pack_view(w, d.address.source);
      routing_pack_view(w, d.address.destination);
      break;
    case RoutingOpcode::Dispatch:
      w.insert(w.end(), {d.dispatch.context_id,
                         d.dispatch.address_template_descriptor_id,
                         d.dispatch.block_rows});
      break;
    case RoutingOpcode::Combine:
      w.push_back(d.combine.context_id); routing_pack_view(w,d.combine.source); routing_pack_view(w,d.combine.destination);
      break;
    case RoutingOpcode::WeightTable:
      w.insert(w.end(),
               {d.weight_table.workset_id, d.weight_table.experts,
                d.weight_table.segments,
                static_cast<std::uint32_t>(d.weight_table.words.size())});
      w.insert(w.end(), d.weight_table.words.begin(),
               d.weight_table.words.end());
      break;
    case RoutingOpcode::ExpertTemplate: {
      const auto &t = d.expert_template;
      routing_pack_view(w, t.output);
      w.insert(w.end(), {t.workspace_base, t.scale_bf16,
                         static_cast<std::uint32_t>(t.queues.size())});
      for (const auto &q : t.queues) {
        w.insert(w.end(), {q.kind, q.index, q.weight_base,
                           static_cast<std::uint32_t>(q.bindings.size())});
        for (const auto &b : q.bindings)
          w.insert(w.end(), {b.record, static_cast<std::uint32_t>(b.target),
                             static_cast<std::uint32_t>(b.parameter),
                             b.multiplier, static_cast<std::uint32_t>(b.addend),
                             b.relative ? 1u : 0u});
      }
      break;
    }
    default:
      throw std::invalid_argument("routing descriptor opcode");
    }
  }
  if (w.size() > ExpertWeightTableDescriptor::max_words)
    throw std::invalid_argument("routing descriptor word capacity");
  return w;
}
inline std::vector<RoutingDescriptor>
decode_routing_descriptors(const std::vector<std::uint32_t> &w) {
  std::size_t at = 0;
  const auto get = [&]() {
    if (at == w.size())
      throw std::invalid_argument("truncated routing descriptors");
    return w[at++];
  };
  if (get() != 0x31554352u)
    throw std::invalid_argument("routing descriptor magic");
  auto n = get();
  if (n > RoutingConfig::descriptor_depth)
    throw std::invalid_argument("routing descriptor capacity");
  const auto view = [&]() {
    RoutingTensorView v;
    v.element = static_cast<RoutingElement>(get());
    v.rows = get();
    v.columns = get();
    v.token_stride = get();
    v.feature_stride = get();
    for (auto &p : v.planes) {
      p.hemisphere = get();
      p.slice = get();
      p.bank = get();
      p.base_row = get();
    }
    return v;
  };
  std::vector<RoutingDescriptor> ds;
  for (std::size_t j = 0; j < n; ++j) {
    RoutingDescriptor d;
    d.opcode = static_cast<RoutingOpcode>(get());
    switch (d.opcode) {
    case RoutingOpcode::Select:
      d.select.context_id = get();
      d.select.tokens = get();
      d.select.experts = get();
      d.select.k = get();
      d.select.choice_scores = view();
      d.select.raw_scores = view();
      d.select.eligibility = view();
      d.select.selected_ids = view();
      d.select.selected_scores = view();
      break;
    case RoutingOpcode::Plan:
      d.plan.context_id = get();
      d.plan.weights = view();
      break;
    case RoutingOpcode::Address:
      d.address.context_id = get();
      d.address.active_index = get();
      d.address.row_offset = get();
      d.address.max_rows = get();
      d.address.source = view();
      d.address.destination = view();
      break;
    case RoutingOpcode::Dispatch:
      d.dispatch.context_id = get();
      d.dispatch.address_template_descriptor_id = get();
      d.dispatch.block_rows = get();
      break;
    case RoutingOpcode::Combine:
      d.combine.context_id=get(); d.combine.source=view(); d.combine.destination=view();
      break;
    case RoutingOpcode::WeightTable: {
      auto &t = d.weight_table;
      t.workset_id = get();
      t.experts = get();
      t.segments = get();
      const auto size = get();
      if (size > ExpertWeightTableDescriptor::max_words)
        throw std::invalid_argument("weight table word capacity");
      for (std::uint32_t i = 0; i < size; ++i)
        t.words.push_back(get());
      break;
    }
    case RoutingOpcode::ExpertTemplate: {
      auto &t = d.expert_template;
      t.output = view();
      t.workspace_base = get();
      t.scale_bf16 = get();
      const auto n = get();
      if (n > 64)
        throw std::invalid_argument("expert template queue capacity");
      for (std::uint32_t i = 0; i < n; ++i) {
        ExpertTemplateQueue q;
        q.kind = get();
        q.index = get();
        q.weight_base = get();
        const auto m = get();
        if (m > 32)
          throw std::invalid_argument("expert binding capacity");
        for (std::uint32_t j = 0; j < m; ++j) {
          MacroTaskBinding b;
          b.record = get();
          b.target = static_cast<MacroBindingTarget>(get());
          b.parameter = static_cast<MacroTaskParameter>(get());
          b.multiplier = get();
          b.addend = static_cast<std::int32_t>(get());
          const auto relative = get();
          if (relative > 1)
            throw std::invalid_argument("binding relative flag");
          b.relative = relative != 0;
          q.bindings.push_back(b);
        }
        t.queues.push_back(std::move(q));
      }
      break;
    }
    default:
      throw std::invalid_argument("routing descriptor opcode");
    }
    ds.push_back(d);
  }
  if (at != w.size())
    throw std::invalid_argument("trailing routing descriptors");
  return ds;
}
} // namespace ftlpu
