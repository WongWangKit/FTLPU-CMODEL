#pragma once
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ftlpu {
enum class MacroTaskParameter : std::uint32_t {
  Rows,
  InputBase,
  WeightBase,
  OutputBase,
  WorkspaceBase,
  ScaleBits,
  AssignmentBase,
  Zero
};
enum class MacroBindingTarget : std::uint32_t {
  InnerCount,
  OuterCount,
  Operand,
  ScaleBf16,
  NativeRepeatCount
};
struct MacroTaskBinding {
  std::uint32_t record{};
  MacroBindingTarget target{MacroBindingTarget::OuterCount};
  MacroTaskParameter parameter{MacroTaskParameter::Rows};
  std::uint32_t multiplier{1};
  std::int32_t addend{};
  bool relative{};
};
struct MacroTaskParameters {
  std::array<std::uint32_t, 8> values{};
  std::uint64_t generation{}, task_id{};
};
inline void
validate_macro_task_bindings(const std::vector<MacroTaskBinding> &bindings) {
  if (bindings.size() > 32)
    throw std::invalid_argument("macro task binding capacity");
  for (const auto &b : bindings)
    if (static_cast<unsigned>(b.target) > 4 ||
        static_cast<unsigned>(b.parameter) > 7 || b.multiplier > 65536)
      throw std::invalid_argument("invalid macro task binding");
  for (std::size_t i = 0; i < bindings.size(); ++i) {
    const auto &b = bindings[i];
    if ((b.target == MacroBindingTarget::InnerCount ||
         b.target == MacroBindingTarget::OuterCount ||
         b.target == MacroBindingTarget::NativeRepeatCount) &&
        b.parameter != MacroTaskParameter::Rows &&
        b.parameter != MacroTaskParameter::Zero)
      throw std::invalid_argument(
          "task counts must depend on valid_rows or a constant");
    for (std::size_t j = 0; j < i; ++j)
      if (bindings[j].record == b.record && bindings[j].target == b.target)
        throw std::invalid_argument("duplicate macro task field binding");
  }
}
inline std::uint32_t macro_task_binding_value(const MacroTaskBinding &b,
                                              const MacroTaskParameters &p,
                                              std::uint64_t original = 0) {
  const auto value =
      std::int64_t(p.values.at(static_cast<unsigned>(b.parameter))) *
          b.multiplier +
      b.addend + (b.relative ? static_cast<std::int64_t>(original) : 0);
  if (value < 0 || value > UINT32_MAX)
    throw std::overflow_error("macro task binding value overflow");
  return static_cast<std::uint32_t>(value);
}
} // namespace ftlpu
