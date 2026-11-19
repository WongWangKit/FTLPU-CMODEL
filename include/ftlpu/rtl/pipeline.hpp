#pragma once
#include <optional>
#include <stdexcept>
#include <vector>
namespace ftlpu::rtl {
template <class T> class Pipeline {
public:
  explicit Pipeline(std::size_t stages) : stages_(stages) {
    if (!stages)
      throw std::invalid_argument("pipeline stages");
  }
  const std::optional<T> &output() const { return stages_.back(); }
  const std::optional<T> &tick(std::optional<T> input = {}) {
    for (std::size_t i = stages_.size() - 1; i; --i)
      stages_[i] = std::move(stages_[i - 1]);
    stages_[0] = std::move(input);
    return output();
  }
  void reset() {
    for (auto &s : stages_)
      s.reset();
  }
  bool busy() const {
    for (auto &s : stages_)
      if (s)
        return true;
    return false;
  }

private:
  std::vector<std::optional<T>> stages_;
};
} // namespace ftlpu::rtl
