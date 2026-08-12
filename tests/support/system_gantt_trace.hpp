#pragma once

#include "ftlpu/system/tsp_slice_system.hpp"

#include <cctype>
#include <cstddef>
#include <string>
#include <string_view>

namespace integration_timing {

// Main intentionally has no online performance monitor. Workloads retain this
// inert tracing facade so functional schedules do not depend on instrumentation.
class SystemGanttTrace {
public:
    enum class Detail { Chip, Superlane, Full };

    static constexpr bool enabled() noexcept { return false; }
    static constexpr Detail detail() noexcept { return Detail::Chip; }

    void attach(ftlpu::TspSliceSystem&) noexcept {}
    void detach(ftlpu::TspSliceSystem&) noexcept {}
    void capture(const ftlpu::TspSliceSystem&) noexcept {}

    void phase(std::size_t, std::size_t, std::string) noexcept {}
    void write(std::string_view, std::string_view) const noexcept {}

    static std::string prefix_from_name(std::string_view name)
    {
        auto result = std::string{};
        result.reserve(name.size());
        for (const auto character : name) {
            const auto value = static_cast<unsigned char>(character);
            result.push_back(std::isalnum(value)
                    ? static_cast<char>(std::tolower(value))
                    : '_');
        }
        return result;
    }
};

} // namespace integration_timing
