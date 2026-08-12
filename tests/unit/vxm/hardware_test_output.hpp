#pragma once

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string_view>

namespace vxm_hardware_test {

inline std::filesystem::path results_directory()
{
    const auto source = std::filesystem::path(__FILE__);
    auto hardware = source.is_absolute()
        ? source.parent_path()
        : std::filesystem::current_path() / source.parent_path();
    if (!std::filesystem::exists(hardware)) {
        hardware = std::filesystem::current_path()
            / "tests" / "vxm" / "hardware";
    }
    auto results = hardware / "results";
    std::filesystem::create_directories(results);
    return results;
}

inline std::filesystem::path write_pass_result(
    std::string_view filename, std::string_view test_name)
{
    const auto path = results_directory() / filename;
    auto file = std::ofstream{path, std::ios::trunc};
    if (!file) {
        throw std::runtime_error(
            "cannot create VXM hardware test result");
    }
    file << "VXM hardware test\n"
         << "test=" << test_name << '\n'
         << "status=PASS\n";
    return path;
}

} // namespace vxm_hardware_test
