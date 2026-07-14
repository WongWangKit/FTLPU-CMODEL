#pragma once

#include "ftlpu/core/stream_fabric.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ftlpu {

// Declarative description of the physical SR columns and directed transport
// links in one hemisphere.  A physical SR column contains both eastward and
// westward register banks; direction belongs to StreamId/Route, not to the
// column identity.
class StreamTopology {
public:
    using ColumnId = std::size_t;
    using RouteId = std::size_t;

    enum class RouteKind {
        Normal,
        Bypass,
    };

    struct DirectionalPorts {
        std::optional<ColumnId> input{};
        std::optional<ColumnId> output{};
    };

    struct SliceBinding {
        std::string name{};
        DirectionalPorts east{};
        DirectionalPorts west{};
    };

    struct Route {
        std::string name{};
        ColumnId source{0};
        ColumnId destination{0};
        StreamDirection direction{StreamDirection::East};
        RouteKind kind{RouteKind::Normal};
        bool enabled_by_default{true};
        bool multicast_allowed{false};
    };

    class RouteSelection {
    public:
        explicit RouteSelection(const StreamTopology& topology)
            : enabled_(topology.routes_.size(), false)
            , route_names_(topology.route_names_)
        {
            for (std::size_t i = 0; i < topology.routes_.size(); ++i) {
                enabled_[i] = topology.routes_[i].enabled_by_default;
            }
        }

        void enable(const std::string& route_name)
        {
            enabled_.at(route_id(route_name)) = true;
        }

        void disable(const std::string& route_name)
        {
            enabled_.at(route_id(route_name)) = false;
        }

        bool enabled(RouteId route) const
        {
            return enabled_.at(route);
        }

    private:
        RouteId route_id(const std::string& name) const
        {
            const auto it = route_names_.find(name);
            if (it == route_names_.end()) {
                throw std::out_of_range("unknown stream route: " + name);
            }
            return it->second;
        }

        std::vector<bool> enabled_{};
        std::unordered_map<std::string, RouteId> route_names_{};
    };

    ColumnId add_column(std::string name)
    {
        if (name.empty()) {
            throw std::invalid_argument("SR column name must not be empty");
        }
        if (column_names_.contains(name)) {
            throw std::invalid_argument("duplicate SR column name: " + name);
        }
        const auto id = columns_.size();
        column_names_.emplace(name, id);
        columns_.push_back(std::move(name));
        return id;
    }

    ColumnId column(const std::string& name) const
    {
        const auto it = column_names_.find(name);
        if (it == column_names_.end()) {
            throw std::out_of_range("unknown SR column: " + name);
        }
        return it->second;
    }

    const std::string& column_name(ColumnId id) const
    {
        return columns_.at(id);
    }

    std::size_t column_count() const noexcept
    {
        return columns_.size();
    }

    void bind_slice(SliceBinding binding)
    {
        if (binding.name.empty()) {
            throw std::invalid_argument("functional-slice binding name must not be empty");
        }
        if (slice_names_.contains(binding.name)) {
            throw std::invalid_argument("duplicate functional-slice binding: " + binding.name);
        }
        validate_ports(binding.east);
        validate_ports(binding.west);
        slice_names_.emplace(binding.name, slices_.size());
        slices_.push_back(std::move(binding));
    }

    const SliceBinding& slice(const std::string& name) const
    {
        const auto it = slice_names_.find(name);
        if (it == slice_names_.end()) {
            throw std::out_of_range("unknown functional-slice binding: " + name);
        }
        return slices_.at(it->second);
    }

    RouteId add_route(Route route)
    {
        if (route.name.empty()) {
            throw std::invalid_argument("stream route name must not be empty");
        }
        if (route_names_.contains(route.name)) {
            throw std::invalid_argument("duplicate stream route: " + route.name);
        }
        check_column(route.source);
        check_column(route.destination);
        if (route.source == route.destination) {
            throw std::invalid_argument("stream route must connect two distinct SR columns");
        }
        const auto id = routes_.size();
        route_names_.emplace(route.name, id);
        routes_.push_back(std::move(route));
        return id;
    }

    const std::vector<Route>& routes() const noexcept
    {
        return routes_;
    }

    RouteSelection default_route_selection() const
    {
        return RouteSelection(*this);
    }

    std::vector<StreamRegisterFabric::Link> active_links(
        const RouteSelection& selection) const
    {
        validate_active_routes(selection);
        std::vector<StreamRegisterFabric::Link> result{};
        result.reserve(routes_.size());
        for (std::size_t i = 0; i < routes_.size(); ++i) {
            if (!selection.enabled(i)) {
                continue;
            }
            const auto& route = routes_[i];
            result.push_back(StreamRegisterFabric::Link {
                route.source,
                route.destination,
                route.direction,
                true,
            });
        }
        return result;
    }

    void stage_active_routes(
        StreamRegisterFabric& fabric,
        const RouteSelection& selection) const
    {
        if (fabric.column_count() != column_count()) {
            throw std::invalid_argument(
                "stream fabric column count does not match topology");
        }
        fabric.stage_links(active_links(selection));
    }

private:
    void check_column(ColumnId id) const
    {
        if (id >= columns_.size()) {
            throw std::out_of_range("SR column ID is outside topology");
        }
    }

    void validate_ports(const DirectionalPorts& ports) const
    {
        if (ports.input.has_value()) {
            check_column(*ports.input);
        }
        if (ports.output.has_value()) {
            check_column(*ports.output);
        }
    }

    void validate_active_routes(const RouteSelection& selection) const
    {
        struct Outgoing {
            std::size_t count{0};
            bool all_multicast{true};
        };
        std::vector<Outgoing> east(columns_.size());
        std::vector<Outgoing> west(columns_.size());

        for (std::size_t i = 0; i < routes_.size(); ++i) {
            if (!selection.enabled(i)) {
                continue;
            }
            const auto& route = routes_[i];
            auto& outgoing = route.direction == StreamDirection::East
                ? east[route.source]
                : west[route.source];
            ++outgoing.count;
            outgoing.all_multicast = outgoing.all_multicast && route.multicast_allowed;
        }

        const auto validate_direction = [](const std::vector<Outgoing>& outgoing) {
            for (const auto& entry : outgoing) {
                if (entry.count > 1 && !entry.all_multicast) {
                    throw std::logic_error(
                        "multiple active routes leave one SR column/direction; "
                        "normal and bypass routes must be selected exclusively");
                }
            }
        };
        validate_direction(east);
        validate_direction(west);
    }

    std::vector<std::string> columns_{};
    std::unordered_map<std::string, ColumnId> column_names_{};
    std::vector<SliceBinding> slices_{};
    std::unordered_map<std::string, std::size_t> slice_names_{};
    std::vector<Route> routes_{};
    std::unordered_map<std::string, RouteId> route_names_{};
};

} // namespace ftlpu
