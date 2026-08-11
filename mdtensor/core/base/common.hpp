/**
 * @file
 * @brief Common utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "core.hpp"

namespace mdtensor::core {

template <std::integral index_t>
[[nodiscard]] constexpr index_t bounding_index(index_t index,
                                               const std::size_t &bound) {
    if constexpr (std::is_signed_v<index_t>) {
        if (index < index_t{0}) {
            index = static_cast<index_t>(bound + 1 -
                                         static_cast<std::size_t>(-index));
        }
    }

    if (index < index_t{0} || bound < static_cast<std::size_t>(index)) {
        throw std::out_of_range(
            "Index is out of bounds: " + std::to_string(index) +
            " is not in [0, " + std::to_string(bound) + ").");
    }

    return index;
}

template <std::integral in_t, in_t... ins, typename compare_t>
[[nodiscard]] consteval auto
get_sorted_array(std::integer_sequence<in_t, ins...>,
                 compare_t compare) noexcept {
    auto arr = std::array{ins...};
    std::sort(arr.begin(), arr.end(), compare);
    return arr;
}

template <std::size_t rank, std::integral axes_t, axes_t... axes,
          typename compare_t>
[[nodiscard]] consteval auto
get_sorted_axes(std::integer_sequence<axes_t, axes...>,
                compare_t compare) noexcept {
    constexpr auto arr =
        get_sorted_array(std::index_sequence<static_cast<std::size_t>(
                             bounding_index<axes_t>(axes, rank - 1))...>{},
                         compare);

    if constexpr (1 < arr.size()) {
        static_assert(
            [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return ((arr[Is] != arr[Is + 1]) && ...);
            }(std::make_index_sequence<arr.size() - 1>{}),
            "Duplicate axes are not allowed.");
    }

    return arr;
}

template <typename value_t, std::size_t size>
[[nodiscard]] constexpr bool contains(const std::array<value_t, size> &array,
                                      const value_t &value) noexcept {
    for (const auto element : array) {
        if (element == value) {
            return true;
        }
    }

    return false;
}

} // namespace mdtensor::core
