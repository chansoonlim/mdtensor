/**
 * @file
 * @brief Range generation utilities for mdtensor (arange).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {

template <typename data_t = void, typename start_t, typename stop_t,
          typename step_t = double>
[[nodiscard]] inline constexpr auto arange(start_t &&start, stop_t &&stop,
                                           step_t &&step = step_t{1}) noexcept {
    using value_t =
        std::conditional_t<!std::is_void_v<data_t>, data_t,
                           core::common_data_type_t<start_t, stop_t>>;

    const size_t num = std::ceil((stop - start) / step);
    const value_t step_actual =
        static_cast<value_t>(start + step) - static_cast<value_t>(start);

    auto out = core::make_container<value_t>(mdtensor::dims<1>{num});

    out(0) = start;
    for (size_t i = 1; i < num; i++) {
        out(i) = out(i - 1) + step_actual;
    }

    return out;
}

template <typename data_t = void, typename stop_t>
[[nodiscard]] inline constexpr auto arange(stop_t &&stop) noexcept {
    return arange<data_t>(0, std::forward<stop_t>(stop));
}

// TODO: Develop arange that works in compile-time contexts (e.g., constexpr
// mdarray) and/or with static extents.

} // namespace mdtensor
