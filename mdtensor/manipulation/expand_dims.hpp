/**
 * @file
 * @brief Dimension expansion utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {

template <std::integral axes_t, axes_t... axes>
[[nodiscard]] constexpr auto
expand_dims(auto &&in, std::integer_sequence<axes_t, axes...>) {
    return core::expand_dims(std::forward<decltype(in)>(in),
                             std::integer_sequence<axes_t, axes...>{});
}

template <std::int64_t... axes>
[[nodiscard]] constexpr auto expand_dims(auto &&in) {
    return core::expand_dims<axes...>(std::forward<decltype(in)>(in));
}

} // namespace mdtensor
