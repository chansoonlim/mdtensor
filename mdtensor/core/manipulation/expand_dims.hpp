/**
 * @file
 * @brief Expand dimensions utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "reshape.hpp"

namespace mdtensor::core {

template <std::integral axes_t, axes_t... axes>
[[nodiscard]] constexpr auto
expand_dims(auto &&in, std::integer_sequence<axes_t, axes...>) {
    const auto in_mds = to_mdspan(std::forward<decltype(in)>(in));

    return reshape<Copy::FALSE>(
        in_mds, expand_extents_dims(in_mds.extents(),
                                    std::integer_sequence<axes_t, axes...>{}));
}

template <std::int64_t... axes>
[[nodiscard]] constexpr auto expand_dims(auto &&in) {
    return expand_dims(std::forward<decltype(in)>(in),
                       std::integer_sequence<std::int64_t, axes...>{});
}

} // namespace mdtensor::core
