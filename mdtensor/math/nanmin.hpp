/**
 * @file
 * @brief Minimum reduction without NaN utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../logic/isnan.hpp"
#include "../logic/logical_not.hpp"
#include "min.hpp"

namespace mdtensor {

template <typename dtype = void, bool keepdims = false,
          core::Backend backend = core::Backend::AUTO, std::integral axes_t,
          axes_t... axes, typename out_t = std::nullopt_t,
          typename initial_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto
nanmin(auto &&in, std::integer_sequence<axes_t, axes...>,
       out_t &&out = out_t{std::nullopt},
       initial_t &&initial = initial_t{std::nullopt},
       where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));
    const auto where_mds =
        core::to_const_mdspan(std::forward<decltype(where)>(where));

    const auto is_not_nan = logical_not(isnan(in_mds, std::nullopt, where_mds));

    const auto mask = [&]() {
        if constexpr (core::nullopt_t_c<decltype(where)>) {
            return is_not_nan;

        } else {
            return logical_and(is_not_nan, where_mds);
        }
    }();

    return min<dtype, keepdims, backend>(
        in_mds, std::integer_sequence<axes_t, axes...>{},
        std::forward<decltype(out)>(out),
        std::forward<decltype(initial)>(initial), mask);
}

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename initial_t = std::nullopt_t,
          typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto
nanmin(auto &&in, initial_t &&initial = initial_t{std::nullopt},
       where_t &&where = where_t{std::nullopt}) {
    return nanmin<dtype, false, backend>(
        std::forward<decltype(in)>(in), std::index_sequence<>{}, std::nullopt,
        std::forward<decltype(initial)>(initial),
        std::forward<decltype(where)>(where));
}

} // namespace mdtensor
