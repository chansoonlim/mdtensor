/**
 * @file
 * @brief Output utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "make_reduced_tensor.hpp"
#include "resolve_output.hpp"

namespace mdtensor::core {

template <typename dtype = void, bool keepdims = false, std::integral axes_t,
          axes_t... axes, std::size_t... uranks, extents_c uout_exts_t>
[[nodiscard]] constexpr auto
resolve_reduced_output(auto &&out, std::integer_sequence<axes_t, axes...>,
                       std::index_sequence<uranks...>, uout_exts_t &&uout_exts,
                       auto &&...ins) {
    if constexpr (core::nullopt_t_c<decltype(out)>) {
        return make_reduced_tensor<dtype, keepdims>(
            std::integer_sequence<axes_t, axes...>{},
            std::index_sequence<uranks...>{},
            std::forward<uout_exts_t>(uout_exts),
            std::forward<decltype(ins)>(ins)...);

    } else {
        using value_t = calc_type_t<dtype, decltype(ins)...>;

        const auto out_exts = make_reduced_extents<keepdims>(
            std::integer_sequence<axes_t, axes...>{},
            std::index_sequence<uranks...>{},
            std::forward<uout_exts_t>(uout_exts),
            std::forward<decltype(ins)>(ins)...);

        return resolve_output<value_t>(std::forward<decltype(out)>(out),
                                       out_exts);
    }
}

template <typename dtype = void, bool keepdims = false, std::integral axes_t,
          axes_t... axes, extents_c uout_exts_t>
[[nodiscard]] constexpr auto
resolve_reduced_output(auto &&out, std::integer_sequence<axes_t, axes...>,
                       uout_exts_t &&uout_exts, auto &&...ins) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return resolve_reduced_output<dtype, keepdims>(
            std::forward<decltype(out)>(out),
            std::integer_sequence<axes_t, axes...>{},
            std::index_sequence<((void)Is, 0)...>{},
            std::forward<uout_exts_t>(uout_exts),
            std::forward<decltype(ins)>(ins)...);
    }(std::make_index_sequence<sizeof...(ins)>{});
}

// TODO: develop resolve_reduced_outputs

} // namespace mdtensor::core
