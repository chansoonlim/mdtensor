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

template <typename dtype = void, bool keepdims = false, bool floating = false,
          std::integral axes_t, axes_t... axes, std::size_t... uranks,
          extents_c uout_exts_t>
[[nodiscard]] constexpr auto
resolve_reduced_output(auto &&out, std::integer_sequence<axes_t, axes...>,
                       std::index_sequence<uranks...>, uout_exts_t &&uout_exts,
                       auto &&...ins) {
    if constexpr (floating) {
        // Ensure that the output type is at least float precision
        using value_t = output_value_t<dtype, float, decltype(ins)...>;

        const auto out_md = resolve_reduced_output<value_t, keepdims, false>(
            std::forward<decltype(out)>(out),
            std::integer_sequence<axes_t, axes...>{},
            std::index_sequence<uranks...>{},
            std::forward<uout_exts_t>(uout_exts),
            std::forward<decltype(ins)>(ins)...);

        // Check that resolved output type is at least float precision
        static_assert(core::floating_point_c<
                          typename to_mdspan_t<decltype(out_md)>::value_type>,
                      "Resolved output type must be at least float precision.");

        return out_md;

    } else {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return make_reduced_tensor<dtype, keepdims>(
                std::integer_sequence<axes_t, axes...>{},
                std::index_sequence<uranks...>{},
                std::forward<uout_exts_t>(uout_exts),
                std::forward<decltype(ins)>(ins)...);

        } else {
            const auto out_mds =
                to_output_mdspan(std::forward<decltype(out)>(out));

            // TODO: check same extents with expected extents
            // TODO: check out_md type is enough precision for dtype, if dtype
            // is not void
            // TODO: use resolve_output.

            return out_mds;
        }
    }
}

template <typename dtype = void, bool keepdims = false, bool floating = false,
          std::integral axes_t, axes_t... axes, extents_c uout_exts_t>
[[nodiscard]] constexpr auto
resolve_reduced_output(auto &&out, std::integer_sequence<axes_t, axes...>,
                       uout_exts_t &&uout_exts, auto &&...ins) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return resolve_reduced_output<dtype, keepdims, floating>(
            std::forward<decltype(out)>(out),
            std::integer_sequence<axes_t, axes...>{},
            std::index_sequence<((void)Is, 0)...>{},
            std::forward<uout_exts_t>(uout_exts),
            std::forward<decltype(ins)>(ins)...);
    }(std::make_index_sequence<sizeof...(ins)>{});
}

// TODO: develop resolve_reduced_outputs

} // namespace mdtensor::core
