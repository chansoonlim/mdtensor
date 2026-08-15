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

template <typename dtype = void, bool keepdims = false, std::integral axes_t,
          axes_t... axes, std::size_t... uranks>
[[nodiscard]] constexpr auto resolve_reduced_outputs(
    auto &&out_tuple, std::integer_sequence<axes_t, axes...>,
    std::index_sequence<uranks...>, auto &&uout_exts_tuple, auto &&...ins) {
    static_assert(sizeof...(uranks) == sizeof...(ins),
                  "Number of uranks must match number of inputs.");

    if constexpr (core::nullopt_t_c<decltype(out_tuple)>) {
        return make_reduced_tensors<dtype, keepdims>(
            std::integer_sequence<axes_t, axes...>{},
            std::index_sequence<uranks...>{},
            std::forward<decltype(uout_exts_tuple)>(uout_exts_tuple),
            std::forward<decltype(ins)>(ins)...);

    } else {
        constexpr std::size_t ins_num = sizeof...(uranks);
        constexpr std::size_t outs_num =
            std::tuple_size_v<std::remove_cvref_t<decltype(uout_exts_tuple)>>;

        if constexpr (ins_num == 0) {
            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return std::tuple{resolve_output<dtype>(
                    std::get<Is>(out_tuple), std::get<Is>(uout_exts_tuple))...};
            }(std::make_index_sequence<outs_num>{});

        } else {
            using value_t = calc_type_t<dtype, decltype(ins)...>;

            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return std::tuple{resolve_output<value_t>(
                    std::get<Is>(out_tuple),
                    make_reduced_extents<keepdims>(
                        std::integer_sequence<axes_t, axes...>{},
                        std::index_sequence<uranks...>{},
                        std::get<Is>(uout_exts_tuple),
                        std::forward<decltype(ins)>(ins)...))...};
            }(std::make_index_sequence<outs_num>{});
        }
    }
}

template <typename dtype = void, bool keepdims = false, std::integral axes_t,
          axes_t... axes>
[[nodiscard]] constexpr auto
resolve_reduced_outputs(auto &&out_tuple,
                        std::integer_sequence<axes_t, axes...>,
                        auto &&uout_exts_tuple, auto &&...ins) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return resolve_reduced_outputs<dtype, keepdims>(
            std::forward<decltype(out_tuple)>(out_tuple),
            std::integer_sequence<axes_t, axes...>{},
            std::index_sequence<((void)Is, 0)...>{},
            std::forward<decltype(uout_exts_tuple)>(uout_exts_tuple),
            std::forward<decltype(ins)>(ins)...);
    }(std::make_index_sequence<sizeof...(ins)>{});
}

} // namespace mdtensor::core
