/**
 * @file
 * @brief Output utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "make_broadcasted_tensor.hpp"
#include "resolve_output.hpp"

namespace mdtensor::core {

template <typename dtype = void, std::size_t... uranks, extents_c uout_exts_t>
[[nodiscard]] constexpr auto
resolve_broadcasted_output(auto &&out, std::index_sequence<uranks...>,
                           uout_exts_t &&uout_exts, auto &&...ins) {
    if constexpr (core::nullopt_t_c<decltype(out)>) {
        return make_broadcasted_tensor<dtype>(
            std::index_sequence<uranks...>{},
            std::forward<uout_exts_t>(uout_exts),
            std::forward<decltype(ins)>(ins)...);

    } else {
        using value_t = calc_type_t<dtype, decltype(ins)...>;

        const auto out_exts =
            make_broadcasted_extents(std::index_sequence<uranks...>{},
                                     std::forward<uout_exts_t>(uout_exts),
                                     std::forward<decltype(ins)>(ins)...);

        return resolve_output<value_t>(std::forward<decltype(out)>(out),
                                       out_exts);
    }
}

template <typename dtype = void, extents_c uout_exts_t>
[[nodiscard]] constexpr auto
resolve_broadcasted_output(auto &&out, uout_exts_t &&uout_exts, auto &&...ins) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return resolve_broadcasted_output<dtype>(
            std::forward<decltype(out)>(out),
            std::index_sequence<((void)Is, 0)...>{},
            std::forward<uout_exts_t>(uout_exts),
            std::forward<decltype(ins)>(ins)...);
    }(std::make_index_sequence<sizeof...(ins)>{});
}

// TODO: develop resolve_broadcasted_outputs

} // namespace mdtensor::core
