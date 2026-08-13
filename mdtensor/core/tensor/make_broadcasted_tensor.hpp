/**
 * @file
 * @brief Output utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../broadcast/broadcast.hpp"
#include "make_tensor.hpp"
#include "output_value_type.hpp"

namespace mdtensor::core {

template <typename dtype = void, std::size_t... uranks, extents_c uout_exts_t>
[[nodiscard]] constexpr auto
make_broadcasted_tensor(std::index_sequence<uranks...>, uout_exts_t &&uout_exts,
                        auto &&...ins) {
    static_assert(sizeof...(uranks) == sizeof...(ins),
                  "Number of uranks must match number of inputs.");

    constexpr std::size_t ins_num = sizeof...(uranks);

    if constexpr (ins_num == 0) {
        return make_tensor<dtype>(std::forward<uout_exts_t>(uout_exts));

    } else {
        using value_t = calc_type_t<dtype, decltype(ins)...>;

        // calculate broadcasted extents
        const auto bexts = get_broadcast_extents(
            std::index_sequence<uranks...>{},
            to_const_mdspan(std::forward<decltype(ins)>(ins))...);

        return make_tensor<value_t>(
            compose_extents(bexts, std::forward<uout_exts_t>(uout_exts)));
    }
}

template <typename dtype = void, extents_c uout_exts_t>
[[nodiscard]] constexpr auto make_broadcasted_tensor(uout_exts_t &&uout_exts,
                                                     auto &&...ins) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return make_broadcasted_tensor<dtype>(
            std::index_sequence<((void)Is, 0)...>{},
            std::forward<uout_exts_t>(uout_exts),
            std::forward<decltype(ins)>(ins)...);
    }(std::make_index_sequence<sizeof...(ins)>{});
}

template <typename dtype = void, std::size_t... uranks>
[[nodiscard]] constexpr auto
make_broadcasted_tensors(std::index_sequence<uranks...>, auto &&uout_exts_tuple,
                         auto &&...ins) {
    static_assert(sizeof...(uranks) == sizeof...(ins),
                  "Number of uranks must match number of inputs.");

    constexpr std::size_t ins_num = sizeof...(uranks);
    constexpr std::size_t outs_num =
        std::tuple_size_v<std::remove_cvref_t<decltype(uout_exts_tuple)>>;

    if constexpr (ins_num == 0) {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return std::tuple{
                make_tensor<dtype>(std::get<Is>(uout_exts_tuple))...};
        }(std::make_index_sequence<outs_num>{});

    } else {
        using value_t = calc_type_t<dtype, decltype(ins)...>;

        // calculate broadcasted extents
        const auto bexts = get_broadcast_extents(
            std::index_sequence<uranks...>{},
            to_const_mdspan(std::forward<decltype(ins)>(ins))...);

        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return std::tuple{make_tensor<value_t>(
                compose_extents(bexts, std::get<Is>(uout_exts_tuple)))...};
        }(std::make_index_sequence<outs_num>{});
    }
}

template <typename dtype = void>
[[nodiscard]] constexpr auto make_broadcasted_tensors(auto &&uout_exts_tuple,
                                                      auto &&...ins) {
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return make_broadcasted_tensors<dtype>(
            std::index_sequence<((void)Is, 0)...>{},
            std::forward<decltype(uout_exts_tuple)>(uout_exts_tuple),
            std::forward<decltype(ins)>(ins)...);
    }(std::make_index_sequence<sizeof...(ins)>{});
}

} // namespace mdtensor::core
