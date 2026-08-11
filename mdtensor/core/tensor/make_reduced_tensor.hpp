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
#include "tensor.hpp"

namespace mdtensor::core {

template <typename dtype = void, bool keepdims = false, std::integral axes_t,
          axes_t... axes, std::size_t... uranks, extents_c uout_exts_t>
[[nodiscard]] constexpr auto
make_reduced_tensor(std::integer_sequence<axes_t, axes...>,
                    std::index_sequence<uranks...>, uout_exts_t &&uout_exts,
                    auto &&...ins) {
    static_assert(sizeof...(ins) > 0, "At least one input must be provided.");
    static_assert(sizeof...(uranks) == sizeof...(ins),
                  "Number of uranks must match number of inputs.");

    // calculate input broadcasted extents
    const auto ins_bexts = detail::get_broadcast_extents(
        std::index_sequence<uranks...>{},
        to_const_mdspan(std::forward<decltype(ins)>(ins))...);

    using ins_bexts_t = decltype(ins_bexts);

    // get sorted array
    constexpr auto axes_sorted = [&]() {
        if constexpr (ins_bexts_t::rank() == 0) {
            return std::array<std::size_t, 0>{};

        } else if constexpr (sizeof...(axes) == 0) {
            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return std::array{(ins_bexts_t::rank() - 1 - Is)...};
            }(std::make_index_sequence<ins_bexts_t::rank()>{});

        } else {
            return get_sorted_axes<ins_bexts_t::rank()>(
                std::integer_sequence<axes_t, axes...>{},
                std::greater<std::size_t>{});
        }
    }();

    // generate out_bexts
    constexpr auto not_axes_size = ins_bexts_t::rank() - axes_sorted.size();

    const auto out_bexts = [&]() {
        using index_t = typename ins_bexts_t::index_type;

        if constexpr (keepdims) {
            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return extents<index_t,
                               (contains(axes_sorted, Is)
                                    ? std::size_t{1}
                                    : ins_bexts_t::static_extent(Is))...>{
                    (contains(axes_sorted, Is)
                         ? index_t{1}
                         : static_cast<index_t>(ins_bexts.extent(Is)))...};
            }(std::make_index_sequence<ins_bexts_t::rank()>{});

        } else {
            // generate unselected axes sequence
            constexpr auto not_axes_arr = [&]() {
                auto not_axes_arr = std::array<std::size_t, not_axes_size>{};

                std::size_t not_axes_idx = 0;
                for (std::size_t i = 0; i < ins_bexts_t::rank(); i++) {
                    if (!contains(axes_sorted, i)) {
                        not_axes_arr[not_axes_idx++] = i;
                    }
                }

                return not_axes_arr;
            }();

            return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return extents<index_t,
                               ins_bexts_t::static_extent(not_axes_arr[Is])...>{
                    ins_bexts.extent(not_axes_arr[Is])...};
            }(std::make_index_sequence<not_axes_arr.size()>{});
        }
    }();

    // generate out
    using value_t = output_value_t<dtype, decltype(ins)...>;

    return make_tensor<value_t>(
        compose_extents(out_bexts, std::forward<uout_exts_t>(uout_exts)));
}

// TODO: develop make_reduce_outputs

} // namespace mdtensor::core
