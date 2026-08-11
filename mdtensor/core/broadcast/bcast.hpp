/**
 * @file
 * @brief Broadcast utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "bcast_extents.hpp"
#include "bcast_to.hpp"

namespace mdtensor::core {
namespace detail {

template <std::size_t... uranks, mdspan_c... ios_t>
[[nodiscard]] constexpr auto
get_broadcast_extents(std::index_sequence<uranks...>, ios_t &&...ios) {
    static_assert(sizeof...(uranks) == sizeof...(ios_t),
                  "Number of uranks must match number of inputs.");
    static_assert(((ios.rank() >= uranks) && ...),
                  "Input rank must be greater than or equal to urank.");

    return broadcast_extents(
        slice_extents_from_left<ios.rank() - uranks>(ios.extents())...);
}

} // namespace detail

template <std::size_t... uranks, bool... bcast>
[[nodiscard]] constexpr auto broadcast(std::index_sequence<uranks...>,
                                       std::integer_sequence<bool, bcast...>,
                                       auto &&...ios) {
    static_assert(sizeof...(uranks) == sizeof...(ios));
    static_assert(sizeof...(bcast) == sizeof...(ios));

    // calculate mdspans for inputs and outputs
    const auto ios_mds =
        std::make_tuple(to_mdspan(std::forward<decltype(ios)>(ios))...);

    // calculate broadcasted extents
    constexpr auto ur = std::array{uranks...};

    const auto bexts = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return detail::get_broadcast_extents(std::index_sequence<ur[Is]...>{},
                                             std::get<Is>(ios_mds)...);
    }(std::make_index_sequence<sizeof...(ios)>{});

    // calculate broadcasted mdspans
    constexpr auto bc = std::array{bcast...};

    return std::make_tuple(
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            const auto get_broadcasted = [&]<std::size_t I>() {
                if constexpr (!bc[I]) {
                    // change to mdspan without broadcasting
                    return std::get<I>(ios_mds);

                } else if constexpr (bexts.rank() == 0) {
                    // change to const mdspan without broadcasting
                    return to_const_mdspan(std::get<I>(ios_mds));

                } else {
                    // broadcast to const mdspan
                    return broadcast_to(
                        std::get<I>(ios_mds),
                        compose_extents(bexts,
                                        slice_extents_from_right<ur[I]>(
                                            std::get<I>(ios_mds).extents())));
                }
            };

            return std::make_tuple(
                get_broadcasted.template operator()<Is>()...);
        }(std::make_index_sequence<sizeof...(ios)>{}),
        bexts);
}

} // namespace mdtensor::core
