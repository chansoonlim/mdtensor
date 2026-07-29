/**
 * @file
 * @brief Transpose utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {

template <std::integral axes_t, axes_t... axes>
[[nodiscard]] constexpr auto transpose(auto &&in,
                                       std::integer_sequence<axes_t, axes...>) {
    const auto in_mds = core::to_mdspan(std::forward<decltype(in)>(in));
    using in_mds_t = decltype(in_mds);

    constexpr std::size_t rank = in_mds_t::rank();

    if constexpr (rank < 2) {
        return in_mds;

    } else if constexpr (sizeof...(axes) == 0) {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return transpose(std::forward<decltype(in)>(in),
                             std::index_sequence<(rank - 1 - Is)...>{});
        }(std::make_index_sequence<rank>{});

    } else {
        static_assert(sizeof...(axes) == rank,
                      "Number of axes must match rank.");

        constexpr auto axes_arr = std::array{static_cast<std::size_t>(
            core::bounding_index<axes_t>(axes, rank - 1))...};

        static_assert(
            [&]() {
                for (std::size_t i = 0; i < axes_arr.size(); i++) {
                    for (std::size_t j = i + 1; j < axes_arr.size(); j++) {
                        if (axes_arr[i] == axes_arr[j]) {
                            return false;
                        }
                    }
                }
                return true;
            }(),
            "Axes must be unique.");

        const auto new_extents =
            [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return core::extents<typename in_mds_t::index_type,
                                     in_mds_t::static_extent(axes_arr[Is])...>{
                    in_mds.extent(axes_arr[Is])...};
            }(std::make_index_sequence<axes_arr.size()>{});

        const auto new_strides =
            [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return std::array<typename in_mds_t::index_type, rank>{
                    in_mds.stride(axes_arr[Is])...};
            }(std::make_index_sequence<axes_arr.size()>{});

        return core::mdspan<typename in_mds_t::element_type,
                            std::remove_cvref_t<decltype(new_extents)>,
                            core::stdex::layout_stride,
                            typename in_mds_t::accessor_type>{
            in_mds.data_handle(),
            core::stdex::layout_stride::mapping{new_extents, new_strides}};
    }
}

template <std::int64_t... axes>
[[nodiscard]] constexpr auto transpose(auto &&in) {
    return transpose(std::forward<decltype(in)>(in),
                     std::integer_sequence<std::int64_t, axes...>{});
}

} // namespace mdtensor
