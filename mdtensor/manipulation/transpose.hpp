/**
 * @file
 * @brief Transpose utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include <array>
#include <type_traits>
#include <utility>

#include "../core/core.hpp"

namespace mdtensor {

template <typename in_t, std::integral AxesType, AxesType... Axes>
[[nodiscard]] inline constexpr auto
transpose(in_t &&in, std::integer_sequence<AxesType, Axes...>) noexcept {
    const auto in_mds = core::to_mdspan(std::forward<in_t>(in));
    using in_mds_t = decltype(in_mds);

    constexpr size_t rank = in_mds_t::rank();

    static_assert(sizeof...(Axes) == rank, "Number of axes must match rank.");

    if constexpr (rank == 0 || rank == 1) {
        return in_mds;

    } else {
        constexpr auto axes = std::array<size_t, rank>{static_cast<size_t>(
            ((Axes % static_cast<AxesType>(rank)) + (rank)) % rank)...};

        const auto new_extents = [&]<size_t... Is>(std::index_sequence<Is...>) {
            return extents<typename in_mds_t::index_type,
                           in_mds_t::static_extent(axes[Is])...>{
                in_mds.extent(axes[Is])...};
        }(std::make_index_sequence<rank>{});

        const auto new_strides = [&]<size_t... Is>(std::index_sequence<Is...>) {
            return std::array<typename in_mds_t::index_type, rank>{
                in_mds.stride(axes[Is])...};
        }(std::make_index_sequence<rank>{});

        return mdspan<typename in_mds_t::element_type,
                      std::remove_cvref_t<decltype(new_extents)>, layout_stride,
                      typename in_mds_t::accessor_type>{
            in_mds.data_handle(),
            layout_stride::mapping{new_extents, new_strides}};
    }
}

template <typename in_t>
[[nodiscard]] inline constexpr auto transpose(in_t &&in) noexcept {
    const auto in_mds = core::to_mdspan(std::forward<in_t>(in));
    using in_mds_t = decltype(in_mds);

    constexpr size_t rank = in_mds_t::rank();

    return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return transpose(std::forward<in_t>(in),
                         std::integer_sequence<int64_t, rank - 1 - Is...>{});
    }(std::make_index_sequence<rank>{});
}

} // namespace mdtensor
