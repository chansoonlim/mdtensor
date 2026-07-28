/**
 * @file
 * @brief Submdspan utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "mdspan.hpp"

namespace mdtensor {
namespace core {

template <size_t lspace = 0, size_t rspace = 0, typename in_t,
          typename... slices_t>
[[nodiscard]] inline constexpr auto submdspan(in_t &&in,
                                              slices_t &&...slices) noexcept {
    return [&]<size_t... Is, size_t... Js>(std::index_sequence<Is...>,
                                           std::index_sequence<Js...>) {
        return std::experimental::submdspan(to_mdspan(std::forward<in_t>(in)),
                                            ((void)Is, stdex::full_extent)...,
                                            std::forward<slices_t>(slices)...,
                                            ((void)Js, stdex::full_extent)...);
    }(std::make_index_sequence<lspace>{}, std::make_index_sequence<rspace>{});
}

template <size_t lspace = 0, typename in_t, typename... slices_t>
[[nodiscard]] inline constexpr auto
submdspan_from_left(in_t &&in, slices_t &&...slices) noexcept {
    using in_base_t = std::remove_reference_t<in_t>;

    constexpr size_t rspace =
        to_mdspan_t<in_base_t>::rank() - (lspace + sizeof...(slices_t));

    return submdspan<lspace, rspace>(std::forward<in_t>(in),
                                     std::forward<slices_t>(slices)...);
}

template <size_t rspace = 0, typename in_t, typename... slices_t>
[[nodiscard]] inline constexpr auto
submdspan_from_right(in_t &&in, slices_t &&...slices) noexcept {
    using in_base_t = std::remove_reference_t<in_t>;

    constexpr size_t lspace =
        to_mdspan_t<in_base_t>::rank() - (rspace + sizeof...(slices_t));

    return submdspan<lspace, rspace>(std::forward<in_t>(in),
                                     std::forward<slices_t>(slices)...);
}

} // namespace core
} // namespace mdtensor
