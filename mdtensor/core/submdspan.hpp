/**
 * @file
 * @brief Submdspan utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "convert.hpp"
#include "type.hpp"

namespace mdtensor {

template <typename in_t, typename... slices_t>
[[nodiscard]] inline constexpr auto submdspan(in_t &&in,
                                              slices_t &&...slices) noexcept {
    return std::experimental::submdspan(core::to_mdspan(std::forward<in_t>(in)),
                                        std::forward<slices_t>(slices)...);
}

namespace core {

template <size_t lspace, size_t rspace, typename in_t, typename... slices_t>
[[nodiscard]] inline constexpr auto
submdspan_with_space(in_t &&in, slices_t &&...slices) noexcept {
    return [&]<size_t... Is, size_t... Js>(std::index_sequence<Is...>,
                                           std::index_sequence<Js...>) {
        return submdspan(std::forward<in_t>(in), ((void)Is, full_extent)...,
                         std::forward<slices_t>(slices)...,
                         ((void)Js, full_extent)...);
    }(std::make_index_sequence<lspace>{}, std::make_index_sequence<rspace>{});
}

template <size_t lspace = 0, typename in_t, typename... slices_t>
[[nodiscard]] inline constexpr auto
submdspan_from_left(in_t &&in, slices_t &&...slices) noexcept {
    using in_base_t = std::remove_reference_t<in_t>;

    constexpr size_t rspace =
        to_mdspan_t<in_base_t>::rank() - (lspace + sizeof...(slices_t));

    return submdspan_with_space<lspace, rspace>(
        std::forward<in_t>(in), std::forward<slices_t>(slices)...);
}

template <size_t rspace = 0, typename in_t, typename... slices_t>
[[nodiscard]] inline constexpr auto
submdspan_from_right(in_t &&in, slices_t &&...slices) noexcept {
    using in_base_t = std::remove_reference_t<in_t>;

    constexpr size_t lspace =
        to_mdspan_t<in_base_t>::rank() - (rspace + sizeof...(slices_t));

    return submdspan_with_space<lspace, rspace>(
        std::forward<in_t>(in), std::forward<slices_t>(slices)...);
}

} // namespace core
} // namespace mdtensor
