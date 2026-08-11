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

namespace mdtensor::core {

constexpr auto full_extent = stdex::full_extent;

template <std::size_t start, std::size_t end>
using slice =
    stdex::strided_slice<std::integral_constant<std::size_t, start>,
                         std::integral_constant<std::size_t, end - start>,
                         std::integral_constant<std::size_t, 1>>;

[[nodiscard]] constexpr auto submdspan(auto &&io, auto &&...slices) {
    return stdex::submdspan(to_mdspan(std::forward<decltype(io)>(io)),
                            std::forward<decltype(slices)>(slices)...);
}

template <std::size_t lspace = 0, std::size_t rspace = 0>
[[nodiscard]] constexpr auto submdspan_with_space(auto &&io, auto &&...slices) {
    return [&]<std::size_t... Is, std::size_t... Js>(
               std::index_sequence<Is...>, std::index_sequence<Js...>) {
        return submdspan(to_mdspan(std::forward<decltype(io)>(io)),
                         ((void)Is, full_extent)...,
                         std::forward<decltype(slices)>(slices)...,
                         ((void)Js, full_extent)...);
    }(std::make_index_sequence<lspace>{}, std::make_index_sequence<rspace>{});
}

template <std::size_t lspace = 0>
[[nodiscard]] constexpr auto submdspan_from_left(auto &&io, auto &&...slices) {
    using base_t = std::remove_reference_t<decltype(io)>;

    constexpr std::size_t rspace =
        to_mdspan_t<base_t>::rank() - (lspace + sizeof...(slices));

    return submdspan_with_space<lspace, rspace>(
        std::forward<decltype(io)>(io),
        std::forward<decltype(slices)>(slices)...);
}

template <std::size_t rspace = 0>
[[nodiscard]] constexpr auto submdspan_from_right(auto &&io, auto &&...slices) {
    using base_t = std::remove_reference_t<decltype(io)>;

    constexpr std::size_t lspace =
        to_mdspan_t<base_t>::rank() - (rspace + sizeof...(slices));

    return submdspan_with_space<lspace, rspace>(
        std::forward<decltype(io)>(io),
        std::forward<decltype(slices)>(slices)...);
}

} // namespace mdtensor::core
