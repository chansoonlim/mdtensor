/**
 * @file
 * @brief Flatten utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "reshape.hpp"

namespace mdtensor {
namespace detail {

template <std::size_t... Extents>
[[nodiscard]] consteval std::size_t flatten_static_extent() noexcept {
    static_assert(sizeof...(Extents) > 0,
                  "At least one extent must be provided for flattening.");

    if constexpr (((Extents == core::dyn) || ...)) {
        return core::dyn;

    } else {
        return (Extents * ...);
    }
}

template <typename index_t, std::convertible_to<index_t>... exts_t>
[[nodiscard]] constexpr index_t flatten_extent(exts_t &&...exts) noexcept {
    static_assert(sizeof...(exts) > 0,
                  "At least one extent must be provided for flattening.");

    return (exts * ...);
}

template <core::extents_c in_t>
[[nodiscard]] constexpr auto flatten_extents(in_t &&in) noexcept {
    using base_t = std::remove_cvref_t<in_t>;
    using index_t = typename base_t::index_type;

    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return core::extents<
            typename base_t::index_type,
            flatten_static_extent<base_t::static_extent(Is)...>()>{
            flatten_extent<index_t>(in.extent(Is)...)};
    }(std::make_index_sequence<base_t::rank()>{});
}

} // namespace detail

[[nodiscard]] constexpr auto flatten(auto &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    return reshape<core::Copy::TRUE>(in_mds,
                                     detail::flatten_extents(in_mds.extents()));
}

} // namespace mdtensor
