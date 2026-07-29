/**
 * @file
 * @brief Extents utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "type.hpp"

namespace mdtensor {
namespace core {

template <extents_c in_t>
[[nodiscard]] inline constexpr size_t extents_size(in_t &&in) noexcept {
    using base_t = std::remove_cvref_t<in_t>;

    if constexpr (base_t::rank() == 0) {
        return 0;

    } else if constexpr (base_t::rank_dynamic() == 0) {
        return []<size_t... Is>(std::index_sequence<Is...>) {
            return (base_t::static_extent(Is) * ...);
        }(std::make_index_sequence<base_t::rank()>{});

    } else {
        return [&]<size_t... Is>(std::index_sequence<Is...>) {
            return (in.extent(Is) * ...);
        }(std::make_index_sequence<base_t::rank()>{});
    }
}

template <extents_c in_t>
[[nodiscard]] inline constexpr bool same_extents(in_t &&in) noexcept {
    return true;
}

template <extents_c in1_t, extents_c in2_t, extents_c... ins_t>
[[nodiscard]] inline constexpr bool same_extents(in1_t &&in1, in2_t &&in2,
                                                 ins_t &&...ins) noexcept {
    // FIXME: direct comparision of extents has issues:
    // It cannot capture the case where the ranks are different.

    if constexpr (std::remove_cvref_t<in1_t>::rank() !=
                  std::remove_cvref_t<in2_t>::rank()) {
        return false;

    } else if (std::forward<in1_t>(in1) != std::forward<in2_t>(in2)) {
        return false;
    }

    if constexpr (sizeof...(ins_t) != 0) {
        return same_extents(std::forward<in2_t>(in2),
                            std::forward<ins_t>(ins)...);

    } else {
        return true;
    }
}

template <size_t offset, size_t rank, extents_c in_t>
[[nodiscard]] inline constexpr auto slice_extents(in_t &&in) noexcept {
    using base_t = std::remove_cvref_t<in_t>;

    static_assert(base_t::rank() >= offset + rank,
                  "Incompatible offset and rank for slicing.");

    return [&]<size_t... Is>(std::index_sequence<Is...>) {
        return stdex::extents<typename base_t::index_type,
                              base_t::static_extent(offset + Is)...>{
            in.extent(offset + Is)...};
    }(std::make_index_sequence<rank>{});
}

template <size_t rank, extents_c in_t>
[[nodiscard]] inline constexpr auto
slice_extents_from_left(in_t &&in) noexcept {
    return slice_extents<0, rank>(std::forward<in_t>(in));
}

template <size_t rank, extents_c in_t>
[[nodiscard]] inline constexpr auto
slice_extents_from_right(in_t &&in) noexcept {
    return slice_extents<std::remove_cvref_t<in_t>::rank() - rank, rank>(
        std::forward<in_t>(in));
}

template <extents_c in1_t, extents_c in2_t, extents_c... ins_t>
[[nodiscard]] inline constexpr auto compose_extents(in1_t &&in1, in2_t &&in2,
                                                    ins_t &&...ins) noexcept {
    using base1_t = std::remove_cvref_t<in1_t>;
    using base2_t = std::remove_cvref_t<in2_t>;
    using index_t = common_index_type_t<typename base1_t::index_type,
                                        typename base2_t::index_type>;

    const auto cexts =
        [&]<size_t... Is, size_t... Js>(std::index_sequence<Is...>,
                                        std::index_sequence<Js...>) {
            return stdex::extents<index_t, base1_t::static_extent(Is)...,
                                  base2_t::static_extent(Js)...>{
                static_cast<index_t>(in1.extent(Is))...,
                static_cast<index_t>(in2.extent(Js))...};
        }(std::make_index_sequence<base1_t::rank()>{},
          std::make_index_sequence<base2_t::rank()>{});

    if constexpr (sizeof...(ins_t) == 0) {
        return cexts;

    } else {
        return compose_extents(cexts, std::forward<ins_t>(ins)...);
    }
}

} // namespace core
} // namespace mdtensor
