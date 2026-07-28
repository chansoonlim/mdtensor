/**
 * @file
 * @brief Identity matrix (eye) creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "empty.hpp"

namespace mdtensor {
namespace detail {

template <typename in_t> inline constexpr void eye_impl(in_t &&in) {
    using in_base_t = std::remove_cvref_t<in_t>;
    using index_t = typename in_base_t::index_type;

    for (index_t i = 0; i < in.extent(0); i++) {
        for (index_t j = 0; j < in.extent(1); j++) {
            in(i, j) = (i == j) ? 1 : 0;
        }
    }
}

} // namespace detail

template <MPMode mpmode = MPMode::NONE, typename in_t>
inline constexpr void eye_to(in_t &&in) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::eye_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2>{}, std::integer_sequence<bool, true>{},
        std::forward<in_t>(in));
}

template <typename dtype, MPMode mpmode = MPMode::NONE,
          extents_c exts_t = extents<uint8_t>>
[[nodiscard]] inline constexpr auto eye(exts_t &&exts = exts_t{}) {
    auto out = empty<dtype>(std::forward<exts_t>(exts));
    eye_to<mpmode>(out);
    return out;
}

} // namespace mdtensor
