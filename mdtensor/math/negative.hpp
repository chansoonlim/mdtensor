/**
 * @file
 * @brief Element-wise negation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {
namespace detail {

template <typename in_t, typename out_t>
inline constexpr void negative_impl(in_t &&in, out_t &&out) {
    out() = -in();
}

} // namespace detail

template <core::MPMode mpmode = core::MPMode::NONE, typename in_t,
          typename out_t>
inline constexpr void negative_to(in_t &&in, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::negative_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, false, true>{}, std::forward<in_t>(in),
        std::forward<out_t>(out));
}

template <typename dtype = void, core::MPMode mpmode = core::MPMode::NONE,
          typename in_t>
[[nodiscard]] inline constexpr auto negative(in_t &&in) {
    auto out = core::create_out<dtype>(core::extents<uint8_t>{},
                                       std::forward<in_t>(in));

    negative_to<mpmode>(std::forward<in_t>(in), out);

    return out;
}

} // namespace mdtensor
