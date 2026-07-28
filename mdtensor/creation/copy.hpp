/**
 * @file
 * @brief Copy utilities for mdtensor.
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
inline constexpr void copy_impl(in_t &&in, out_t &&out) {
    out() = in();
}

} // namespace detail

template <MPMode mpmode = MPMode::NONE, typename in_t, typename out_t>
inline constexpr void copy_to(in_t &&in, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::copy_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, false, true>{}, std::forward<in_t>(in),
        std::forward<out_t>(out));
}

template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto copy(in_t &&in) {
    auto out =
        core::create_out<dtype>(extents<uint8_t>{}, std::forward<in_t>(in));

    copy_to<mpmode>(std::forward<in_t>(in), out);

    return out;
}

} // namespace mdtensor
