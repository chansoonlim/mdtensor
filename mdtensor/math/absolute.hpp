/**
 * @file
 * @brief Element-wise absolute value utilities for mdtensor.
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
inline constexpr void absolute_impl(in_t &&in, out_t &&out) {
#ifdef REAL_GCC
    out() = std::abs(in());

#else
    // NOTE: std::abs is not constexpr in clang 16.
    if constexpr (std::is_signed_v<std::remove_cvref_t<decltype(in())>>) {
        out() = in() < 0 ? -in() : in();

    } else {
        out() = in();
    }

#endif
}

} // namespace detail

template <core::MPMode mpmode = core::MPMode::NONE, typename in_t,
          typename out_t>
inline constexpr void absolute_to(in_t &&in, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::absolute_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, false, true>{}, std::forward<in_t>(in),
        std::forward<out_t>(out));
}

template <typename dtype = void, core::MPMode mpmode = core::MPMode::NONE,
          typename in_t>
[[nodiscard]] inline constexpr auto absolute(in_t &&in) {
    auto out = core::create_out<dtype>(core::stdex::extents<uint8_t>{},
                                       std::forward<in_t>(in));

    absolute_to<mpmode>(std::forward<in_t>(in), out);

    return out;
}

} // namespace mdtensor
