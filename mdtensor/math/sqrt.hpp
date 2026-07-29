/**
 * @file
 * @brief Element-wise square root utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {
namespace detail {

#ifndef REAL_GCC

template <std::floating_point dtype>
[[nodiscard]] inline constexpr dtype
sqrt_newton_raphson(dtype &&x, dtype &&curr, dtype &&prev) {
    return (curr == prev)
               ? curr
               : sqrt_newton_raphson(x, (curr + x / curr) / (dtype)2, curr);
}

#endif

template <typename in_t, typename out_t>
inline constexpr void sqrt_impl(in_t &&in, out_t &&out) {
#ifdef REAL_GCC
    out() = std::sqrt(in());

#else
    using value_t = core::common_data_type_t<decltype(in()), float>;

    out() = (in() >= 0 && in() < std::numeric_limits<value_t>::infinity())
                ? sqrt_newton_raphson(static_cast<value_t>(in()),
                                      static_cast<value_t>(in()),
                                      static_cast<value_t>(0))
                : std::numeric_limits<value_t>::quiet_NaN();

#endif
}

} // namespace detail

template <core::MPMode mpmode = core::MPMode::NONE, typename in_t,
          typename out_t>
inline constexpr void sqrt_to(in_t &&in, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::sqrt_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, false, true>{}, std::forward<in_t>(in),
        std::forward<out_t>(out));
}

template <typename dtype = void, core::MPMode mpmode = core::MPMode::NONE,
          typename in_t>
[[nodiscard]] inline constexpr auto sqrt(in_t &&in) {
    auto out = core::create_out<dtype>(core::extents<uint8_t>{},
                                       std::forward<in_t>(in));

    sqrt_to<mpmode>(std::forward<in_t>(in), out);

    return out;
}

} // namespace mdtensor
