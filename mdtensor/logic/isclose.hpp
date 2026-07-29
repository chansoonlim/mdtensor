/**
 * @file
 * @brief Element-wise approximate equality utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../math/absolute.hpp"

namespace mdtensor {
namespace detail {

template <typename in1_t, typename in2_t, typename out_t>
inline constexpr void isclose_impl(in1_t &&in1, in2_t &&in2, out_t &&out,
                                   const double &rtol = 1e-05,
                                   const double &atol = 1e-08) {
    out() = absolute(in1() - in2()) <=
            (atol + rtol * static_cast<double>(absolute(in2())));
}

} // namespace detail

template <core::MPMode mpmode = core::MPMode::NONE, typename in1_t,
          typename in2_t, typename out_t>
inline constexpr void isclose_to(in1_t &&in1, in2_t &&in2, out_t &&out,
                                 const double &rtol = 1e-05,
                                 const double &atol = 1e-08) {
    core::batch<mpmode>(
        [&](auto &&...elems) {
            detail::isclose_impl(std::forward<decltype(elems)>(elems)..., rtol,
                                 atol);
        },
        std::integer_sequence<bool, false, false, true>{},
        std::forward<in1_t>(in1), std::forward<in2_t>(in2),
        std::forward<out_t>(out));
}

template <typename dtype = bool, core::MPMode mpmode = core::MPMode::NONE,
          typename in1_t, typename in2_t>
[[nodiscard]] inline constexpr auto isclose(in1_t &&in1, in2_t &&in2,
                                            const double &rtol = 1e-05,
                                            const double &atol = 1e-08) {
    auto out = core::create_out<dtype>(core::extents<uint8_t>{},
                                       std::forward<in1_t>(in1),
                                       std::forward<in2_t>(in2));

    isclose_to<mpmode>(std::forward<in1_t>(in1), std::forward<in2_t>(in2), out,
                       rtol, atol);

    return out;
}

} // namespace mdtensor
