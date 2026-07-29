/**
 * @file
 * @brief Element-wise greater-or-equal comparison utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {
namespace detail {

template <typename in1_t, typename in2_t, typename out_t>
inline constexpr void greater_equal_impl(in1_t &&in1, in2_t &&in2,
                                         out_t &&out) {
    out() = (in1() >= in2());
}

} // namespace detail

template <core::MPMode mpmode = core::MPMode::NONE, typename in1_t,
          typename in2_t, typename out_t>
inline constexpr void greater_equal_to(in1_t &&in1, in2_t &&in2, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::greater_equal_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, false, false, true>{},
        std::forward<in1_t>(in1), std::forward<in2_t>(in2),
        std::forward<out_t>(out));
}

template <typename dtype = bool, core::MPMode mpmode = core::MPMode::NONE,
          typename in1_t, typename in2_t>
[[nodiscard]] inline constexpr auto greater_equal(in1_t &&in1, in2_t &&in2) {
    auto out = core::create_out<dtype>(core::extents<uint8_t>{},
                                       std::forward<in1_t>(in1),
                                       std::forward<in2_t>(in2));

    greater_equal_to<mpmode>(std::forward<in1_t>(in1), std::forward<in2_t>(in2),
                             out);

    return out;
}

} // namespace mdtensor
