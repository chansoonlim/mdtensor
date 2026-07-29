/**
 * @file
 * @brief Element-wise atan2 utilities for mdtensor.
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
inline constexpr void atan2_impl(in1_t &&in1, in2_t &&in2, out_t &&out) {
    using value_t = core::common_data_type_t<decltype(in1()), decltype(in2()),
                                             decltype(out())>;

    out() =
        std::atan2(static_cast<value_t>(in1()), static_cast<value_t>(in2()));
}

} // namespace detail

template <core::MPMode mpmode = core::MPMode::NONE, typename in1_t,
          typename in2_t, typename out_t>
inline constexpr void atan2_to(in1_t &&in1, in2_t &&in2, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::atan2_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, false, false, true>{},
        std::forward<in1_t>(in1), std::forward<in2_t>(in2),
        std::forward<out_t>(out));
}

template <typename dtype = void, core::MPMode mpmode = core::MPMode::NONE,
          typename in1_t, typename in2_t>
[[nodiscard]] inline constexpr auto atan2(in1_t &&in1, in2_t &&in2) {
    auto out = core::create_out<dtype>(core::extents<uint8_t>{},
                                       std::forward<in1_t>(in1),
                                       std::forward<in2_t>(in2));

    atan2_to<mpmode>(std::forward<in1_t>(in1), std::forward<in2_t>(in2), out);

    return out;
}

} // namespace mdtensor
