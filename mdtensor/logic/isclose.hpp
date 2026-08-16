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
#include "isinf.hpp"
#include "isnan.hpp"

namespace mdtensor {
namespace ufunc {

constexpr void isclose_ufunc(auto &&in1, auto &&in2, auto &&out, auto &&rtol,
                             auto &&atol, const bool equal_nan) {
    using common_t = core::promote_type_t<decltype(in1), decltype(in2),
                                          decltype(rtol), decltype(atol)>;

    const bool is_in1_nan = isnan(in1);
    const bool is_in2_nan = isnan(in2);

    if (is_in1_nan || is_in2_nan) {
        // true if equal_nan is true and both inputs are nan
        out = equal_nan && is_in1_nan && is_in2_nan;

    } else if (in1 == in2) {
        // true when exact finite equality, signed zeroes, or same-sign
        // infinities
        out = true;

    } else if (isinf(in1) || isinf(in2)) {
        // false when one input is infinite and the other is finite or
        // opposite-sign infinity
        out = false;

    } else {
        out =
            absolute(static_cast<common_t>(in1) - static_cast<common_t>(in2)) <=
            (static_cast<common_t>(atol) +
             static_cast<common_t>(rtol) *
                 absolute(static_cast<common_t>(in2)));
    }
}

} // namespace ufunc

template <core::Backend backend = core::Backend::AUTO, typename rtol_t = double,
          typename atol_t = double, typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto
isclose(auto &&in1, auto &&in2, rtol_t &&rtol = rtol_t{1e-05},
        atol_t &&atol = atol_t{1e-08}, out_t &&out = out_t{std::nullopt},
        const bool equal_nan = false) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));
    const auto rtol_mds =
        core::to_const_mdspan(std::forward<decltype(rtol)>(rtol));
    const auto atol_mds =
        core::to_const_mdspan(std::forward<decltype(atol)>(atol));

    auto out_md = core::resolve_broadcasted_output<bool>(
        std::forward<decltype(out)>(out), core::extents<std::uint8_t>{},
        in1_mds, in2_mds, rtol_mds, atol_mds);

    core::batch<backend>(
        [&](auto &&...elems) {
            ufunc::isclose_ufunc(std::forward<decltype(elems)>(elems)...,
                                 equal_nan);
        },
        std::integer_sequence<bool, true, true, false, true, true>{}, in1_mds,
        in2_mds, out_md, rtol_mds, atol_mds);

    return out_md;
}

} // namespace mdtensor
