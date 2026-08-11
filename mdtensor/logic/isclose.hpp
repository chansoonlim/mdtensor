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
namespace ufunc {

constexpr void isclose_ufunc(auto &&in1, auto &&in2, auto &&out, auto &&rtol,
                             auto &&atol, const bool equal_nan) {
    if constexpr (requires {
                      { std::isnan(in1()) } -> std::convertible_to<bool>;
                      { std::isnan(in2()) } -> std::convertible_to<bool>;
                  }) {
        if (equal_nan) {
            if (std::isnan(in1()) && std::isnan(in2())) {
                out() = true;
                return;
            }
        }
    }

    // if both inputs are inf and same sign, return true (numpy-like)
    if constexpr (requires {
                      { std::isinf(in1()) } -> std::convertible_to<bool>;
                      { std::isinf(in2()) } -> std::convertible_to<bool>;
                      { std::signbit(in1()) } -> std::convertible_to<bool>;
                      { std::signbit(in2()) } -> std::convertible_to<bool>;
                  }) {
        if (std::isinf(in1()) && std::isinf(in2()) &&
            std::signbit(in1()) == std::signbit(in2())) {
            out() = true;
            return;
        }
    }

    using out_t = std::remove_cvref_t<decltype(out())>;
    using calc_t = core::common_data_type_t<decltype(in1()), decltype(in2()),
                                            decltype(rtol()), decltype(atol())>;

    out() = static_cast<out_t>(
        absolute(static_cast<calc_t>(in1()) - static_cast<calc_t>(in2())) <=
        (static_cast<calc_t>(atol()) +
         static_cast<calc_t>(rtol()) * absolute(static_cast<calc_t>(in2()))));
}

} // namespace ufunc

template <typename dtype = bool, core::Backend backend = core::Backend::AUTO,
          typename rtol_t = double, typename atol_t = double,
          typename out_t = std::nullopt_t>
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

    auto out_md = [&]() {
        if constexpr (core::nullopt_t_c<decltype(out)>) {
            return core::make_output<dtype>(core::extents<std::uint8_t>{},
                                            in1_mds, in2_mds, rtol_mds,
                                            atol_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [&](auto &&...elems) {
            ufunc::isclose_ufunc(std::forward<decltype(elems)>(elems)...,
                                 equal_nan);
        },
        std::integer_sequence<bool, true, true, false, true, true>{}, in1_mds,
        in2_mds, out_md, rtol_mds, atol_mds);

    return out_md;
}

} // namespace mdtensor
