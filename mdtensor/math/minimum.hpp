/**
 * @file
 * @brief Element-wise minimum utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {
namespace ufunc {

constexpr void minimum_ufunc(auto &&in1, auto &&in2, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where == false } -> std::convertible_to<bool>;
                  }) {
        if (where == false) {
            return;
        }
    }

    // if one of the inputs is NaN, return NaN (numpy-like)
    if constexpr (requires {
                      { std::isnan(in1) } -> std::convertible_to<bool>;
                  }) {
        if (std::isnan(in1)) {
            out = in1;
            return;
        }
    }

    if constexpr (requires {
                      { std::isnan(in2) } -> std::convertible_to<bool>;
                  }) {
        if (std::isnan(in2)) {
            out = in2;
            return;
        }
    }

    using common_t = core::promote_type_t<decltype(in1), decltype(in2)>;

    out = std::min(static_cast<common_t>(in1), static_cast<common_t>(in2));
}

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto minimum(auto &&in1, auto &&in2,
                                     out_t &&out = out_t{std::nullopt},
                                     where_t &&where = where_t{std::nullopt}) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    auto out_md = core::resolve_broadcasted_output<dtype>(
        std::forward<decltype(out)>(out), core::extents<std::uint8_t>{},
        in1_mds, in2_mds);

    core::batch<backend>(
        [](auto &&...elems) {
            ufunc::minimum_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, true, false, true>{}, in1_mds,
        in2_mds, out_md, std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
