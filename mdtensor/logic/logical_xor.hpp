/**
 * @file
 * @brief Element-wise logical XOR utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {
namespace ufunc {

constexpr void logical_xor_ufunc(auto &&in1, auto &&in2, auto &&out,
                                 auto &&where) {
    if constexpr (requires { static_cast<bool>(where); }) {
        if (!static_cast<bool>(where)) {
            return;
        }
    }

    out = (static_cast<bool>(in1) != static_cast<bool>(in2));
}

} // namespace ufunc

template <core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto
logical_xor(auto &&in1, auto &&in2, out_t &&out = out_t{std::nullopt},
            where_t &&where = where_t{std::nullopt}) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    auto out_md = core::resolve_broadcasted_output<bool>(
        std::forward<decltype(out)>(out), core::extents<std::uint8_t>{},
        in1_mds, in2_mds);

    core::batch<backend>(
        [](auto &&...elems) {
            ufunc::logical_xor_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, true, false, true>{}, in1_mds,
        in2_mds, out_md, std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
