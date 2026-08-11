/**
 * @file
 * @brief Element-wise less-than comparison utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {
namespace ufunc {

constexpr void less_ufunc(auto &&in1, auto &&in2, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where == false } -> std::convertible_to<bool>;
                  }) {
        if (where == false) {
            return;
        }
    }

    out = (in1 < in2);
}

} // namespace ufunc

template <typename dtype = bool, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto less(auto &&in1, auto &&in2,
                                  out_t &&out = out_t{std::nullopt},
                                  where_t &&where = where_t{std::nullopt}) {
    const auto in1_mds =
        core::to_const_mdspan(std::forward<decltype(in1)>(in1));
    const auto in2_mds =
        core::to_const_mdspan(std::forward<decltype(in2)>(in2));

    auto out_md = core::resolve_broadcasted_output<dtype>(
        std::forward<decltype(out)>(out), core::extents<std::uint8_t>{},
        in1_mds, in2_mds);

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::less_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, true, false, true>{}, in1_mds,
        in2_mds, out_md, std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
