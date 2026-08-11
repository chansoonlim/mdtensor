/**
 * @file
 * @brief Element-wise isnan check for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../creation/empty_like.hpp"

namespace mdtensor {
namespace ufunc {

constexpr void isnan_ufunc(auto &&in, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where() == false } -> std::convertible_to<bool>;
                  }) {
        if (where() == false) {
            return;
        }
    }

    if constexpr (requires {
                      { std::isnan(in()) } -> std::convertible_to<bool>;
                  }) {
        if (std::isnan(in())) {
            out() = true;
            return;
        }
    }

    out() = false;
}

} // namespace ufunc

template <typename dtype = bool, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto isnan(auto &&in, out_t &&out = out_t{std::nullopt},
                                   where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = core::resolve_output_like<dtype>(
        std::forward<decltype(out)>(out), in_mds);

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::isnan_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, false, true>{}, in_mds, out_md,
        std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
