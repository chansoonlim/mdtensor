/**
 * @file
 * @brief Element-wise cosine utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {
namespace ufunc {

template <typename dtype = void>
constexpr void cos_ufunc(auto &&in, auto &&out, auto &&where) {
    if constexpr (requires {
                      { where == false } -> std::convertible_to<bool>;
                  }) {
        if (where == false) {
            return;
        }
    }

    using calc_t =
        core::floating_calc_type_t<dtype, decltype(in), decltype(out)>;

    out = std::cos(static_cast<calc_t>(in));
}

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t, typename where_t = std::nullopt_t>
[[nodiscard]] constexpr auto cos(auto &&in, out_t &&out = out_t{std::nullopt},
                                 where_t &&where = where_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    using calc_t = core::floating_calc_type_t<dtype, decltype(in_mds)>;

    auto out_md = core::resolve_output_like<calc_t, true>(
        std::forward<decltype(out)>(out), in_mds);

    core::batch<backend>(
        [](auto &&...elems) {
            ufunc::cos_ufunc<calc_t>(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, false, true>{}, in_mds, out_md,
        std::forward<decltype(where)>(where));

    return out_md;
}

} // namespace mdtensor
