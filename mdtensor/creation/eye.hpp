/**
 * @file
 * @brief Identity matrix (eye) creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "empty.hpp"

namespace mdtensor {
namespace ufunc {

constexpr void eye_ufunc(auto &&out, const int &k) {
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    using index_t = typename decltype(out_mds)::index_type;

    for (index_t i = 0; i < out_mds.extent(0); i++) {
        for (index_t j = 0; j < out_mds.extent(1); j++) {
            out_mds(i, j) = (i + k == j) ? 1 : 0;
        }
    }
}

} // namespace ufunc

template <core::arithmetic value_t = std::int8_t,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto eye(auto &&shape, const int &k = 0,
                                 out_t &&out = out_t{std::nullopt}) {
    using shape_t = std::remove_cvref_t<decltype(shape)>;

    if constexpr (core::extents_c<shape_t>) {
        const auto exts =
            core::to_extents(std::forward<decltype(shape)>(shape));

        static_assert(exts.rank() >= 2, "eye requires rank >= 2");

        auto out_md =
            core::resolve_output<value_t>(std::forward<decltype(out)>(out),
                                          std::forward<decltype(exts)>(exts));

        core::batch<backend>(
            [&](auto &&...elems) {
                ufunc::eye_ufunc(std::forward<decltype(elems)>(elems)..., k);
            },
            std::index_sequence<2>{}, std::integer_sequence<bool, false>{},
            out_md);

        return out_md;

    } else if constexpr (core::integral_c<shape_t>) {
        if (shape < shape_t{0}) {
            throw std::invalid_argument("exts must be non-negative");
        }

        using index_t = std::make_unsigned_t<shape_t>;

        return eye<value_t, backend>(
            core::dextents<index_t, 2>{static_cast<index_t>(shape),
                                       static_cast<index_t>(shape)},
            k);
    }
}

template <std::size_t N, core::arithmetic value_t = std::int8_t,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto eye(const int &k = 0,
                                 out_t &&out = out_t{std::nullopt}) {
    return eye<value_t, backend>(core::extents<std::size_t, N, N>{}, k,
                                 std::forward<out_t>(out));
}

} // namespace mdtensor
