/**
 * @file
 * @brief Copy utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "empty_like.hpp"

namespace mdtensor {
namespace ufunc {

constexpr void copy_ufunc(auto &&in, auto &&out) { out() = in(); }

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto copy(auto &&in,
                                  out_t &&out = out_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = [&]() {
        if constexpr (core::is_nullopt_t_c<decltype(out)>) {
            return empty_like<dtype>(in_mds);

        } else {
            return core::to_output_mdspan(std::forward<decltype(out)>(out));
        }
    }();

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::copy_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, false>{},
        std::forward<decltype(in)>(in), out_md);

    return out_md;
}

} // namespace mdtensor
