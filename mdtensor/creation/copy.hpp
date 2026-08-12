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

template <typename dtype = void>
constexpr void copy_ufunc(auto &&in, auto &&out) {
    if constexpr (std::is_void_v<dtype>) {
        out = in;

    } else {
        out = static_cast<dtype>(in);
    }
}

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto copy(auto &&in,
                                  out_t &&out = out_t{std::nullopt}) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    auto out_md = core::resolve_output_like<dtype>(
        std::forward<decltype(out)>(out), in_mds);

    core::batch_with_broadcast<backend>(
        [](auto &&...elems) {
            ufunc::copy_ufunc<dtype>(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, true, false>{}, in_mds, out_md);

    return out_md;
}

} // namespace mdtensor
