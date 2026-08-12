/**
 * @file
 * @brief Full creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "empty.hpp"

namespace mdtensor {
namespace ufunc {

constexpr void full_ufunc(auto &&out, auto &&val) { out = val; }

} // namespace ufunc

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto full(auto &&shape, auto &&val,
                                  out_t &&out = out_t{std::nullopt}) {
    const auto val_mds =
        core::to_const_mdspan(std::forward<decltype(val)>(val));

    auto out_md =
        core::resolve_output<core::output_value_t<dtype, decltype(val_mds)>>(
            std::forward<decltype(out)>(out),
            core::to_extents(std::forward<decltype(shape)>(shape)));

    core::batch<backend>(
        [](auto &&...elems) {
            ufunc::full_ufunc(std::forward<decltype(elems)>(elems)...);
        },
        std::integer_sequence<bool, false, true>{}, out_md, val_mds);

    return out_md;
}

} // namespace mdtensor
