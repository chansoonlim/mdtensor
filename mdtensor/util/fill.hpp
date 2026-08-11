/**
 * @file
 * @brief Fill utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {
namespace ufunc {

constexpr void fill_ufunc(auto &&out, auto &&val) { out = val; }

} // namespace ufunc

template <core::Backend backend = core::Backend::AUTO>
constexpr void fill(auto &&out, auto &&val) {
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    core::batch<backend, out_mds.rank()>(
        [&](auto &&...elems) {
            ufunc::fill_ufunc(std::forward<decltype(elems)>(elems)...,
                              std::forward<decltype(val)>(val));
        },
        out_mds);
}

} // namespace mdtensor
