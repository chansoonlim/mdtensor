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
namespace detail {

template <typename in_t, typename val_t>
inline constexpr void fill_impl(in_t &&in, val_t &&val) {
    in() = val;
}

} // namespace detail

/**
 * @brief Fill all elements of a tensor with a scalar value (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Output tensor to fill (mdspan, mdarray, scalar, etc.).
 * @param val Value to assign to every element.
 *
 * @note Equivalent to `in[...] = val` in terms of array broadcasting.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t, typename val_t>
inline constexpr void fill(in_t &&in, val_t &&val) {
    core::batch<mpmode>(
        [&](auto &&...elems) {
            detail::fill_impl(std::forward<decltype(elems)>(elems)..., val);
        },
        core::to_mdspan(std::forward<in_t>(in)));
}

} // namespace mdtensor
