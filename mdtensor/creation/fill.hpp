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

template <core::MPMode mpmode = core::MPMode::NONE, typename in_t,
          typename val_t>
inline constexpr void fill(in_t &&in, val_t &&val) {
    core::batch<mpmode>(
        [&](auto &&...elems) {
            detail::fill_impl(std::forward<decltype(elems)>(elems)..., val);
        },
        std::integer_sequence<bool, true>{}, std::forward<in_t>(in));
}

} // namespace mdtensor
