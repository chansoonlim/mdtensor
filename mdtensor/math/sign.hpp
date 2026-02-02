/**
 * @file
 * @brief Element-wise sign utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {
namespace detail {

template <typename in_t, typename out_t>
inline constexpr void sign_impl(in_t &&in, out_t &&out) {
    out() = (in() > 0) - (in() < 0);
}

} // namespace detail

/**
 * @brief Compute sign element-wise (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @see mdtensor::sign for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t, typename out_t>
inline constexpr void sign_to(in_t &&in, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::sign_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in_t>(in)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Compute sign element-wise (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. Default is int8_t.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar.
 *
 * @note By default, the result type is int8_t to represent {-1, 0, 1}.
 *
 * @see mdtensor::sign_to for the in-place version that writes into an output.
 */
template <MPMode mpmode = MPMode::NONE, typename dtype = int8_t, typename in_t>
[[nodiscard]] inline constexpr auto sign(in_t &&in) {
    return core::batch_out<mpmode, dtype>(
        [](auto &&...elems) {
            detail::sign_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in_t>(in)));
}

} // namespace mdtensor
