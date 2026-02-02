/**
 * @file
 * @brief Element-wise minimum utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {
namespace detail {

template <typename in1_t, typename in2_t, typename out_t>
inline constexpr void minimum_impl(in1_t &&in1, in2_t &&in2, out_t &&out) {
    using value_t = std::remove_cvref_t<decltype(out())>;

    out() = std::min(static_cast<value_t>(in1()), static_cast<value_t>(in2()));
}

} // namespace detail

/**
 * @brief Compute element-wise minimum of two inputs (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = std::min(in1, in2) in terms of array broadcasting.
 *
 * @see mdtensor::minimum for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t,
          typename out_t>
inline constexpr void minimum_to(in1_t &&in1, in2_t &&in2, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::minimum_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Compute element-wise minimum of two inputs (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 *         inputs.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar.
 *
 * @note Equivalent to out = std::min(in1, in2) in terms of array broadcasting.
 *
 * @see mdtensor::minimum_to for the in-place version that writes into an
 *      output.
 */
template <MPMode mpmode = MPMode::NONE, typename dtype = void, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto minimum(in1_t &&in1, in2_t &&in2) {
    return core::batch_out<mpmode, dtype>(
        [](auto &&...elems) {
            detail::minimum_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)));
}

} // namespace mdtensor
