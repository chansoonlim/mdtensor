/**
 * @file
 * @brief Element-wise addition utilities for mdtensor.
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
inline constexpr void add_impl(in1_t &&in1, in2_t &&in2, out_t &&out) {
    out() = in1() + in2();
}

} // namespace detail

/**
 * @brief Add arguments element-wise (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = in1 + in2 in terms of array broadcasting.
 *
 * @see mdtensor::add for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t,
          typename out_t>
inline constexpr void add_to(in1_t &&in1, in2_t &&in2, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::add_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Add arguments element-wise (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) data type of the result. If void, deduced from
 *         inputs.
 * @param in1 mdspan, mdarray, scalar, etc.
 * @param in2 mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar.
 *
 * @note Equivalent to out = in1 + in2 in terms of array broadcasting.
 *
 * @see mdtensor::add_to for the in-place version that modifies the output.
 */
template <MPMode mpmode = MPMode::NONE, typename dtype = void, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto add(in1_t &&in1, in2_t &&in2) {
    return core::batch_out<mpmode, dtype>(
        [](auto &&...elems) {
            detail::add_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)));
}

} // namespace mdtensor
