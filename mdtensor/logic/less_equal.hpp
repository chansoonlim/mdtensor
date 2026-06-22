/**
 * @file
 * @brief Element-wise less-or-equal comparison utilities for mdtensor.
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
inline constexpr void less_equal_impl(in1_t &&in1, in2_t &&in2, out_t &&out) {
    out() = (in1() <= in2());
}

} // namespace detail

/**
 * @brief Compare two inputs element-wise for less-or-equal (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = (in1 <= in2) in terms of array broadcasting.
 *
 * @see mdtensor::less_equal for the out-of-place version that returns the
 * result.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t,
          typename out_t>
inline constexpr void less_equal_to(in1_t &&in1, in2_t &&in2, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::less_equal_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Compare two inputs element-wise for less-or-equal (out-of-place).
 *
 * @tparam dtype (optional) Data type of the result. Default is int8_t.
 *         If an integral type is used, results represent boolean values (0/1).
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar containing element-wise comparison results.
 *
 * @note Equivalent to out = (in1 <= in2) in terms of array broadcasting.
 *
 * @see mdtensor::less_equal_to for the in-place version that writes into an
 * output.
 */
template <typename dtype = int8_t, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto less_equal(in1_t &&in1, in2_t &&in2) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::less_equal_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)));
}

} // namespace mdtensor
