/**
 * @file
 * @brief Element-wise approximate equality utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../math/absolute.hpp"

namespace mdtensor {
namespace detail {

template <typename in1_t, typename in2_t, typename out_t>
inline constexpr void isclose_impl(in1_t &&in1, in2_t &&in2, out_t &&out,
                                   const double &rtol = 1e-05,
                                   const double &atol = 1e-08) {
    out() = absolute(in1() - in2()) <=
            (atol + rtol * static_cast<double>(absolute(in2())));
}

} // namespace detail

/**
 * @brief Compare two inputs element-wise for approximate equality (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 * @param rtol Relative tolerance. Default is 1e-05.
 * @param atol Absolute tolerance. Default is 1e-08.
 *
 * @note Equivalent to out = (|in1 - in2| <= (atol + rtol * |in2|)) in terms of
 * array broadcasting.
 *
 * @see mdtensor::isclose for the out-of-place version that returns the result.
 * @see mdtensor::allclose for full-tensor reduction using isclose.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t,
          typename out_t>
inline constexpr void isclose_to(in1_t &&in1, in2_t &&in2, out_t &&out,
                                 const double &rtol = 1e-05,
                                 const double &atol = 1e-08) {
    core::batch<mpmode>(
        [&](auto &&...elems) {
            detail::isclose_impl(std::forward<decltype(elems)>(elems)..., rtol,
                                 atol);
        },
        core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Compare two inputs element-wise for approximate equality
 * (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. Default is int8_t.
 *         If an integral type is used, results represent boolean values (0/1).
 *
 * @param in1 First input mdspan, mdarray, scalar, etc.
 * @param in2 Second input mdspan, mdarray, scalar, etc.
 * @param rtol Relative tolerance. Default is 1e-05.
 * @param atol Absolute tolerance. Default is 1e-08.
 *
 * @return mdarray or scalar containing element-wise comparison results.
 *
 * @note Equivalent to out = (|in1 - in2| <= (atol + rtol * |in2|)) in terms of
 * array broadcasting.
 *
 * @see mdtensor::isclose_to for the in-place version that writes into an
 * output.
 * @see mdtensor::allclose for full-tensor reduction using isclose.
 */
template <typename dtype = int8_t, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto isclose(in1_t &&in1, in2_t &&in2,
                                            const double &rtol = 1e-05,
                                            const double &atol = 1e-08) {
    return core::batch_out<dtype, mpmode>(
        [&](auto &&...elems) {
            detail::isclose_impl(std::forward<decltype(elems)>(elems)..., rtol,
                                 atol);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in1_t>(in1)),
        core::to_const_mdspan(std::forward<in2_t>(in2)));
}

} // namespace mdtensor
