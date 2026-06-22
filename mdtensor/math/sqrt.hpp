/**
 * @file
 * @brief Element-wise square root utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {
namespace detail {

#ifndef REAL_GCC

template <floating_point_c dtype>
[[nodiscard]] inline constexpr dtype
sqrt_newton_raphson(dtype &&x, dtype &&curr, dtype &&prev) {
    return (curr == prev)
               ? curr
               : sqrt_newton_raphson(x, (curr + x / curr) / (dtype)2, curr);
}

#endif

template <typename in_t, typename out_t>
inline constexpr void sqrt_impl(in_t &&in, out_t &&out) {
#ifdef REAL_GCC
    out() = std::sqrt(in());

#else
    using value_t = core::common_data_type_t<decltype(in()), float>;

    out() = (in() >= 0 && in() < std::numeric_limits<value_t>::infinity())
                ? sqrt_newton_raphson(static_cast<value_t>(in()),
                                      static_cast<value_t>(in()),
                                      static_cast<value_t>(0))
                : std::numeric_limits<value_t>::quiet_NaN();

#endif
}

} // namespace detail

/**
 * @brief Compute square root element-wise (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note Equivalent to out = std::sqrt(in) in terms of array broadcasting.
 *
 * @see mdtensor::sqrt for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t, typename out_t>
inline constexpr void sqrt_to(in_t &&in, out_t &&out) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::sqrt_impl(std::forward<decltype(elems)>(elems)...);
        },
        core::to_const_mdspan(std::forward<in_t>(in)),
        core::to_mdspan(std::forward<out_t>(out)));
}

/**
 * @brief Compute square root element-wise (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 *         input.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar.
 *
 * @note Equivalent to out = std::sqrt(in) in terms of array broadcasting.
 *
 * @see mdtensor::sqrt_to for the in-place version that writes into an output.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto sqrt(in_t &&in) {
    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::sqrt_impl(std::forward<decltype(elems)>(elems)...);
        },
        extents<uint8_t>{}, core::to_const_mdspan(std::forward<in_t>(in)));
}

} // namespace mdtensor
