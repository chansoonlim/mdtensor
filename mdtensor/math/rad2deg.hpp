/**
 * @file
 * @brief Radian-to-degree conversion utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include <numbers>

#include "multiply.hpp"

namespace mdtensor {

/**
 * @brief Convert radians to degrees element-wise (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input angles in radians (mdspan, mdarray, scalar, etc.).
 * @param out Output angles in degrees (mdspan, mdarray, scalar, etc.).
 *
 * @note Equivalent to out = in * (180 / pi) in terms of array broadcasting.
 * @note The scaling constant is computed in a common type of the input value
 *       type and float to ensure floating-point conversion.
 *
 * @see mdtensor::rad2deg for the out-of-place version that returns the result.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t, typename out_t>
inline constexpr void rad2deg_to(in_t &&in, out_t &&out) {
    using value_t = std::common_type_t<typename decltype(core::to_mdspan(
                                           std::forward<in_t>(in)))::value_type,
                                       float>;

    constexpr value_t R2D = std::numbers::inv_pi_v<value_t> * value_t(180);

    multiply_to<mpmode>(std::forward<in_t>(in), R2D, std::forward<out_t>(out));
}

/**
 * @brief Convert radians to degrees element-wise (out-of-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 *         input.
 *
 * @param in Input angles in radians (mdspan, mdarray, scalar, etc.).
 *
 * @return mdarray or scalar.
 *
 * @note Equivalent to out = in * (180 / pi) in terms of array broadcasting.
 * @note The scaling constant is computed in a common type of the input value
 *       type and float to ensure floating-point conversion.
 *
 * @see mdtensor::rad2deg_to for the in-place version that writes into an
 *      output.
 */
template <MPMode mpmode = MPMode::NONE, typename dtype = void, typename in_t>
[[nodiscard]] inline constexpr auto rad2deg(in_t &&in) {
    using value_t = std::common_type_t<typename decltype(core::to_mdspan(
                                           std::forward<in_t>(in)))::value_type,
                                       float>;

    constexpr value_t R2D = std::numbers::inv_pi_v<value_t> * value_t(180);

    return multiply<mpmode, dtype>(std::forward<in_t>(in), R2D);
}

} // namespace mdtensor
