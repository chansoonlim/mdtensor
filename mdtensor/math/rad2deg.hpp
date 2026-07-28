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

template <MPMode mpmode = MPMode::NONE, typename in_t, typename out_t>
inline constexpr void rad2deg_to(in_t &&in, out_t &&out) {
    using value_t =
        core::common_data_type_t<typename decltype(core::to_mdspan(
                                     std::forward<in_t>(in)))::value_type,
                                 float>;

    constexpr value_t R2D = std::numbers::inv_pi_v<value_t> * value_t(180);

    multiply_to<mpmode>(std::forward<in_t>(in), R2D, std::forward<out_t>(out));
}

template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto rad2deg(in_t &&in) {
    using value_t =
        core::common_data_type_t<typename decltype(core::to_mdspan(
                                     std::forward<in_t>(in)))::value_type,
                                 float>;

    constexpr value_t R2D = std::numbers::inv_pi_v<value_t> * value_t(180);

    return multiply<dtype, mpmode>(std::forward<in_t>(in), R2D);
}

} // namespace mdtensor
