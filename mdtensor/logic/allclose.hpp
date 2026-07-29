/**
 * @file
 * @brief All-close comparison utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "all.hpp"
#include "isclose.hpp"

namespace mdtensor {

template <core::MPMode mpmode = core::MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr bool allclose(in1_t &&in1, in2_t &&in2,
                                             const double &rtol = 1e-05,
                                             const double &atol = 1e-08) {
    return all(isclose<int8_t, mpmode>(std::forward<in1_t>(in1),
                                       std::forward<in2_t>(in2), rtol, atol));
}

} // namespace mdtensor
