/**
 * @file
 * @brief Array equivalence utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "all.hpp"
#include "equal.hpp"

namespace mdtensor {

template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t>
[[nodiscard]] inline constexpr bool array_equiv(in1_t &&in1, in2_t &&in2) {
    return all(equal<int8_t, mpmode>(std::forward<in1_t>(in1),
                                     std::forward<in2_t>(in2)));
}

} // namespace mdtensor
