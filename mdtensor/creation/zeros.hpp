/**
 * @file
 * @brief Zeros tensor creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "full.hpp"

namespace mdtensor {

template <typename dtype, core::MPMode mpmode = core::MPMode::NONE,
          core::extents_c exts_t = core::extents<uint8_t>>
[[nodiscard]] inline constexpr auto zeros(exts_t &&exts = exts_t{}) {
    return full<dtype, mpmode, exts_t>(0, std::forward<exts_t>(exts));
}

template <typename dtype, core::MPMode mpmode = core::MPMode::NONE>
[[nodiscard]] inline constexpr auto zeros(const size_t &len) {
    return full<dtype, mpmode>(0, mdtensor::dims<1>{len});
}

} // namespace mdtensor
