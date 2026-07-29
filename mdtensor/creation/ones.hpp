/**
 * @file
 * @brief Ones tensor creation utilities for mdtensor.
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
[[nodiscard]] inline constexpr auto ones(exts_t &&exts = exts_t{}) {
    return full<dtype, mpmode, exts_t>(1, std::forward<exts_t>(exts));
}

template <typename dtype, core::MPMode mpmode = core::MPMode::NONE>
[[nodiscard]] inline constexpr auto ones(const size_t &len) {
    return full<dtype, mpmode>(1, dims<1>{len});
}

} // namespace mdtensor
