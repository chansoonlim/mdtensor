/**
 * @file
 * @brief Empty tensor creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {

template <typename dtype, extents_c exts_t = extents<uint8_t>>
[[nodiscard]] inline constexpr auto empty(exts_t &&exts = exts_t{}) noexcept {
    return core::make_container<dtype>(std::forward<exts_t>(exts));
}

} // namespace mdtensor
