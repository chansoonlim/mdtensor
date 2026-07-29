/**
 * @file
 * @brief Broadcast extents utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/broadcast.hpp"

namespace mdtensor {

template <typename index_t, typename... exts_t>
[[nodiscard]] inline constexpr index_t
broadcast_extents(exts_t... exts) noexcept {
    return core::broadcast_extents<index_t>(std::forward<exts_t>(exts)...);
}

} // namespace mdtensor
