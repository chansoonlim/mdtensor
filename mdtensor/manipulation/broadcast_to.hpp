/**
 * @file
 * @brief Broadcast_to utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {

template <typename in_t, extents_c new_extents_t>
[[nodiscard]] inline constexpr auto
broadcast_to(in_t &&in, new_extents_t &&new_extents) noexcept {
    return core::broadcast_to(std::forward<in_t>(in),
                              std::forward<new_extents_t>(new_extents));
}

} // namespace mdtensor
