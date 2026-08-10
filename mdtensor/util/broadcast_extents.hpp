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

template <core::extents_c... ins_t>
[[nodiscard]] constexpr auto broadcast_extents(ins_t &&...ins) {
    return core::broadcast_extents(std::forward<ins_t>(ins)...);
}

} // namespace mdtensor
