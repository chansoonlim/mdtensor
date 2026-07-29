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

[[nodiscard]] constexpr auto broadcast_to(auto &&in, auto &&shape) {
    return core::broadcast_to(std::forward<decltype(in)>(in),
                              std::forward<decltype(shape)>(shape));
}

} // namespace mdtensor
