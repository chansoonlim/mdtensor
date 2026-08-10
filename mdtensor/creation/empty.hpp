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

template <typename dtype = double>
[[nodiscard]] constexpr auto empty(auto &&shape) {
    return core::make_container<dtype>(
        core::to_extents(std::forward<decltype(shape)>(shape)));
}

} // namespace mdtensor
