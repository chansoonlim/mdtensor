/**
 * @file
 * @brief Reshape utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {

template <core::CopyMode copy = core::CopyMode::AUTO>
[[nodiscard]] constexpr auto reshape(auto &&in, auto &&shape) {
    return core::reshape<copy>(std::forward<decltype(in)>(in),
                               std::forward<decltype(shape)>(shape));
}

} // namespace mdtensor
