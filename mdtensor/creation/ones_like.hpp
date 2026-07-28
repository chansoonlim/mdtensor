/**
 * @file
 * @brief Ones-like tensor creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "full_like.hpp"

namespace mdtensor {

template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto ones_like(in_t &&in) {
    return full_like<dtype, mpmode>(std::forward<in_t>(in), 1);
}

} // namespace mdtensor
