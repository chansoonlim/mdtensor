/**
 * @file
 * @brief Zeros-like tensor creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "full_like.hpp"

namespace mdtensor {

template <typename dtype = void, core::MPMode mpmode = core::MPMode::NONE,
          typename in_t>
[[nodiscard]] inline constexpr auto zeros_like(in_t &&in) {
    return full_like<dtype, mpmode>(std::forward<in_t>(in), 0);
}

} // namespace mdtensor
