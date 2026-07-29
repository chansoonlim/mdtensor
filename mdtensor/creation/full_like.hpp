/**
 * @file
 * @brief Full-like creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "empty_like.hpp"
#include "fill.hpp"

namespace mdtensor {

template <typename dtype = void, core::MPMode mpmode = core::MPMode::NONE,
          typename in_t, typename val_t>
[[nodiscard]] inline constexpr auto full_like(in_t &&in, val_t &&val) {
    auto out = empty_like<dtype>(std::forward<in_t>(in));
    fill<mpmode>(out, std::forward<val_t>(val));
    return out;
}

} // namespace mdtensor
