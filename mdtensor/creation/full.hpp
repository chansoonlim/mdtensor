/**
 * @file
 * @brief Full creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "empty.hpp"
#include "fill.hpp"

namespace mdtensor {

template <typename dtype, core::MPMode mpmode = core::MPMode::NONE,
          core::extents_c exts_t = core::stdex::extents<uint8_t>>
[[nodiscard]] inline constexpr auto full(dtype &&val,
                                         exts_t &&exts = exts_t{}) {
    auto out = empty<dtype>(std::forward<exts_t>(exts));
    fill<mpmode>(out, std::forward<dtype>(val));
    return out;
}

} // namespace mdtensor
