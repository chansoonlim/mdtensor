/**
 * @file
 * @brief Copy-into-like utilities for mdtensor (copy_like).
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "copy.hpp"
#include "empty_like.hpp"

namespace mdtensor {

template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in1_t,
          typename in2_t>
[[nodiscard]] inline constexpr auto copy_like(in1_t &&in1, in2_t &&in2) {
    auto out = empty_like<dtype>(std::forward<in1_t>(in1));
    copy_to<mpmode>(std::forward<in2_t>(in2), out);
    return out;
}

} // namespace mdtensor
