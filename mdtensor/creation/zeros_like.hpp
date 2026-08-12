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

template <typename dtype = void, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto zeros_like(auto &&in,
                                        out_t &&out = out_t{std::nullopt}) {
    return full_like<dtype, backend>(std::forward<decltype(in)>(in), 0,
                                     std::forward<decltype(out)>(out));
}

} // namespace mdtensor
