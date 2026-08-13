/**
 * @file
 * @brief Zeros tensor creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "full.hpp"

namespace mdtensor {

template <core::arithmetic value_t = double,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto zeros(auto &&shape,
                                   out_t &&out = out_t{std::nullopt}) {
    return full<value_t, backend>(std::forward<decltype(shape)>(shape),
                                  value_t{0}, std::forward<decltype(out)>(out));
}

} // namespace mdtensor
