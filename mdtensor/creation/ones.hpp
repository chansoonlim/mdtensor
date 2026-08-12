/**
 * @file
 * @brief Ones tensor creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "full.hpp"

namespace mdtensor {

template <typename dtype = double, core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto ones(auto &&shape,
                                  out_t &&out = out_t{std::nullopt}) {
    using calc_t = core::output_value_with_nullopt_t<dtype, decltype(out)>;

    return full<calc_t, backend>(std::forward<decltype(shape)>(shape),
                                 calc_t{1}, std::forward<decltype(out)>(out));
}

} // namespace mdtensor
