/**
 * @file
 * @brief Full-like creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "full.hpp"

namespace mdtensor {

template <typename dtype = void, core::Backend backend = core::Backend::AUTO>
[[nodiscard]] constexpr auto full_like(auto &&in, auto &&val) {
    using value_t = core::output_value_t<dtype, decltype(in)>;

    return full<value_t, backend>(in.extents(),
                                  std::forward<decltype(val)>(val));
}

} // namespace mdtensor
