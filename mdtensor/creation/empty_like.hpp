/**
 * @file
 * @brief Empty-like tensor creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "empty.hpp"

namespace mdtensor {

template <typename dtype = void>
[[nodiscard]] constexpr auto empty_like(auto &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<decltype(in)>(in));

    using value_t = core::output_value_t<dtype, decltype(in_mds)>;

    return empty<value_t>(in_mds.extents());
}

} // namespace mdtensor
