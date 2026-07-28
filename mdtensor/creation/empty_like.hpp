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

template <typename dtype = void, typename in_t>
[[nodiscard]] inline constexpr auto empty_like(in_t &&in) noexcept {
    using value_t = std::conditional_t<std::is_void_v<dtype>,
                                       core::value_type_t<in_t>, dtype>;

    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    return empty<value_t>(in_mds.extents());
}

} // namespace mdtensor
