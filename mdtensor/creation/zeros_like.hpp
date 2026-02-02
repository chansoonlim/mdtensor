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

/**
 * @brief Create a tensor filled with zeros, matching the shape of the input.
 *
 * @tparam dtype (optional) Output value type. If void, deduced from input.
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam in_t Input tensor type (mdspan, mdarray, etc.).
 *
 * @param in Input tensor whose extents determine the output shape.
 *
 * @return mdarray with the same extents as `in`, filled with zeros.
 *
 * @note Equivalent to full_like(in, 0).
 *
 * @see mdtensor::full_like for the general fill-value version.
 */
template <typename dtype = void, MPMode mpmode = MPMode::NONE, typename in_t>
[[nodiscard]] inline constexpr auto zeros_like(in_t &&in) {
    return full_like<dtype, mpmode>(std::forward<in_t>(in), 0);
}

} // namespace mdtensor
