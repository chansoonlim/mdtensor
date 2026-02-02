/**
 * @file
 * @brief Reshape utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {

/**
 * @brief Reshape a tensor into new extents (view-only).
 *
 * @tparam exts_t Extents type satisfying extents_c.
 * @tparam in_t Input type (mdspan, mdarray, scalar, etc.).
 *
 * @param in Input tensor-like object.
 * @param new_exts Target extents (shape). Default constructs exts_t if omitted.
 *
 * @return A reshaped mdspan view sharing the same underlying storage.
 *
 * @note This function does not allocate new memory.
 * @note The total number of elements must match between the input and the
 *       requested extents, as enforced by core::reshape.
 *
 * @see mdtensor::expand_dims for inserting singleton dimensions.
 * @see mdtensor::concatenate for joining tensors along an axis.
 */
template <extents_c exts_t, typename in_t>
[[nodiscard]] inline constexpr auto
reshape(in_t &&in, exts_t &&new_exts = exts_t{}) noexcept {
    return core::reshape(core::to_mdspan(std::forward<in_t>(in)),
                         std::forward<exts_t>(new_exts));
}

} // namespace mdtensor
