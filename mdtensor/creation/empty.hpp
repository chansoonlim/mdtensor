/**
 * @file
 * @brief Empty tensor creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {

/**
 * @brief Create an uninitialized tensor with the given extents.
 *
 * @tparam dtype Element type of the created tensor.
 * @tparam exts_t (optional) Extents type describing the tensor shape.
 *         Default is extents<size_t>.
 *
 * @param exts Tensor extents (shape). If omitted, creates a scalar-like tensor.
 *
 * @return A newly allocated mdarray-like tensor of type `dtype` with the given
 * extents.
 *
 * @note This function does not initialize memory. Use `zeros`, `ones`, or
 *       `full` if initialization is required.
 *
 * @see mdtensor::empty_like for creating an uninitialized tensor matching the
 * shape of an existing tensor.
 */
template <typename dtype, extents_c exts_t = extents<size_t>>
[[nodiscard]] inline constexpr auto empty(exts_t &&exts = exts_t{}) noexcept {
    return core::create_data<dtype>(std::forward<exts_t>(exts));
}

} // namespace mdtensor
