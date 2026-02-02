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

/**
 * @brief Create a tensor filled with ones (general extents overload).
 *
 * @tparam dtype Output value type.
 * @tparam exts_t (optional) Extents type specifying the tensor shape.
 * Default is `extents<size_t>`.
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param exts Extents describing the output shape.
 *
 * @return mdarray filled with ones.
 *
 * @note Equivalent to full(1, exts).
 *
 * @see mdtensor::full for the general fill-value version.
 */
template <typename dtype, extents_c exts_t = extents<size_t>,
          MPMode mpmode = MPMode::NONE>
[[nodiscard]] inline constexpr auto ones(exts_t &&exts = exts_t{}) {
    return full<dtype, exts_t, mpmode>(1, std::forward<exts_t>(exts));
}

/**
 * @brief Create a 1D tensor filled with ones (length overload).
 *
 * @tparam dtype Output value type.
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param len Length of the output 1D tensor.
 *
 * @return 1D mdarray of length `len`, filled with ones.
 *
 * @note Equivalent to full(1, dims<1>{len}).
 *
 * @see mdtensor::ones(extents) for the general extents version.
 */
template <typename dtype, MPMode mpmode = MPMode::NONE>
[[nodiscard]] inline constexpr auto ones(const size_t len) {
    return full<dtype, dims<1>, mpmode>(1, dims<1>{len});
}

} // namespace mdtensor
