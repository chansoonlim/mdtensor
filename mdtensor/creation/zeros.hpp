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

/**
 * @brief Create a tensor filled with zeros (general extents overload).
 *
 * @tparam dtype Output value type.
 * @tparam exts_t (optional) Extents type specifying the tensor shape.
 * Default is `extents<size_t>`.
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param exts Extents describing the output shape.
 *
 * @return mdarray filled with zeros.
 *
 * @note Equivalent to full(0, exts).
 *
 * @see mdtensor::full for the general fill-value version.
 */
template <typename dtype, extents_c exts_t = extents<size_t>,
          MPMode mpmode = MPMode::NONE>
[[nodiscard]] inline constexpr auto zeros(exts_t &&exts = exts_t{}) {
    return full<dtype, exts_t, mpmode>(0, std::forward<exts_t>(exts));
}

/**
 * @brief Create a 1D tensor filled with zeros (length overload).
 *
 * @tparam dtype Output value type.
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param len Length of the output 1D tensor.
 *
 * @return 1D mdarray of length `len`, filled with zeros.
 *
 * @note Equivalent to full(0, dims<1>{len}).
 *
 * @see mdtensor::zeros(extents) for the general extents version.
 */
template <typename dtype, MPMode mpmode = MPMode::NONE>
[[nodiscard]] inline constexpr auto zeros(const size_t len) {
    return full<dtype, dims<1>, mpmode>(0, dims<1>{len});
}

} // namespace mdtensor
