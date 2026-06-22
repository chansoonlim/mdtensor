/**
 * @file
 * @brief All-close comparison utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "all.hpp"
#include "isclose.hpp"

namespace mdtensor {

/**
 * @brief Return whether two inputs are element-wise close for all elements.
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam in1_t First input type (mdspan, mdarray, scalar, etc.).
 * @tparam in2_t Second input type (mdspan, mdarray, scalar, etc.).
 *
 * @param in1 First input tensor-like object.
 * @param in2 Second input tensor-like object.
 * @param rtol Relative tolerance.
 * @param atol Absolute tolerance.
 *
 * @return true if all elements satisfy isclose(in1, in2, rtol, atol),
 *         otherwise false.
 *
 * @note This is equivalent to all(isclose(in1, in2, rtol, atol)).
 * @note Broadcasting semantics follow those of mdtensor::isclose.
 *
 * @see mdtensor::isclose for the element-wise predicate.
 * @see mdtensor::all for logical reduction utilities.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t>
[[nodiscard]] inline constexpr bool allclose(in1_t &&in1, in2_t &&in2,
                                             const double &rtol = 1e-05,
                                             const double &atol = 1e-08) {
    return all(isclose<int8_t, mpmode>(std::forward<in1_t>(in1),
                                       std::forward<in2_t>(in2), rtol, atol));
}

} // namespace mdtensor
