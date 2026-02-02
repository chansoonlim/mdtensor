/**
 * @file
 * @brief Array equivalence utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "all.hpp"
#include "equal.hpp"

namespace mdtensor {

/**
 * @brief Return whether two inputs are element-wise equal for all elements,
 *        with broadcasting.
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam in1_t First input type (mdspan, mdarray, scalar, etc.).
 * @tparam in2_t Second input type (mdspan, mdarray, scalar, etc.).
 *
 * @param in1 First input tensor-like object.
 * @param in2 Second input tensor-like object.
 *
 * @return true if all elements satisfy equal(in1, in2) after broadcasting;
 *         otherwise false.
 *
 * @note This is equivalent to all(equal(in1, in2)).
 * @note Broadcasting semantics follow those of mdtensor::equal.
 *
 * @see mdtensor::equal for the element-wise predicate.
 * @see mdtensor::all for logical reduction utilities.
 * @see mdtensor::array_equal for exact shape-and-value equality without
 *      broadcasting.
 */
template <MPMode mpmode = MPMode::NONE, typename in1_t, typename in2_t>
[[nodiscard]] inline constexpr bool array_equiv(in1_t &&in1, in2_t &&in2) {
    return all(equal<mpmode, int8_t>(std::forward<in1_t>(in1),
                                     std::forward<in2_t>(in2)));
}

} // namespace mdtensor
