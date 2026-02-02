/**
 * @file
 * @brief Exact array equality utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {

/**
 * @brief Return whether two inputs are exactly equal in shape and values.
 *
 * @tparam in1_t First input type (mdspan, mdarray, scalar, etc.).
 * @tparam in2_t Second input type (mdspan, mdarray, scalar, etc.).
 *
 * @param in1 First input tensor-like object.
 * @param in2 Second input tensor-like object.
 *
 * @return true if rank, extents, and all elements match exactly; otherwise
 *         false.
 *
 * @note This function does not apply broadcasting. Shapes must be identical.
 * @note For rank-0 inputs, this compares the two scalar values directly.
 * @note For rank>0 inputs, this checks extents and then recurses by slicing
 *       from the left.
 *
 * @see mdtensor::isclose for tolerant element-wise comparison.
 * @see mdtensor::allclose for tolerant full-tensor comparison.
 */
template <typename in1_t, typename in2_t>
[[nodiscard]] inline constexpr bool array_equal(in1_t &&in1, in2_t &&in2) {
    const auto in1_mds = core::to_const_mdspan(std::forward<in1_t>(in1));
    const auto in2_mds = core::to_const_mdspan(std::forward<in2_t>(in2));
    using in1_mds_t = decltype(in1_mds);
    using in2_mds_t = decltype(in2_mds);

    if constexpr (in1_mds_t::rank() != in2_mds_t::rank()) {
        return false;
    }

    for (size_t i = 0; i < in1_mds_t::rank(); i++) {
        if (in1_mds.extent(i) != in2_mds.extent(i)) {
            return false;
        }
    }

    if constexpr (in1_mds_t::rank() == 0) {
        return in1_mds() == in2_mds();

    } else {
        for (typename in1_mds_t::index_type i = 0; i < in1_mds.extent(0); i++) {
            if (!array_equal(core::submdspan_from_left(in1_mds, i),
                             core::submdspan_from_left(in2_mds, i))) {
                return false;
            }
        }
    }

    return true;
}

} // namespace mdtensor
