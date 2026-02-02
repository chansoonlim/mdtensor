/**
 * @file
 * @brief Dimension expansion utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "reshape.hpp"

namespace mdtensor {

/**
 * @brief Insert a new axis of length 1 into the input (out-of-place view).
 *
 * @tparam Axis Axis position to insert the new dimension. Negative values are
 *         supported and normalized (NumPy-like semantics).
 * @tparam in_t Input type (mdspan, mdarray, scalar, etc.).
 *
 * @param in Input tensor-like object.
 *
 * @return A reshaped view with rank increased by 1.
 *
 * @note This function does not allocate new storage; it returns a view
 *       (via reshape or direct mdspan construction for rank-0).
 * @note For rank-0 inputs, the resulting shape is (1).
 * @note For rank>0 inputs, the new shape matches the input with a singleton
 *       inserted at Axis.
 *
 * @see mdtensor::reshape for general reshaping utilities.
 */
template <int64_t Axis, typename in_t>
[[nodiscard]] inline constexpr auto expand_dims(in_t &&in) noexcept {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    using in_mds_t = decltype(in_mds);

    constexpr size_t rank = in_mds_t::rank();

    if constexpr (rank == 0) {
        auto new_extents = extents<typename in_mds_t::index_type, 1>{1};
        return mdspan<typename in_mds_t::element_type, decltype(new_extents)>{
            in_mds.data_handle(), new_extents};

    } else {
        constexpr size_t axis = static_cast<size_t>(
            ((Axis % static_cast<int64_t>(rank + 1)) + (rank + 1)) %
            (rank + 1));

        const auto new_extents = [&in_mds]<size_t... Is>(
                                     std::index_sequence<Is...>) {
            return extents<
                typename in_mds_t::index_type,
                (Is < axis
                     ? in_mds_t::static_extent(Is)
                     : (Is == axis ? 1 : in_mds_t::static_extent(Is - 1)))...>{
                (Is < axis ? in_mds.extent(Is)
                           : (Is == axis ? 1 : in_mds.extent(Is - 1)))...};
        }(std::make_index_sequence<rank + 1>{});

        return reshape(std::forward<in_t>(in), new_extents);
    }
}

} // namespace mdtensor
