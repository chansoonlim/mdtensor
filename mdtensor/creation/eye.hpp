/**
 * @file
 * @brief Identity matrix (eye) creation utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "empty.hpp"

namespace mdtensor {
namespace detail {

template <typename in_t> inline constexpr void eye_impl(in_t &&in) {
    using in_base_t = std::remove_cvref_t<in_t>;
    using index_t = typename in_base_t::index_type;

    for (index_t i = 0; i < in.extent(0); i++) {
        for (index_t j = 0; j < in.extent(1); j++) {
            in(i, j) = (i == j) ? 1 : 0;
        }
    }
}

} // namespace detail

/**
 * @brief Fill a 2D tensor with an identity matrix pattern (in-place).
 *
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Output matrix to fill (mdspan, mdarray, etc.). Rank must be greater
 *        than or equal to 2.
 *
 * @note This function writes `1` on the main diagonal and `0` elsewhere.
 *
 * @see mdtensor::eye for the out-of-place version that allocates and returns
 * an identity matrix.
 */
template <MPMode mpmode = MPMode::NONE, typename in_t>
inline constexpr void eye_to(in_t &&in) {
    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::eye_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<2>{}, core::to_mdspan(std::forward<in_t>(in)));
}

/**
 * @brief Create an identity matrix (out-of-place).
 *
 * @tparam dtype Element type of the created matrix.
 * @tparam exts_t Extents type describing the matrix shape.
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param exts Output matrix extents (shape). Rank must be greater than or equal
 * to 2.
 *
 * @return A newly allocated matrix filled as an identity matrix.
 *
 * @note Equivalent to allocating `empty<dtype>(exts)` and then calling
 *       `eye_to` on it.
 *
 * @see mdtensor::eye_to for the in-place version that fills an existing output.
 */
template <typename dtype, extents_c exts_t, MPMode mpmode = MPMode::NONE>
[[nodiscard]] inline constexpr auto eye(exts_t &&exts = exts_t{}) {
    auto out = empty<dtype>(std::forward<exts_t>(exts));
    eye_to<mpmode>(out);
    return out;
}

} // namespace mdtensor
