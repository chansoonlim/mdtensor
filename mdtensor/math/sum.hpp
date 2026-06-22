/**
 * @file
 * @brief Sum reduction utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"
#include "../creation/fill.hpp"
#include "add.hpp"

namespace mdtensor {
namespace detail {

template <typename in_t, typename out_t>
inline constexpr void sum_impl(in_t &&in, out_t &&out) {
    fill(std::forward<out_t>(out), 0);

    for (typename std::remove_cvref_t<in_t>::index_type i = 0; i < in.extent(0);
         i++) {
        add_to(std::forward<out_t>(out),
               core::submdspan_from_left(std::forward<in_t>(in), i),
               std::forward<out_t>(out));
    }
}

} // namespace detail

/**
 * @brief Compute sum along a specified axis (in-place).
 *
 * @tparam Axis Axis to reduce. Negative values are supported and normalized
 *         by the input rank (NumPy-like semantics).
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 * @param out Output mdspan, mdarray, scalar, etc.
 *
 * @note The reduction is performed along the specified axis and written to out.
 * @note Broadcasting is not performed; this is a reduction operation.
 *
 * @see mdtensor::sum for the out-of-place axis-reduction version that returns
 *      the result.
 */
template <int64_t Axis, MPMode mpmode = MPMode::NONE, typename in_t,
          typename out_t>
inline constexpr void sum_to(in_t &&in, out_t &&out) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    const auto out_mds = core::to_mdspan(std::forward<out_t>(out));

    constexpr size_t in_rank = decltype(in_mds)::rank();
    constexpr size_t rin_rank =
        in_rank -
        static_cast<size_t>(
            ((Axis % static_cast<int64_t>(in_rank)) + (in_rank)) % in_rank);

    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::sum_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<rin_rank, rin_rank - 1>{}, in_mds, out_mds);
}

/**
 * @brief Compute sum along a specified axis (out-of-place).
 *
 * @tparam Axis Axis to reduce. Negative values are supported and normalized
 *         by the input rank (NumPy-like semantics).
 * @tparam mpmode (optional) Parallel execution mode. Default is MPMode::NONE.
 * @tparam dtype (optional) Data type of the result. If void, deduced from
 *         input.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 *
 * @return mdarray or scalar (depending on the input rank and reduction).
 *
 * @note The reduction is performed along the specified axis.
 *
 * @see mdtensor::sum_to for the in-place version that writes into an output.
 */
template <int64_t Axis, MPMode mpmode = MPMode::NONE, typename dtype = void,
          typename in_t>
[[nodiscard]] inline constexpr auto sum(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    constexpr size_t in_rank = decltype(in_mds)::rank();
    constexpr size_t rin_rank =
        in_rank -
        static_cast<size_t>(
            ((Axis % static_cast<int64_t>(in_rank)) + (in_rank)) % in_rank);

    return core::batch_out<dtype, mpmode>(
        [](auto &&...elems) {
            detail::sum_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<rin_rank>{},
        core::slice_from_right<rin_rank - 1>(in_mds.extents()), in_mds);
}

} // namespace mdtensor
