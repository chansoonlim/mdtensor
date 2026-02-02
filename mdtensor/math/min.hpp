/**
 * @file
 * @brief Minimum reduction utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {
namespace detail {

template <md_c in_t, md_c out_t>
    requires(std::remove_cvref_t<out_t>::rank() == 0)
inline constexpr void min_impl(in_t &&in, out_t &&out) {
    using in_base_t = std::remove_cvref_t<in_t>;
    using index_t = typename in_base_t::index_type;

    out() = in(0);

    for (index_t i = 1; i < in.extent(0); i++) {
        if (in(i) < out()) {
            out() = in(i);
        }
    }
}

template <md_c in_t, md_c out_t>
    requires(std::remove_cvref_t<out_t>::rank() > 0)
inline constexpr void min_impl(in_t &&in, out_t &&out) {
    using in_base_t = std::remove_cvref_t<in_t>;
    using index_t = typename in_base_t::index_type;

    for (index_t i = 0; i < in.extent(in_base_t::rank() - 1); i++) {
        min_impl(core::submdspan_from_right(in, i),
                 core::submdspan_from_right(out, i));
    }
}

} // namespace detail

namespace {

template <md_c in_t, arithmetic_c out_t>
inline constexpr void min_arithmetic_impl(const in_t &in, out_t &out) {
    // NOTE: CANNOT USE THIS FUNCTION DIRECTLY. OUT SHOULD BE INITIALIZED
    using in_base_t = std::remove_cvref_t<in_t>;
    using index_t = typename in_base_t::index_type;

    if constexpr (in_base_t::rank() == 0) {
        if (in() < out) {
            out = in();
        }

    } else {
        for (index_t i = 0; i < in.extent(0); i++) {
            min_arithmetic_impl(core::submdspan_from_left(in, i), out);
        }
    }
}

} // namespace

/**
 * @brief Compute minimum along a specified axis (in-place).
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
 * @see mdtensor::min for the out-of-place axis-reduction version that returns
 *      the result.
 */
template <int64_t Axis, MPMode mpmode = MPMode::NONE, typename in_t,
          typename out_t>
inline constexpr void min_to(in_t &&in, out_t &&out) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    const auto out_mds = core::to_mdspan(std::forward<out_t>(out));

    constexpr size_t in_rank = decltype(in_mds)::rank();
    constexpr size_t rin_rank =
        in_rank -
        static_cast<size_t>(
            ((Axis % static_cast<int64_t>(in_rank)) + (in_rank)) % in_rank);

    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::min_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<rin_rank, rin_rank - 1>{}, in_mds, out_mds);
}

/**
 * @brief Compute minimum along a specified axis (out-of-place).
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
 * @see mdtensor::min_to for the in-place version that writes into an output.
 */
template <int64_t Axis, MPMode mpmode = MPMode::NONE, typename dtype = void,
          typename in_t>
[[nodiscard]] inline constexpr auto min(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    constexpr size_t in_rank = decltype(in_mds)::rank();
    constexpr size_t rin_rank =
        in_rank -
        static_cast<size_t>(
            ((Axis % static_cast<int64_t>(in_rank)) + (in_rank)) % in_rank);

    return core::batch_out<mpmode, dtype>(
        [](auto &&...elems) {
            detail::min_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<rin_rank>{},
        core::slice_from_right<rin_rank - 1>(in_mds.extents()), in_mds);
}

/**
 * @brief Compute minimum over all elements (full reduction).
 *
 * @tparam in_t Input type.
 *
 * @param in Input mdspan, mdarray, scalar, etc.
 *
 * @return Minimum value as an arithmetic scalar.
 *
 * @note This overload reduces all elements to a single scalar value.
 * @note The accumulator is initialized to the highest representable value.
 */
template <typename in_t> [[nodiscard]] inline constexpr auto min(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    using in_mds_t = decltype(in_mds);

    typename in_mds_t::value_type out =
        std::numeric_limits<typename in_mds_t::value_type>::max();

    min_arithmetic_impl(in_mds, out);

    return out;
}

} // namespace mdtensor
