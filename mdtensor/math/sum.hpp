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

template <int64_t Axis, core::MPMode mpmode = core::MPMode::NONE, typename in_t,
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
        std::index_sequence<rin_rank, rin_rank - 1>{},
        std::integer_sequence<bool, false, true>{}, in_mds, out_mds);
}

template <int64_t Axis, core::MPMode mpmode = core::MPMode::NONE,
          typename dtype = void, typename in_t>
[[nodiscard]] inline constexpr auto sum(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    constexpr size_t in_rank = decltype(in_mds)::rank();
    constexpr size_t rin_rank =
        in_rank -
        static_cast<size_t>(
            ((Axis % static_cast<int64_t>(in_rank)) + (in_rank)) % in_rank);

    auto out = core::create_out<dtype>(
        std::index_sequence<rin_rank>{},
        core::slice_extents_from_right<rin_rank - 1>(in_mds.extents()), in_mds);

    sum_to<Axis, mpmode>(std::forward<in_t>(in), out);

    return out;
}

} // namespace mdtensor
