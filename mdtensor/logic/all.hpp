/**
 * @file
 * @brief Logical all-reduction utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {
namespace detail {

template <typename in_t, typename out_t>
inline constexpr void all_impl(in_t &&in, out_t &&out) {
    using in_base_t = std::remove_cvref_t<in_t>;
    using out_base_t = std::remove_cvref_t<out_t>;

    static_assert(in_base_t::rank() == out_base_t::rank() + 1,
                  "Input rank must be one greater than output rank.");

    if constexpr (in_base_t::rank() == 1) {
        for (typename in_base_t::index_type i = 0; i < in.extent(0); i++) {
            if (!static_cast<bool>(in(i))) {
                out() = false;
                return;
            }
        }

        out() = true;

    } else {
        for (typename in_base_t::index_type i = 0;
             i < in.extent(in_base_t::rank() - 1); i++) {
            all_impl(core::submdspan_from_right(std::forward<in_t>(in), i),
                     core::submdspan_from_right(std::forward<out_t>(out), i));
        }
    }
}

} // namespace detail

template <int64_t Axis, MPMode mpmode = MPMode::NONE, typename in_t,
          typename out_t>
inline constexpr void all(in_t &&in, out_t &&out) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));
    const auto out_mds = core::to_mdspan(std::forward<out_t>(out));

    constexpr size_t in_rank = decltype(in_mds)::rank();
    constexpr size_t rin_rank =
        in_rank -
        static_cast<size_t>(
            ((Axis % static_cast<int64_t>(in_rank)) + (in_rank)) % in_rank);

    core::batch<mpmode>(
        [](auto &&...elems) {
            detail::all_impl(std::forward<decltype(elems)>(elems)...);
        },
        std::index_sequence<rin_rank, rin_rank - 1>{},
        std::integer_sequence<bool, false, true>{}, in_mds, out_mds);
}

template <int64_t Axis, typename dtype = bool, MPMode mpmode = MPMode::NONE,
          typename in_t>
[[nodiscard]] inline constexpr auto all(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    constexpr size_t in_rank = decltype(in_mds)::rank();
    constexpr size_t rin_rank =
        in_rank -
        static_cast<size_t>(
            ((Axis % static_cast<int64_t>(in_rank)) + (in_rank)) % in_rank);

    auto out = core::create_out<dtype>(
        std::index_sequence<rin_rank>{},
        core::slice_extents_from_right<rin_rank - 1>(in_mds.extents()), in_mds);

    all<Axis, mpmode>(std::forward<in_t>(in), out);

    return out;
}

template <typename in_t> [[nodiscard]] inline constexpr bool all(in_t &&in) {
    const auto in_mds = core::to_const_mdspan(std::forward<in_t>(in));

    if constexpr (decltype(in_mds)::rank() == 0) {
        return static_cast<bool>(in_mds());

    } else {
        using index_t = typename decltype(in_mds)::index_type;

        for (index_t i = 0; i < in_mds.extent(0); i++) {
            if (!all(core::submdspan_from_left(in_mds, i))) {
                return false;
            }
        }

        return true;
    }
}

} // namespace mdtensor
