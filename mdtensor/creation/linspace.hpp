/**
 * @file
 * @brief Linspace utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {
namespace detail {

template <core::md_c start_t, core::md_c stop_t, core::md_c out_t>
    requires(std::remove_cvref_t<start_t>::rank() == 0 &&
             std::remove_cvref_t<stop_t>::rank() == 0 &&
             std::remove_cvref_t<out_t>::rank() == 1)
inline constexpr void linspace_impl(start_t &&start, stop_t &&stop, out_t &&out,
                                    const bool endpoint = true) noexcept {
    using out_base_t = std::remove_cvref_t<out_t>;

    using value_t = typename out_base_t::value_type;
    using index_t = typename out_base_t::index_type;

    const index_t num = out.extent(0);

    if (num == 0) [[unlikely]] {
        // do nothing

    } else if (num == 1) [[unlikely]] {
        out(0) = static_cast<value_t>(start());

    } else {
        const value_t start_val = static_cast<value_t>(start());
        const value_t stop_val = static_cast<value_t>(stop());
        const value_t step =
            (stop_val - start_val) / (endpoint ? num - 1 : num);

        for (index_t i = 0; i < num; i++) {
            out(i) = start_val + step * i;
        }
    }
}

} // namespace detail

template <int64_t Axis = 0, typename start_t, typename stop_t, typename out_t>
inline constexpr void linspace_to(start_t &&start, stop_t &&stop, out_t &&out,
                                  const bool endpoint = true) noexcept {
    const auto start_mds = core::to_const_mdspan(std::forward<start_t>(start));
    const auto stop_mds = core::to_const_mdspan(std::forward<stop_t>(stop));
    const auto out_mds = core::to_mdspan(std::forward<out_t>(out));

    using start_mds_t = decltype(start_mds);
    using stop_mds_t = decltype(stop_mds);
    using out_mds_t = decltype(out_mds);

    static_assert(start_mds_t::rank() == stop_mds_t::rank() &&
                      start_mds_t::rank() + 1 == out_mds_t::rank(),
                  "linspace does not support broadcasting");
    // TODO: support broadcasting without changing rank.

    if constexpr (start_mds_t::rank() == 0) {
        detail::linspace_impl(start_mds, stop_mds, out_mds, endpoint);

    } else {
        constexpr size_t out_rank = out_mds_t::rank();
        constexpr size_t axis = static_cast<size_t>(
            ((Axis % static_cast<int64_t>(out_rank)) + (out_rank)) % out_rank);
        constexpr size_t lspace = axis == 0 ? 1 : 0;

        assert(start_mds.extent(0) == stop_mds.extent(0));
        assert(start_mds.extent(0) == out_mds.extent(lspace));

        for (typename out_mds_t::index_type i = 0; i < out_mds.extent(lspace);
             i++) {
            linspace_to<axis - (lspace == 0 ? 1 : 0)>(
                core::submdspan_from_left(start_mds, i),
                core::submdspan_from_left(stop_mds, i),
                core::submdspan_from_left<lspace>(out_mds, i), endpoint);
        }
    }
}

template <int64_t Axis = 0,
          core::extents_c exts_t = core::stdex::extents<uint8_t, 50>,
          typename dtype = void, typename start_t, typename stop_t>
    requires(exts_t::rank() == 1)
[[nodiscard]] inline constexpr auto
linspace(start_t &&start, stop_t &&stop, const exts_t &exts = exts_t{},
         const bool endpoint = true) noexcept {
    const auto start_mds = core::to_const_mdspan(std::forward<start_t>(start));
    const auto stop_mds = core::to_const_mdspan(std::forward<stop_t>(stop));

    using start_mds_t = decltype(start_mds);
    using stop_mds_t = decltype(stop_mds);

    using value_t = std::conditional_t<
        !std::is_void_v<dtype>, dtype,
        core::common_data_type_t<typename start_mds_t::value_type,
                                 typename stop_mds_t::value_type>>;

    constexpr size_t out_rank = start_mds_t::rank() + 1;
    constexpr size_t axis = static_cast<size_t>(
        ((Axis % static_cast<int64_t>(out_rank)) + (out_rank)) % out_rank);

    const auto bexts = start_mds.extents();
    auto out = core::create_out<value_t>(core::compose_extents(
        core::slice_extents_from_left<axis>(bexts), exts,
        core::slice_extents_from_right<decltype(bexts)::rank() - axis>(bexts)));

    linspace_to<Axis>(start_mds, stop_mds, out, endpoint);

    return out;
}

} // namespace mdtensor
