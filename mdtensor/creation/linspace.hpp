/**
 * @file
 * @brief Linspace utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../math/add.hpp"
#include "../math/multiply.hpp"
#include "../math/subtract.hpp"
#include "copy.hpp"

namespace mdtensor {
namespace detail {

template <typename dtype = void, core::Backend backend = core::Backend::AUTO>
constexpr void linspace_impl(auto &&start, auto &&stop, auto &&out,
                             const bool endpoint = true) {
    const auto start_mds =
        core::to_const_mdspan(std::forward<decltype(start)>(start));
    const auto stop_mds =
        core::to_const_mdspan(std::forward<decltype(stop)>(stop));
    const auto out_mds =
        core::to_output_mdspan(std::forward<decltype(out)>(out));

    using index_t = typename decltype(out_mds)::index_type;

    const index_t num = out_mds.extent(0);

    if (num == 0) {
        return;

    } else if (num == 1) {
        if (!endpoint) {
            static_cast<void>(copy<void, backend>(
                start_mds, core::submdspan_from_left(out_mds, 0)));

        } else {
            static_cast<void>(copy<void, backend>(
                stop_mds, core::submdspan_from_left(out_mds, 0)));
        }

    } else {
        using calc_t = core::floating_calc_type_t<dtype, decltype(start_mds),
                                                  decltype(stop_mds)>;

        const calc_t scale =
            calc_t{1} / static_cast<calc_t>(endpoint ? num - 1 : num);

        const auto actual_step = multiply<calc_t, backend>(
            subtract<calc_t, backend>(stop_mds, start_mds), scale);

        static_cast<void>(copy<void, backend>(
            start_mds, core::submdspan_from_left(out_mds, 0)));

        for (index_t i = 1; i < num; i++) {
            static_cast<void>(add<calc_t, backend>(
                core::submdspan_from_left(out_mds, i - 1), actual_step,
                core::submdspan_from_left(out_mds, i)));
        }

        if (endpoint) {
            // Ensure that the last element is exactly equal to the stop value
            static_cast<void>(copy<void, backend>(
                stop_mds, core::submdspan_from_left(out_mds, num - 1)));
        }
    }
}

} // namespace detail

template <std::int64_t axis = 0, typename dtype = void,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto linspace(auto &&shape, auto &&start, auto &&stop,
                                      const bool endpoint = true,
                                      out_t &&out = out_t{std::nullopt}) {
    // TODO: modify batch and use linspace_impl.
    const auto exts = core::to_extents(std::forward<decltype(shape)>(shape));

    static_assert(exts.rank() == 1,
                  "The extents for linspace must be a 1D tensor.");

    const auto [bcasts, bexts] = core::broadcast(
        std::index_sequence<0, 0>{}, std::integer_sequence<bool, true, true>{},
        std::forward<decltype(start)>(start),
        std::forward<decltype(stop)>(stop));
    const auto start_bcast = std::get<0>(bcasts);
    const auto stop_bcast = std::get<1>(bcasts);

    constexpr std::size_t baxis =
        static_cast<std::size_t>(core::bounding_index(axis, bexts.rank()));
    constexpr std::size_t out_urank = bexts.rank() + 1 - baxis;

    using calc_t = core::floating_calc_type_t<dtype, decltype(start_bcast),
                                              decltype(stop_bcast)>;

    auto out_md = core::resolve_output<calc_t>(
        std::forward<decltype(out)>(out),
        core::compose_extents(
            core::slice_extents_from_left<baxis>(bexts), exts,
            core::slice_extents_from_right<out_urank - 1>(bexts)));

    core::batch<backend>(
        [&](auto &&...elems) {
            detail::linspace_impl<calc_t, core::Backend::NATIVE>(
                std::forward<decltype(elems)>(elems)..., endpoint);
        },
        std::index_sequence<out_urank - 1, out_urank - 1, out_urank>{},
        std::integer_sequence<bool, true, true, false>{}, start_bcast,
        stop_bcast, out_md);

    return out_md;
}

template <std::size_t num, std::int64_t axis = 0, typename dtype = void,
          core::Backend backend = core::Backend::AUTO,
          typename out_t = std::nullopt_t>
[[nodiscard]] constexpr auto linspace(auto &&start, auto &&stop,
                                      const bool endpoint = true,
                                      out_t &&out = out_t{std::nullopt}) {
    return linspace<axis, dtype, backend>(
        core::extents<std::size_t, num>{}, std::forward<decltype(start)>(start),
        std::forward<decltype(stop)>(stop), endpoint,
        std::forward<decltype(out)>(out));
}

} // namespace mdtensor
