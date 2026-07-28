/**
 * @file
 * @brief Reshape utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {
namespace detail {

template <md_c in_t>
[[nodiscard]] inline constexpr bool is_always_reshapable() noexcept {
    return in_t::is_always_unique() && in_t::is_always_exhaustive() &&
           in_t::is_always_strided();
}

template <md_c in_t>
[[nodiscard]] inline constexpr bool is_reshapable(in_t &&in) noexcept {
    return in.is_unique() && in.is_exhaustive() && in.is_strided();
}

template <typename in_t, extents_c exts_t>
[[nodiscard]] inline constexpr auto
static_reshape(in_t &&in, exts_t &&new_exts = exts_t{}) noexcept {
    const auto in_mds = to_mdspan(std::forward<in_t>(in));
    using in_mds_t = decltype(in_mds);

    static_assert(exts_t::rank_dynamic() == 0,
                  "static_reshape requires static extents.");

    static_assert(in_mds_t::rank_dynamic() == 0,
                  "static_reshape requires static mdspan.");

    static_assert(is_always_reshapable<in_mds_t>(),
                  "static_reshape requires always reshapable mdspan.");

    static_assert(core::extents_size(typename in_mds_t::extents_type{}) ==
                      core::extents_size(exts_t{}),
                  "static_reshape requires same number of elements.");

    return mdspan<typename in_mds_t::element_type, exts_t>{
        in_mds.data_handle()};
}

} // namespace detail

template <typename in_t, extents_c exts_t>
[[nodiscard]] inline constexpr auto
reshape(in_t &&in, exts_t &&new_exts = exts_t{}) noexcept {
    const auto in_mds = core::to_mdspan(std::forward<in_t>(in));
    using in_mds_t = decltype(in_mds);

    assert(is_reshapable(in_mds));
    assert(core::extents_size(in_mds.extents()) ==
           core::extents_size(std::forward<exts_t>(new_exts)));

    return mdspan<typename in_mds_t::element_type, std::remove_cvref_t<exts_t>>{
        in_mds.data_handle(), std::forward<exts_t>(new_exts)};
}

} // namespace mdtensor
