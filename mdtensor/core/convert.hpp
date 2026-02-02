/**
 * @file
 * @brief Conversion utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "type.hpp"

namespace mdtensor {
namespace core {

template <typename in_t>
[[nodiscard]] inline constexpr auto to_mdspan(in_t &&in) noexcept {
    using in_base_t = std::remove_reference_t<in_t>;

    if constexpr (mdspan_c<in_base_t>) {
        return std::forward<in_t>(in);

    } else if constexpr (mdarray_c<in_base_t>) {
        return in.to_mdspan();

    } else if constexpr (requires { in.to_mdspan(); }) {
        return in.to_mdspan();

    } else {
        auto exts = extents<size_t>{};
        return mdspan<in_base_t, decltype(exts)>{&in, exts};
    }
}

template <md_c in_t>
[[nodiscard]] inline constexpr auto to_const(in_t &&in) noexcept {
    using in_base_t = std::remove_reference_t<in_t>;

    if constexpr (std::is_const_v<typename in_base_t::element_type>) {
        return std::forward<in_t>(in);

    } else {
        using element_t = const typename in_base_t::value_type;

        return mdspan<element_t, typename in_base_t::extents_type,
                      typename in_base_t::layout_type,
                      default_accessor<element_t>>(in.data_handle(),
                                                   in.mapping(), {});
    }
}

template <typename in_t>
[[nodiscard]] inline constexpr auto to_const_mdspan(in_t &&in) noexcept {
    return to_const(to_mdspan(std::forward<in_t>(in)));
}

template <md_c in_t>
[[nodiscard]] inline constexpr bool is_always_reshapable() noexcept {
    return in_t::is_always_unique() && in_t::is_always_exhaustive() &&
           in_t::is_always_strided();
}

template <md_c in_t>
[[nodiscard]] inline constexpr bool is_reshapable(in_t &&in) noexcept {
    return in.is_unique() && in.is_exhaustive() && in.is_strided();
}

template <extents_c exts_t, typename in_t>
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

    static_assert(core::static_size<typename in_mds_t::extents_type>() ==
                      core::static_size<exts_t>(),
                  "static_reshape requires same number of elements.");

    return mdspan<typename in_mds_t::element_type, exts_t>{
        in_mds.data_handle()};
}

template <extents_c exts_t, typename in_t>
[[nodiscard]] inline constexpr auto
reshape(in_t &&in, exts_t &&new_exts = exts_t{}) noexcept {
    const auto in_mds = to_mdspan(std::forward<in_t>(in));
    using in_mds_t = decltype(in_mds);

    assert(is_reshapable(in_mds));
    assert(core::size(in_mds.extents()) ==
           core::size(std::forward<exts_t>(new_exts)));

    return mdspan<typename in_mds_t::element_type, std::remove_cvref_t<exts_t>>{
        in_mds.data_handle(), std::forward<exts_t>(new_exts)};
}

template <typename T>
using to_mdspan_t = decltype(to_mdspan(std::declval<T>()));

template <typename T>
using to_mdcontainer_t =
    std::conditional_t<to_mdspan_t<T>::rank() == 0,
                       typename to_mdspan_t<T>::value_type,
                       mdarray<typename to_mdspan_t<T>::value_type,
                               typename to_mdspan_t<T>::extents_type>>;

} // namespace core
} // namespace mdtensor
