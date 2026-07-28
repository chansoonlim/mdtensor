/**
 * @file
 * @brief Mdspan utilities for mdtensor.
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
        auto exts = stdex::extents<size_t>{};
        return stdex::mdspan<in_base_t, decltype(exts)>{&in, exts};
    }
}

template <typename in_t>
[[nodiscard]] inline constexpr auto to_const_mdspan(in_t &&in) noexcept {
    auto mds = to_mdspan(std::forward<in_t>(in));

    return stdex::mdspan<
        const typename decltype(mds)::value_type,
        typename decltype(mds)::extents_type,
        typename decltype(mds)::layout_type,
        stdex::default_accessor<const typename decltype(mds)::value_type>>(
        mds.data_handle(), mds.mapping(), {});
}

template <typename T>
using to_mdspan_t = decltype(to_mdspan(std::declval<T>()));

} // namespace core
} // namespace mdtensor
