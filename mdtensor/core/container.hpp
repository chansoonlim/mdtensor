/**
 * @file
 * @brief Container utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "extents.hpp"
#include "mdspan.hpp"

namespace mdtensor {
namespace core {

using bool_value_t = int8_t;

template <typename value_t, extents_c extent_t>
using container = std::conditional_t<
    extent_t::rank() == 0, value_t,
    std::conditional_t<
        extent_t::rank_dynamic() == 0,
        stdex::mdarray<value_t, extent_t, stdex::layout_right,
                       std::array<value_t, core::extents_size(extent_t{})>>,
        std::conditional_t<
            std::is_same_v<value_t, bool>,
            stdex::mdarray<bool_value_t, extent_t, stdex::layout_right,
                           std::vector<bool_value_t>>,
            stdex::mdarray<value_t, extent_t, stdex::layout_right,
                           std::vector<value_t>>>>>;

template <typename value_t, extents_c extent_t>
[[nodiscard]] inline constexpr auto make_container(extent_t &&exts) noexcept {
    using base_t = std::remove_cvref_t<extent_t>;

    if constexpr (base_t::rank() == 0) {
        return container<value_t, base_t>{};

    } else {
        return container<value_t, base_t>{std::forward<extent_t>(exts)};
    }
}

template <typename in_t>
[[nodiscard]] inline constexpr auto make_container_like(in_t &&in) noexcept {
    const auto in_mds = to_const_mdspan(std::forward<in_t>(in));

    using value_t = typename decltype(in_mds)::value_type;

    return make_container<value_t>(in_mds.extents());
}

template <typename T>
using make_container_like_t = decltype(make_container_like(std::declval<T>()));

} // namespace core
} // namespace mdtensor
