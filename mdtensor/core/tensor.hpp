/**
 * @file
 * @brief Tensor utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "extents.hpp"
#include "mdspan.hpp"

namespace mdtensor::core {

using bool_value_t = std::int8_t;

template <typename value_t, extents_c extent_t>
using tensor = std::conditional_t<
    extent_t::rank() == 0, value_t,
    std::conditional_t<
        extent_t::rank_dynamic() == 0,
        core::mdarray<value_t, extent_t, stdex::layout_right,
                      std::array<value_t, core::extents_size(extent_t{})>>,
        std::conditional_t<
            std::is_same_v<value_t, bool>,
            core::mdarray<bool_value_t, extent_t, stdex::layout_right,
                          std::vector<bool_value_t>>,
            core::mdarray<value_t, extent_t, stdex::layout_right,
                          std::vector<value_t>>>>>;

template <typename value_t = double, extents_c exts_t>
[[nodiscard]] constexpr auto make_tensor(exts_t &&exts) {
    using base_t = std::remove_cvref_t<decltype(exts)>;

    if constexpr (base_t::rank() == 0) {
        return tensor<value_t, base_t>{};

    } else {
        return tensor<value_t, base_t>{std::forward<exts_t>(exts)};
    }
}

[[nodiscard]] constexpr auto make_tensor_like(auto &&in) {
    const auto in_mds = to_const_mdspan(std::forward<decltype(in)>(in));

    using value_t = typename decltype(in_mds)::value_type;

    return make_tensor<value_t>(in_mds.extents());
}

template <typename T>
using make_tensor_like_t = decltype(make_tensor_like(std::declval<T>()));

} // namespace mdtensor::core
