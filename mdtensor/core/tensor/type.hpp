/**
 * @file
 * @brief Tensor type utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../base/base.hpp"

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

} // namespace mdtensor::core
