/**
 * @file
 * @brief Broadcast_arrays utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "broadcast_to.hpp"

namespace mdtensor {

template <typename... ins_t>
[[nodiscard]] inline constexpr auto broadcast_arrays(ins_t &&...ins) noexcept {
    if constexpr (sizeof...(ins_t) == 0) {
        return core::to_const_mdspan(std::forward<ins_t>(ins)...);

    } else {
        const auto ins_mds = [&]<size_t... Is>(std::index_sequence<Is...>) {
            return std::make_tuple(
                core::to_const_mdspan(std::forward<ins_t>(ins))...);
        }(std::make_index_sequence<sizeof...(ins_t)>{});

        const auto bexts = [&]<size_t... Is>(std::index_sequence<Is...>) {
            return core::broadcast_extents(std::get<Is>(ins_mds).extents()...);
        }(std::make_index_sequence<sizeof...(ins_t)>{});

        return [&]<size_t... Is>(std::index_sequence<Is...>) {
            return std::make_tuple(
                broadcast_to(std::get<Is>(ins_mds), bexts)...);
        }(std::make_index_sequence<sizeof...(ins_t)>{});
    }
}

} // namespace mdtensor
