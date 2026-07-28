/**
 * @file
 * @brief Broadcast utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/broadcast.hpp"

namespace mdtensor {

template <typename... ins_t>
[[nodiscard]] inline constexpr auto broadcast(ins_t &&...ins) noexcept {
    return std::get<0>([&]<size_t... Is>(std::index_sequence<Is...>) {
        return core::broadcast(
            std::index_sequence<((void)Is, 0)...>{},
            std::integer_sequence<bool, (void(Is), false)...>{},
            std::forward<ins_t>(ins)...);
    }(std::make_index_sequence<sizeof...(ins_t)>{}));
}

} // namespace mdtensor
