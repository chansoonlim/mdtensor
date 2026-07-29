/**
 * @file
 * @brief Broadcast utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "../core/core.hpp"

namespace mdtensor {

[[nodiscard]] constexpr auto broadcast(auto &&...ins) {
    return std::get<0>([&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return core::broadcast(
            std::index_sequence<((void)Is, 0)...>{},
            std::integer_sequence<bool, (void(Is), true)...>{},
            std::forward<decltype(ins)>(ins)...);
    }(std::make_index_sequence<sizeof...(ins)>{}));
}

} // namespace mdtensor
