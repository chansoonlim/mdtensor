/**
 * @file
 * @brief Array equivalence utilities for mdtensor.
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#pragma once

#include "array_equal.hpp"

namespace mdtensor {

[[nodiscard]] constexpr bool array_equiv(auto &&in1, auto &&in2) {
    if constexpr (core::always_cannot_broadcast(
                      std::index_sequence<0, 0>{},
                      std::integer_sequence<bool, true, true>{},
                      std::type_identity<decltype(in1)>{},
                      std::type_identity<decltype(in2)>{})) {
        return false;

    } else {
        try {
            const auto [in1_bcast, in2_bcast] = std::get<0>(
                core::broadcast(std::index_sequence<0, 0>{},
                                std::integer_sequence<bool, true, true>{},
                                std::forward<decltype(in1)>(in1),
                                std::forward<decltype(in2)>(in2)));
            return array_equal(in1_bcast, in2_bcast);

        } catch (const std::exception &e) {
            return false;
        }
    }
}

} // namespace mdtensor
