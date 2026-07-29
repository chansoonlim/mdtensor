/**
 * @file
 * @brief Tests
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#include <gtest/gtest.h>

#include "mdtensor/core/broadcast.hpp"
#include "mdtensor/core/container.hpp"
#include "mdtensor/core/extents.hpp"

namespace stdex = std::experimental;
namespace md = mdtensor;

TEST(broadcast, 1) {
    using value_t = int;
    using index_t = size_t;

    constexpr auto x =
        md::core::container<value_t, stdex::extents<index_t, 1, 3>>{{1, 2, 3}};
    constexpr auto y =
        md::core::container<value_t, stdex::extents<index_t, 2, 1>>{{4, 5}};

    constexpr auto x_bcast_ref =
        md::core::container<value_t, stdex::extents<index_t, 2, 3>>{
            {1, 2, 3, 1, 2, 3}};
    constexpr auto y_bcast_ref =
        md::core::container<value_t, stdex::extents<index_t, 2, 3>>{
            {4, 4, 4, 5, 5, 5}};

    static_assert([&]() {
        const auto out = md::core::broadcast(
            std::index_sequence<0, 0>{},
            std::integer_sequence<bool, false, false>{}, x, y);
        const auto x_bcast = std::get<0>(std::get<0>(out));
        const auto y_bcast = std::get<1>(std::get<0>(out));

        if (!md::core::same_extents(x_bcast.extents(), x_bcast_ref.extents())) {
            return false;
        }
        if (!md::core::same_extents(y_bcast.extents(), y_bcast_ref.extents())) {
            return false;
        }

        for (size_t i = 0; i < x_bcast.extent(0); i++) {
            for (size_t j = 0; j < x_bcast.extent(1); j++) {
                if (x_bcast(i, j) != x_bcast_ref(i, j)) {
                    return false;
                }
            }
        }

        for (size_t i = 0; i < y_bcast.extent(0); i++) {
            for (size_t j = 0; j < y_bcast.extent(1); j++) {
                if (y_bcast(i, j) != y_bcast_ref(i, j)) {
                    return false;
                }
            }
        }
        return true;
    }());
}

TEST(broadcast, 2) {
    using value_t = int;
    using index_t = size_t;

    constexpr auto a =
        md::core::container<value_t, stdex::extents<index_t, 2, 1, 2>>{};
    constexpr auto b =
        md::core::container<value_t, stdex::extents<index_t, 2, 1>>{};
    auto c = md::core::container<value_t, stdex::extents<index_t, 2, 2, 2>>{};

    const auto out = md::core::broadcast(
        std::index_sequence<0, 0, 0>{},
        std::integer_sequence<bool, false, false, true>{}, a, b, c);

    static_assert(std::get<0>(std::get<0>(out)).rank() == 3);
    static_assert(std::get<1>(std::get<0>(out)).rank() == 3);
    static_assert(std::get<2>(std::get<0>(out)).rank() == 3);
}
