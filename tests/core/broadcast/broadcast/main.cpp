/**
 * @file
 * @brief test
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#include <gtest/gtest.h>

#ifdef MDTENSOR_SINGLE_HEADER_INCLUDE_GUARD_ // for single header include
#include "mdtensor.hpp"
#else
#include "mdtensor/mdtensor.hpp"
#endif

namespace md = mdtensor;

TEST(broadcast, 1) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto x =
        md::core::container<value_t, md::core::extents<index_t, 1, 3>>{
            {1, 2, 3}};
    constexpr auto y =
        md::core::container<value_t, md::core::extents<index_t, 2, 1>>{{4, 5}};

    constexpr auto x_bcast_ref =
        md::core::container<value_t, md::core::extents<index_t, 2, 3>>{
            {1, 2, 3, 1, 2, 3}};
    constexpr auto y_bcast_ref =
        md::core::container<value_t, md::core::extents<index_t, 2, 3>>{
            {4, 4, 4, 5, 5, 5}};

    static_assert([&]() {
        const auto out = md::core::broadcast(
            std::index_sequence<0, 0>{},
            std::integer_sequence<bool, true, true>{}, x, y);
        const auto x_bcast = std::get<0>(std::get<0>(out));
        const auto y_bcast = std::get<1>(std::get<0>(out));

        if (!md::core::is_same_extents(x_bcast.extents(),
                                       x_bcast_ref.extents())) {
            return false;
        }
        if (!md::core::is_same_extents(y_bcast.extents(),
                                       y_bcast_ref.extents())) {
            return false;
        }

        for (std::size_t i = 0; i < x_bcast.extent(0); i++) {
            for (std::size_t j = 0; j < x_bcast.extent(1); j++) {
                if (x_bcast(i, j) != x_bcast_ref(i, j)) {
                    return false;
                }
            }
        }

        for (std::size_t i = 0; i < y_bcast.extent(0); i++) {
            for (std::size_t j = 0; j < y_bcast.extent(1); j++) {
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
    using index_t = std::size_t;

    constexpr auto a =
        md::core::container<value_t, md::core::extents<index_t, 2, 1, 2>>{};
    constexpr auto b =
        md::core::container<value_t, md::core::extents<index_t, 2, 1>>{};
    auto c =
        md::core::container<value_t, md::core::extents<index_t, 2, 2, 2>>{};

    const auto out = md::core::broadcast(
        std::index_sequence<0, 0, 0>{},
        std::integer_sequence<bool, true, true, false>{}, a, b, c);

    static_assert(std::get<0>(std::get<0>(out)).rank() == 3);
    static_assert(std::get<1>(std::get<0>(out)).rank() == 3);
    static_assert(std::get<2>(std::get<0>(out)).rank() == 3);
}
