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

TEST(broadcast_to, 1) {
    using value_t = int;
    using index_t = size_t;

    constexpr auto a =
        md::core::container<value_t, stdex::extents<index_t, 3>>{{1, 2, 3}};
    constexpr auto b =
        md::core::container<value_t, stdex::extents<index_t, 3, 3>>{
            {1, 2, 3, 1, 2, 3, 1, 2, 3}};

    static_assert([&] {
        const auto c =
            md::core::broadcast_to(a, stdex::extents<index_t, 3, 3>{});

        if (!md::core::same_extents(b.extents(), c.extents())) {
            return false;
        }

        for (size_t i = 0; i < b.extent(0); i++) {
            for (size_t j = 0; j < b.extent(1); j++) {
                if (b(i, j) != c(i, j)) {
                    return false;
                }
            }
        }
        return true;
    }());
}
