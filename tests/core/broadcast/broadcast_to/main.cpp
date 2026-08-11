/**
 * @file
 * @brief test
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#include <gtest/gtest.h>

#ifdef MDTENSOR_USE_SINGLE_HEADER // for single header include
#include "mdtensor.hpp"
#else
#include "mdtensor/mdtensor.hpp"
#endif

namespace md = mdtensor;

TEST(broadcast_to, 1) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a =
        md::core::tensor<value_t, md::core::extents<index_t, 3>>{{1, 2, 3}};
    constexpr auto b =
        md::core::tensor<value_t, md::core::extents<index_t, 3, 3>>{
            {1, 2, 3, 1, 2, 3, 1, 2, 3}};

    static_assert([&] {
        const auto c =
            md::core::broadcast_to(a, md::core::extents<index_t, 3, 3>{});

        if (!md::core::is_same_extents(b.extents(), c.extents())) {
            return false;
        }

        for (std::size_t i = 0; i < b.extent(0); i++) {
            for (std::size_t j = 0; j < b.extent(1); j++) {
                if (b(i, j) != c(i, j)) {
                    return false;
                }
            }
        }
        return true;
    }());
}
