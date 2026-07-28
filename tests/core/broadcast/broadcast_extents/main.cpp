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

namespace stdex = std::experimental;
namespace md = mdtensor;

TEST(broadcast_extents, 1) {
    constexpr auto exts1 = stdex::extents<size_t, 8, 6, 2, 3, 1, 4>{};

    constexpr auto bexts = md::core::broadcast_extents(exts1);

    static_assert(bexts == exts1);
}

TEST(broadcast_extents, 2) {
    constexpr auto exts1 = stdex::extents<size_t, 8, 6, 2, 3, 1, 4>{};
    constexpr auto exts2 = stdex::extents<size_t, 1, 3, 5, 4>{};

    constexpr auto bexts = md::core::broadcast_extents(exts1, exts2);

    static_assert(bexts == stdex::extents<size_t, 8, 6, 2, 3, 5, 4>{});
}

TEST(broadcast_extents, 3) {
    constexpr auto exts1 = stdex::extents<size_t, 8, 6, 2, 3, 1, 4>{};
    constexpr auto exts2 = stdex::extents<size_t, 1, 3, 1, 4>{};
    constexpr auto exts3 = stdex::extents<size_t, 2, 1, 5, 1>{};

    constexpr auto bexts = md::core::broadcast_extents(exts1, exts2, exts3);

    static_assert(bexts == stdex::extents<size_t, 8, 6, 2, 3, 5, 4>{});
}
