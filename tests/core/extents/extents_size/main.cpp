/**
 * @file
 * @brief Tests
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#include <gtest/gtest.h>

#include "mdtensor/core/extents.hpp"

namespace stdex = std::experimental;
namespace md = mdtensor;

TEST(extents_size, 1) {
    using index_t = size_t;

    constexpr auto exts = stdex::extents<index_t, 2, 3, 4>{};

    static_assert(md::core::extents_size(exts) == 24);
}

TEST(extents_size, 2) {
    using index_t = size_t;

    constexpr auto exts = stdex::dextents<index_t, 3>{2, 3, 4};

    ASSERT_EQ(md::core::extents_size(exts), 24);
}
