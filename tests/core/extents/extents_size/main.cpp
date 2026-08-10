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

TEST(extents_size, 1) {
    using index_t = std::size_t;

    constexpr auto exts = md::core::extents<index_t, 2, 3, 4>{};

    static_assert(md::core::extents_size(exts) == 24);
}

TEST(extents_size, 2) {
    using index_t = std::size_t;

    constexpr auto exts = md::core::dextents<index_t, 3>{2, 3, 4};

    EXPECT_EQ(md::core::extents_size(exts), 24);
}
