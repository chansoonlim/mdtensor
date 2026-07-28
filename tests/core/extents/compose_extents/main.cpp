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

TEST(compose_extents, 1) {
    constexpr auto exts1 = stdex::extents<size_t, 1, 2>{};
    constexpr auto exts2 = stdex::extents<size_t, 3, 4>{};
    constexpr auto exts = stdex::extents<size_t, 1, 2, 3, 4>{};

    static_assert(md::core::compose_extents(exts1, exts2) == exts);
}
