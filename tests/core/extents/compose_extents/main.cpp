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

TEST(compose_extents, 1) {
    constexpr auto exts1 = md::core::extents<std::size_t, 1, 2>{};
    constexpr auto exts2 = md::core::extents<std::size_t, 3, 4>{};
    constexpr auto exts = md::core::extents<std::size_t, 1, 2, 3, 4>{};

    static_assert(md::core::compose_extents(exts1, exts2) == exts);
}
