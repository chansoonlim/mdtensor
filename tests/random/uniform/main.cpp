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

TEST(run_time, 1) {
    const auto out = md::random::uniform(1000, -1, 0);

    std::cout << md::to_string(out) << std::endl;

    EXPECT_TRUE(md::all(md::greater_equal(out, -1)));
    EXPECT_TRUE(md::all(md::less(out, 0)));
}

TEST(compile_time, 1) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto out =
        md::random::uniform<value_t, md::random::generator::SplitMix64>(
            md::extents<index_t, 1000>{}, -1, 0, std::nullopt,
            md::random::seed_t{0});

    std::cout << md::to_string(out) << std::endl;

    static_assert(md::all(md::greater_equal(out, -1)));
    static_assert(md::all(md::less(out, 0)));
}
