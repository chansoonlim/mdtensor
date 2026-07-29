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
    using value_t = int;
    using index_t = std::size_t;

    const auto a = md::container<value_t, md::dims<1>>{{4, 2}, md::dims<1>{2}};
    const auto b = md::container<value_t, md::dims<1>>{{2, 2}, md::dims<1>{2}};
    const auto c = md::greater(a, b);

    EXPECT_TRUE(md::array_equal(
        c, md::container<bool, md::extents<index_t, 2>>{{true, false}}));
}

TEST(compile_time, 1) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a = md::container<value_t, md::extents<index_t, 2>>{{4, 2}};
    constexpr auto b = md::container<value_t, md::extents<index_t, 2>>{{2, 2}};
    constexpr auto c = md::greater(a, b);

    static_assert(md::array_equal(
        c, md::container<bool, md::extents<index_t, 2>>{{true, false}}));
}
