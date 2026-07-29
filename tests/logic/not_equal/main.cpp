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

    const auto a = md::container<value_t, md::dims<1>>{{1, 2}, md::dims<1>{2}};
    const auto b = md::container<value_t, md::dims<1>>{{1, 3}, md::dims<1>{2}};
    const auto c = md::not_equal(a, b);

    EXPECT_TRUE(md::array_equal(
        c, md::container<bool, md::extents<index_t, 2>>{{false, true}}));
}

TEST(run_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    const auto a = md::container<value_t, md::dims<1>>{{1, 2}, md::dims<1>{2}};
    const auto b =
        md::container<value_t, md::dims<2>>{{1, 3, 1, 4}, md::dims<2>{2, 2}};
    const auto c = md::not_equal(a, b);

    EXPECT_TRUE(
        md::array_equal(c, md::container<bool, md::extents<index_t, 2, 2>>{
                               {false, true, false, true}}));
}

TEST(compile_time, 1) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a = md::container<value_t, md::extents<index_t, 2>>{{1, 2}};
    constexpr auto b = md::container<value_t, md::extents<index_t, 2>>{{1, 3}};
    constexpr auto c = md::not_equal(a, b);

    static_assert(md::array_equal(
        c, md::container<bool, md::extents<index_t, 2>>{{false, true}}));
}

TEST(compile_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a = md::container<value_t, md::extents<index_t, 2>>{{1, 2}};
    constexpr auto b =
        md::container<value_t, md::extents<index_t, 2, 2>>{{1, 3, 1, 4}};
    constexpr auto c = md::not_equal(a, b);

    static_assert(
        md::array_equal(c, md::container<bool, md::extents<index_t, 2, 2>>{
                               {false, true, false, true}}));
}
