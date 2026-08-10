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

TEST(run_time, 1) {
    using value_t = int;

    const auto a = md::container<value_t, md::dims<1>>{{1, 2}, md::dims<1>{2}};
    const auto b = md::container<value_t, md::dims<1>>{{1, 2}, md::dims<1>{2}};

    EXPECT_TRUE(md::array_equal(a, b));
}

TEST(run_time, 2) {
    using value_t = int;

    const auto a = md::container<value_t, md::dims<1>>{{1, 2}, md::dims<1>{2}};
    const auto b =
        md::container<value_t, md::dims<1>>{{1, 2, 3}, md::dims<1>{3}};

    EXPECT_FALSE(md::array_equal(a, b));
}

TEST(run_time, 3) {
    using value_t = int;

    const auto a = md::container<value_t, md::dims<1>>{{1, 2}, md::dims<1>{2}};
    const auto b = md::container<value_t, md::dims<1>>{{1, 4}, md::dims<1>{2}};

    EXPECT_FALSE(md::array_equal(a, b));
}

TEST(run_time, 4) {
    using value_t = double;

    const auto a = md::container<value_t, md::dims<1>>{
        {1, std::numeric_limits<value_t>::quiet_NaN()}, md::dims<1>{2}};

    EXPECT_FALSE(md::array_equal(a, a));
}

TEST(run_time, 5) {
    using value_t = double;

    const auto a = md::container<value_t, md::dims<1>>{
        {1, std::numeric_limits<value_t>::quiet_NaN()}, md::dims<1>{2}};

    EXPECT_TRUE(md::array_equal(a, a, true));
}

TEST(compile_time, 1) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a = md::container<value_t, md::extents<index_t, 2>>{{1, 2}};
    constexpr auto b = md::container<value_t, md::extents<index_t, 2>>{{1, 2}};

    static_assert(md::array_equal(a, b));
}

TEST(compile_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a = md::container<value_t, md::extents<index_t, 2>>{{1, 2}};
    constexpr auto b =
        md::container<value_t, md::extents<index_t, 3>>{{1, 2, 3}};

    static_assert(!md::array_equal(a, b));
}

TEST(compile_time, 3) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a = md::container<value_t, md::extents<index_t, 2>>{{1, 2}};
    constexpr auto b = md::container<value_t, md::extents<index_t, 2>>{{1, 4}};

    static_assert(!md::array_equal(a, b));
}

TEST(compile_time, 4) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a = md::container<value_t, md::extents<index_t, 2>>{
        {1, std::numeric_limits<value_t>::quiet_NaN()}};

    static_assert(!md::array_equal(a, a));
}

TEST(compile_time, 5) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a = md::container<value_t, md::extents<index_t, 2>>{
        {1, std::numeric_limits<value_t>::quiet_NaN()}};

    static_assert(md::array_equal(a, a, true));
}
