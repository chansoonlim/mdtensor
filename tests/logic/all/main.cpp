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
    using value_t = bool;

    const auto a = md::container<value_t, md::dims<2>>{
        {true, false, true, true}, md::dims<2>{2, 2}};

    EXPECT_FALSE(md::all(a));
}

TEST(run_time, 2) {
    using value_t = bool;
    using index_t = std::size_t;

    const auto a = md::container<value_t, md::dims<2>>{
        {true, false, true, true}, md::dims<2>{2, 2}};

    EXPECT_TRUE(md::array_equal(
        md::all<0>(a),
        md::container<value_t, md::extents<index_t, 2>>{{true, false}}));
}

TEST(run_time, 3) {
    using value_t = int;

    const auto a =
        md::container<value_t, md::dims<1>>{{-1, 4, 5}, md::dims<1>{3}};

    EXPECT_TRUE(md::all(a));
}

TEST(run_time, 4) {
    using value_t = double;

    const auto a = md::container<value_t, md::dims<1>>{
        {1, std::numeric_limits<value_t>::quiet_NaN()}, md::dims<1>{2}};

    EXPECT_TRUE(md::all(a));
}

TEST(compile_time, 1) {
    using value_t = bool;
    using index_t = std::size_t;

    constexpr auto a = md::container<value_t, md::extents<index_t, 2, 2>>{
        {true, false, true, true}};

    static_assert(!md::all(a));
}

TEST(compile_time, 2) {
    using value_t = bool;
    using index_t = std::size_t;

    constexpr auto a = md::container<value_t, md::extents<index_t, 2, 2>>{
        {true, false, true, true}};

    static_assert(md::array_equal(
        md::all<0>(a),
        md::container<value_t, md::extents<index_t, 2>>{{true, false}}));
}

TEST(compile_time, 3) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a =
        md::container<value_t, md::extents<index_t, 3>>{{-1, 4, 5}};

    static_assert(md::all(a));
}

TEST(compile_time, 4) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a = md::container<value_t, md::extents<index_t, 2>>{
        {1, std::numeric_limits<value_t>::quiet_NaN()}};

    static_assert(md::all(a));
}
