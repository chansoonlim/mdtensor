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
    using index_t = std::size_t;

    const auto x =
        md::core::container<value_t, md::core::extents<index_t, 2>>{{1, 2}};
    const auto y = md::core::expand_dims<0>(x);

    EXPECT_TRUE(md::array_equal(
        y, md::core::container<value_t, md::core::extents<index_t, 1, 2>>{
               {1, 2}}));
}

TEST(run_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    const auto x =
        md::core::container<value_t, md::core::extents<index_t, 2>>{{1, 2}};
    const auto y = md::core::expand_dims<1>(x);

    EXPECT_TRUE(md::array_equal(
        y, md::core::container<value_t, md::core::extents<index_t, 2, 1>>{
               {1, 2}}));
}

TEST(run_time, 3) {
    using value_t = int;
    using index_t = std::size_t;

    const auto x =
        md::core::container<value_t, md::core::extents<index_t, 2>>{{1, 2}};
    const auto y = md::core::expand_dims<0, 1>(x);

    EXPECT_TRUE(md::array_equal(
        y, md::core::container<value_t, md::core::extents<index_t, 1, 1, 2>>{
               {1, 2}}));
}

TEST(run_time, 4) {
    using value_t = int;
    using index_t = std::size_t;

    const auto x =
        md::core::container<value_t, md::core::extents<index_t, 2>>{{1, 2}};
    const auto y = md::core::expand_dims<2, 0>(x);

    EXPECT_TRUE(md::array_equal(
        y, md::core::container<value_t, md::core::extents<index_t, 1, 2, 1>>{
               {1, 2}}));
}

TEST(run_time, 5) {
    using value_t = int;
    using index_t = std::size_t;

    const auto x =
        md::core::container<value_t, md::core::extents<index_t, 2>>{{1, 2}};

    EXPECT_TRUE(md::array_equal(md::core::expand_dims<1>(x),
                                md::core::expand_dims<-1>(x)));
    EXPECT_TRUE(md::array_equal(md::core::expand_dims<0>(x),
                                md::core::expand_dims<-2>(x)));

    EXPECT_TRUE(md::array_equal(md::core::expand_dims<0, 2>(x),
                                md::core::expand_dims<2, 0>(x)));
    EXPECT_TRUE(md::array_equal(md::core::expand_dims<0, 2>(x),
                                md::core::expand_dims<0, -1>(x)));
}

TEST(run_time, 6) {
    using value_t = int;
    using index_t = std::size_t;

    const value_t x = 1;

    EXPECT_TRUE(md::array_equal(
        md::core::expand_dims<0>(x),
        md::core::container<value_t, md::core::extents<index_t, 1>>{{1}}));
}

TEST(compile_time, 1) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto x =
        md::core::container<value_t, md::core::extents<index_t, 2>>{{1, 2}};

    static_assert(md::array_equal(
        md::core::expand_dims<0>(x),
        md::core::container<value_t, md::core::extents<index_t, 1, 2>>{
            {1, 2}}));
}

TEST(compile_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto x =
        md::core::container<value_t, md::core::extents<index_t, 2>>{{1, 2}};

    static_assert(md::array_equal(
        md::core::expand_dims<1>(x),
        md::core::container<value_t, md::core::extents<index_t, 2, 1>>{
            {1, 2}}));
}

TEST(compile_time, 3) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto x =
        md::core::container<value_t, md::core::extents<index_t, 2>>{{1, 2}};

    static_assert(md::array_equal(
        md::core::expand_dims<0, 1>(x),
        md::core::container<value_t, md::core::extents<index_t, 1, 1, 2>>{
            {1, 2}}));
}

TEST(compile_time, 4) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto x =
        md::core::container<value_t, md::core::extents<index_t, 2>>{{1, 2}};

    static_assert(md::array_equal(
        md::core::expand_dims<2, 0>(x),
        md::core::container<value_t, md::core::extents<index_t, 1, 2, 1>>{
            {1, 2}}));
}

TEST(compile_time, 5) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto x =
        md::core::container<value_t, md::core::extents<index_t, 2>>{{1, 2}};

    static_assert(md::array_equal(md::core::expand_dims<1>(x),
                                  md::core::expand_dims<-1>(x)));
    static_assert(md::array_equal(md::core::expand_dims<0>(x),
                                  md::core::expand_dims<-2>(x)));

    static_assert(md::array_equal(md::core::expand_dims<0, 2>(x),
                                  md::core::expand_dims<2, 0>(x)));
    static_assert(md::array_equal(md::core::expand_dims<0, 2>(x),
                                  md::core::expand_dims<0, -1>(x)));
}

TEST(compile_time, 6) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr value_t x = 1;

    static_assert(md::array_equal(
        md::core::expand_dims<0>(x),
        md::core::container<value_t, md::core::extents<index_t, 1>>{{1}}));
}
