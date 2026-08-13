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
    using value_t = bool;

    const auto a = md::tensor<value_t, md::dims<2>>{{true, false, true, true},
                                                    md::dims<2>{2, 2}};

    EXPECT_TRUE(md::any(a));
}

TEST(run_time, 2) {
    using value_t = bool;
    using index_t = std::size_t;

    const auto a = md::tensor<value_t, md::dims<2>>{
        {true, false, true, false, false, false}, md::dims<2>{2, 3}};

    EXPECT_TRUE(md::array_equal(
        md::any(a, std::index_sequence<0>{}),
        md::tensor<value_t, md::extents<index_t, 3>>{{true, false, true}}));
}

TEST(run_time, 3) {
    using value_t = int;

    const auto a = md::tensor<value_t, md::dims<1>>{{-1, 0, 5}, md::dims<1>{3}};

    EXPECT_TRUE(md::any(a));
}

TEST(run_time, 4) {
    using value_t = double;

    const auto a = md::tensor<value_t, md::dims<2>>{
        {std::numeric_limits<value_t>::quiet_NaN(),
         std::numeric_limits<value_t>::infinity()},
        md::dims<2>{2, 1}};

    EXPECT_TRUE(md::array_equal(
        md::any<true>(a, std::index_sequence<1>{}),
        md::tensor<bool, md::dims<2>>{{true, true}, md::dims<2>{2, 1}}));
}

TEST(run_time, 5) {
    using value_t = bool;

    const auto a = md::tensor<value_t, md::dims<2>>{{true, false, false, false},
                                                    md::dims<2>{2, 2}};

    const auto where =
        md::tensor<bool, md::dims<2>>{{false, true}, md::dims<2>{2, 1}};

    EXPECT_FALSE(md::any(a, where));
}

TEST(run_time, 6) {
    using value_t = int;
    using index_t = std::size_t;

    const auto a = md::tensor<value_t, md::dims<2>>{{1, 0, 0, 0, 0, 1, 0, 0, 0},
                                                    md::dims<2>{3, 3}};

    EXPECT_TRUE(md::array_equal(
        md::any(a, std::index_sequence<0>{}),
        md::tensor<bool, md::extents<index_t, 3>>{{true, false, true}}));
    EXPECT_TRUE(md::array_equal(
        md::any(a, std::index_sequence<1>{}),
        md::tensor<bool, md::extents<index_t, 3>>{{true, true, false}}));
}

TEST(compile_time, 1) {
    using value_t = bool;
    using index_t = std::size_t;

    constexpr auto a = md::tensor<value_t, md::extents<index_t, 2, 2>>{
        {true, false, true, true}};

    static_assert(md::any(a));
}

TEST(compile_time, 2) {
    using value_t = bool;
    using index_t = std::size_t;

    constexpr auto a = md::tensor<value_t, md::extents<index_t, 2, 3>>{
        {true, false, true, false, false, false}};

    static_assert(md::array_equal(
        md::any(a, std::index_sequence<0>{}),
        md::tensor<value_t, md::extents<index_t, 3>>{{true, false, true}}));
}

TEST(compile_time, 3) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a = md::tensor<value_t, md::extents<index_t, 3>>{{-1, 0, 5}};

    static_assert(md::any(a));
}

TEST(compile_time, 4) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a = md::tensor<value_t, md::extents<index_t, 2, 1>>{
        {std::numeric_limits<value_t>::quiet_NaN(),
         std::numeric_limits<value_t>::infinity()}};

    std::cout << "a: " << md::to_string(a) << std::endl;
    std::cout << "any: "
              << md::to_string(md::any<true>(a, std::index_sequence<0>{}))
              << std::endl;

    static_assert(md::array_equal(
        md::any<true>(a, std::index_sequence<1>{}),
        md::tensor<bool, md::extents<index_t, 2, 1>>{{true, true}}));
}

TEST(compile_time, 5) {
    using value_t = bool;
    using index_t = std::size_t;

    constexpr auto a = md::tensor<value_t, md::extents<index_t, 2, 2>>{
        {true, false, false, false}};

    constexpr auto where =
        md::tensor<bool, md::extents<index_t, 2, 1>>{{false, true}};

    static_assert(!md::any(a, where));
}

TEST(compile_time, 6) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a = md::tensor<value_t, md::extents<index_t, 3, 3>>{
        {1, 0, 0, 0, 0, 1, 0, 0, 0}};

    static_assert(md::array_equal(
        md::any(a, std::index_sequence<0>{}),
        md::tensor<bool, md::extents<index_t, 3>>{{true, false, true}}));
    static_assert(md::array_equal(
        md::any(a, std::index_sequence<1>{}),
        md::tensor<bool, md::extents<index_t, 3>>{{true, true, false}}));
}
