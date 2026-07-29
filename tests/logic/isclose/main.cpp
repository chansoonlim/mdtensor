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
    using value_t = double;
    using index_t = std::size_t;

    const auto a =
        md::container<value_t, md::dims<1>>{{1e10, 1e-7}, md::dims<1>{2}};
    const auto b =
        md::container<value_t, md::dims<1>>{{1.00001e10, 1e-8}, md::dims<1>{2}};
    const auto c = md::isclose(a, b);

    EXPECT_TRUE(md::array_equal(
        c, md::container<bool, md::extents<index_t, 2>>{{true, false}}));
}

TEST(run_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    const auto a =
        md::container<value_t, md::dims<1>>{{1e10, 1e-8}, md::dims<1>{2}};
    const auto b =
        md::container<value_t, md::dims<1>>{{1.00001e10, 1e-9}, md::dims<1>{2}};
    const auto c = md::isclose(a, b);

    EXPECT_TRUE(md::array_equal(
        c, md::container<bool, md::extents<index_t, 2>>{{true, true}}));
}

TEST(run_time, 3) {
    using value_t = double;
    using index_t = std::size_t;

    const auto a =
        md::container<value_t, md::dims<1>>{{1e10, 1e-8}, md::dims<1>{2}};
    const auto b =
        md::container<value_t, md::dims<1>>{{1.0001e10, 1e-9}, md::dims<1>{2}};
    const auto c = md::isclose(a, b);

    EXPECT_TRUE(md::array_equal(
        c, md::container<bool, md::extents<index_t, 2>>{{false, true}}));
}

TEST(run_time, 4) {
    using value_t = double;
    using index_t = std::size_t;

    const auto a = md::container<value_t, md::dims<1>>{
        {1, std::numeric_limits<value_t>::quiet_NaN()}, md::dims<1>{2}};
    const auto b = md::container<value_t, md::dims<1>>{
        {1, std::numeric_limits<value_t>::quiet_NaN()}, md::dims<1>{2}};
    const auto c = md::isclose(a, b);

    EXPECT_TRUE(md::array_equal(
        c, md::container<bool, md::extents<index_t, 2>>{{true, false}}));
}

TEST(run_time, 5) {
    using value_t = double;
    using index_t = std::size_t;

    const auto a = md::container<value_t, md::dims<1>>{
        {1, std::numeric_limits<value_t>::quiet_NaN()}, md::dims<1>{2}};
    const auto b = md::container<value_t, md::dims<1>>{
        {1, std::numeric_limits<value_t>::quiet_NaN()}, md::dims<1>{2}};
    const auto c = md::isclose(a, b, 1e-05, 1e-08, std::nullopt, true);

    EXPECT_TRUE(md::array_equal(
        c, md::container<bool, md::extents<index_t, 2>>{{true, true}}));
}

TEST(run_time, 6) {
    using value_t = double;
    using index_t = std::size_t;

    const auto a =
        md::container<value_t, md::dims<1>>{{1e-8, 1e-7}, md::dims<1>{2}};
    const auto b = md::container<value_t, md::dims<1>>{{0, 0}, md::dims<1>{2}};
    const auto c = md::isclose(a, b);

    EXPECT_TRUE(md::array_equal(
        c, md::container<bool, md::extents<index_t, 2>>{{true, false}}));
}

TEST(run_time, 7) {
    using value_t = double;
    using index_t = std::size_t;

    const auto a =
        md::container<value_t, md::dims<1>>{{1e-100, 1e-7}, md::dims<1>{2}};
    const auto b = md::container<value_t, md::dims<1>>{{0, 0}, md::dims<1>{2}};
    const auto c = md::isclose(a, b, 1e-05, 0);

    EXPECT_TRUE(md::array_equal(
        c, md::container<bool, md::extents<index_t, 2>>{{false, false}}));
}

TEST(run_time, 8) {
    using value_t = double;
    using index_t = std::size_t;

    const auto a =
        md::container<value_t, md::dims<1>>{{1e-10, 1e-10}, md::dims<1>{2}};
    const auto b =
        md::container<value_t, md::dims<1>>{{1e-20, 0}, md::dims<1>{2}};
    const auto c = md::isclose(a, b);

    EXPECT_TRUE(md::array_equal(
        c, md::container<bool, md::extents<index_t, 2>>{{true, true}}));
}

TEST(run_time, 9) {
    using value_t = double;
    using index_t = std::size_t;

    const auto a =
        md::container<value_t, md::dims<1>>{{1e-10, 1e-10}, md::dims<1>{2}};
    const auto b = md::container<value_t, md::dims<1>>{{1e-20, 0.999999e-10},
                                                       md::dims<1>{2}};
    const auto c = md::isclose(a, b, 1e-05, 0);

    EXPECT_TRUE(md::array_equal(
        c, md::container<bool, md::extents<index_t, 2>>{{false, true}}));
}

TEST(compile_time, 1) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a =
        md::container<value_t, md::extents<index_t, 2>>{{1e10, 1e-7}};
    constexpr auto b =
        md::container<value_t, md::extents<index_t, 2>>{{1.00001e10, 1e-8}};
    constexpr auto c = md::isclose(a, b);

    static_assert(md::array_equal(
        c, md::container<bool, md::extents<index_t, 2>>{{true, false}}));
}

TEST(compile_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a =
        md::container<value_t, md::extents<index_t, 2>>{{1e10, 1e-8}};
    constexpr auto b =
        md::container<value_t, md::extents<index_t, 2>>{{1.00001e10, 1e-9}};
    constexpr auto c = md::isclose(a, b);

    static_assert(md::array_equal(
        c, md::container<bool, md::extents<index_t, 2>>{{true, true}}));
}

TEST(compile_time, 3) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a =
        md::container<value_t, md::extents<index_t, 2>>{{1e10, 1e-8}};
    constexpr auto b =
        md::container<value_t, md::extents<index_t, 2>>{{1.0001e10, 1e-9}};
    constexpr auto c = md::isclose(a, b);

    static_assert(md::array_equal(
        c, md::container<bool, md::extents<index_t, 2>>{{false, true}}));
}

TEST(compile_time, 4) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a = md::container<value_t, md::extents<index_t, 2>>{
        {1, std::numeric_limits<value_t>::quiet_NaN()}};
    constexpr auto b = md::container<value_t, md::extents<index_t, 2>>{
        {1, std::numeric_limits<value_t>::quiet_NaN()}};
    constexpr auto c = md::isclose(a, b);

    static_assert(md::array_equal(
        c, md::container<bool, md::extents<index_t, 2>>{{true, false}}));
}

TEST(compile_time, 5) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a = md::container<value_t, md::extents<index_t, 2>>{
        {1, std::numeric_limits<value_t>::quiet_NaN()}};
    constexpr auto b = md::container<value_t, md::extents<index_t, 2>>{
        {1, std::numeric_limits<value_t>::quiet_NaN()}};
    constexpr auto c = md::isclose(a, b, 1e-05, 1e-08, std::nullopt, true);

    static_assert(md::array_equal(
        c, md::container<bool, md::extents<index_t, 2>>{{true, true}}));
}

TEST(compile_time, 6) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a =
        md::container<value_t, md::extents<index_t, 2>>{{1e-8, 1e-7}};
    constexpr auto b = md::container<value_t, md::extents<index_t, 2>>{{0, 0}};
    constexpr auto c = md::isclose(a, b);

    static_assert(md::array_equal(
        c, md::container<bool, md::extents<index_t, 2>>{{true, false}}));
}

TEST(compile_time, 7) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a =
        md::container<value_t, md::extents<index_t, 2>>{{1e-100, 1e-7}};
    constexpr auto b = md::container<value_t, md::extents<index_t, 2>>{{0, 0}};
    constexpr auto c = md::isclose(a, b, 1e-05, 0);

    static_assert(md::array_equal(
        c, md::container<bool, md::extents<index_t, 2>>{{false, false}}));
}

TEST(compile_time, 8) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a =
        md::container<value_t, md::extents<index_t, 2>>{{1e-10, 1e-10}};
    constexpr auto b =
        md::container<value_t, md::extents<index_t, 2>>{{1e-20, 0}};
    constexpr auto c = md::isclose(a, b);

    static_assert(md::array_equal(
        c, md::container<bool, md::extents<index_t, 2>>{{true, true}}));
}

TEST(compile_time, 9) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a =
        md::container<value_t, md::extents<index_t, 2>>{{1e-10, 1e-10}};
    constexpr auto b =
        md::container<value_t, md::extents<index_t, 2>>{{1e-20, 0.999999e-10}};
    constexpr auto c = md::isclose(a, b, 1e-05, 0);

    static_assert(md::array_equal(
        c, md::container<bool, md::extents<index_t, 2>>{{false, true}}));
}
