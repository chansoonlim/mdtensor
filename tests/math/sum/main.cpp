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
    using value_t = double;

    const auto a = md::tensor<value_t, md::dims<1>>{{0.5, 1.5}, md::dims<1>{2}};
    const auto b = md::sum(a);

    static_assert(std::is_same_v<std::remove_cvref_t<decltype(b)>, value_t>);

    EXPECT_EQ(b, 2);
}

TEST(run_time, 2) {
    using value_t = double;

    const auto a =
        md::tensor<value_t, md::dims<1>>{{0.5, 0.7, 0.2, 1.5}, md::dims<1>{4}};
    const auto b = md::sum<int>(a);

    static_assert(std::is_same_v<std::remove_cvref_t<decltype(b)>, int>);

    EXPECT_EQ(b, 1);
}

TEST(run_time, 3) {
    using value_t = int;

    const auto a =
        md::tensor<value_t, md::dims<1>>{{0, 1, 0, 5}, md::dims<1>{4}};
    const auto b = md::sum(a);

    static_assert(std::is_same_v<std::remove_cvref_t<decltype(b)>, value_t>);

    EXPECT_EQ(b, 6);
}

TEST(run_time, 4) {
    using value_t = int;
    using index_t = std::size_t;

    const auto a =
        md::tensor<value_t, md::dims<2>>{{0, 1, 0, 5}, md::dims<2>{2, 2}};
    const auto b = md::sum(a, std::index_sequence<0>{});

    static_assert(std::is_same_v<typename decltype(b)::value_type, value_t>);

    EXPECT_TRUE(md::array_equal(
        b, md::tensor<value_t, md::extents<index_t, 2>>{{0, 6}}));
}

TEST(run_time, 5) {
    using value_t = int;
    using index_t = std::size_t;

    const auto a =
        md::tensor<value_t, md::dims<2>>{{0, 1, 0, 5}, md::dims<2>{2, 2}};
    const auto b = md::sum(a, std::index_sequence<1>{});

    static_assert(std::is_same_v<typename decltype(b)::value_type, value_t>);

    EXPECT_TRUE(md::array_equal(
        b, md::tensor<value_t, md::extents<index_t, 2>>{{1, 5}}));
}

TEST(run_time, 6) {
    using value_t = double;
    using index_t = std::size_t;

    const auto a = md::tensor<value_t, md::dims<2>>{
        {0, 1, std::numeric_limits<value_t>::quiet_NaN(), 5},
        md::dims<2>{2, 2}};
    const auto b =
        md::sum(a, std::index_sequence<1>{}, std::nullopt, 0,
                md::tensor<bool, md::dims<1>>{{false, true}, md::dims<1>{2}});

    static_assert(std::is_same_v<typename decltype(b)::value_type, value_t>);

    EXPECT_TRUE(md::array_equal(
        b, md::tensor<value_t, md::extents<index_t, 2>>{{1, 5}}));
}

TEST(run_time, 7) {
    using value_t = std::int8_t;

    const auto a = md::ones<value_t>(128);
    const auto b = md::sum(a);

    static_assert(std::is_same_v<std::remove_cvref_t<decltype(b)>, value_t>);

    EXPECT_EQ(b, -128);
}

TEST(run_time, 8) {
    using value_t = int;

    const auto a = md::tensor<value_t, md::dims<1>>{{10}, md::dims<1>{1}};
    const auto b = md::sum(a, 5);

    static_assert(std::is_same_v<std::remove_cvref_t<decltype(b)>, value_t>);

    EXPECT_EQ(b, 15);
}

TEST(compile_time, 1) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a = md::tensor<value_t, md::extents<index_t, 2>>{{0.5, 1.5}};
    constexpr auto b = md::sum(a);

    static_assert(std::is_same_v<std::remove_cvref_t<decltype(b)>, value_t>);

    static_assert(b == 2);
}

TEST(compile_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a =
        md::tensor<value_t, md::extents<index_t, 4>>{{0.5, 0.7, 0.2, 1.5}};
    constexpr auto b = md::sum<int>(a);

    static_assert(std::is_same_v<std::remove_cvref_t<decltype(b)>, int>);

    static_assert(b == 1);
}

TEST(compile_time, 3) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a =
        md::tensor<value_t, md::extents<index_t, 4>>{{0, 1, 0, 5}};
    constexpr auto b = md::sum(a);

    static_assert(std::is_same_v<std::remove_cvref_t<decltype(b)>, value_t>);

    static_assert(b == 6);
}

TEST(compile_time, 4) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a =
        md::tensor<value_t, md::extents<index_t, 2, 2>>{{0, 1, 0, 5}};
    constexpr auto b = md::sum(a, std::index_sequence<0>{});

    static_assert(std::is_same_v<typename decltype(b)::value_type, value_t>);

    static_assert(md::array_equal(
        b, md::tensor<value_t, md::extents<index_t, 2>>{{0, 6}}));
}

TEST(compile_time, 5) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a =
        md::tensor<value_t, md::extents<index_t, 2, 2>>{{0, 1, 0, 5}};
    constexpr auto b = md::sum(a, std::index_sequence<1>{});

    static_assert(std::is_same_v<typename decltype(b)::value_type, value_t>);

    static_assert(md::array_equal(
        b, md::tensor<value_t, md::extents<index_t, 2>>{{1, 5}}));
}

TEST(compile_time, 6) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a = md::tensor<value_t, md::extents<index_t, 2, 2>>{
        {0, 1, std::numeric_limits<value_t>::quiet_NaN(), 5}};
    constexpr auto b =
        md::sum(a, std::index_sequence<1>{}, std::nullopt, 0,
                md::tensor<bool, md::extents<index_t, 2>>{{false, true}});

    static_assert(std::is_same_v<typename decltype(b)::value_type, value_t>);

    static_assert(md::array_equal(
        b, md::tensor<value_t, md::extents<index_t, 2>>{{1, 5}}));
}

TEST(compile_time, 7) {
    using value_t = std::int8_t;
    using index_t = std::size_t;

    constexpr auto a = md::ones<value_t>(md::extents<index_t, 128>{});
    constexpr auto b = md::sum(a);

    static_assert(std::is_same_v<std::remove_cvref_t<decltype(b)>, value_t>);

    static_assert(b == -128);
}

TEST(compile_time, 8) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a = md::tensor<value_t, md::extents<index_t, 1>>{{10}};
    constexpr auto b = md::sum(a, 5);

    static_assert(std::is_same_v<std::remove_cvref_t<decltype(b)>, value_t>);

    static_assert(b == 15);
}
