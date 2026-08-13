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
    const auto a = md::reshape(md::arange(4), md::dims<2>{2, 2});
    const auto a_max = md::max(a);

    std::cout << "a_max: " << a_max << std::endl;

    ASSERT_EQ(a_max, 3);
}

TEST(run_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    const auto a = md::reshape(md::arange(4), md::dims<2>{2, 2});

    const auto a_max1 = md::max(a, std::index_sequence<0>{});
    const auto a_max2 = md::max(a, std::index_sequence<1>{});
    const auto a_max3 =
        md::max(a, std::index_sequence<0>{}, std::nullopt, -1,
                md::tensor<bool, md::dims<1>>{{false, true}, md::dims<1>{2}});

    std::cout << "a_max1: " << md::to_string(a_max1) << std::endl;
    std::cout << "a_max2: " << md::to_string(a_max2) << std::endl;
    std::cout << "a_max3: " << md::to_string(a_max3) << std::endl;

    ASSERT_TRUE(md::array_equal(
        a_max1, md::tensor<value_t, md::extents<index_t, 2>>{{2, 3}}));
    ASSERT_TRUE(md::array_equal(
        a_max2, md::tensor<value_t, md::extents<index_t, 2>>{{1, 3}}));
    ASSERT_TRUE(md::array_equal(
        a_max3, md::tensor<value_t, md::extents<index_t, 2>>{{-1, 3}}));
}

TEST(run_time, 3) {
    using value_t = double;

    auto b = md::arange<value_t>(5);
    b(2) = std::numeric_limits<value_t>::quiet_NaN();

    const auto b_max1 = md::max(b);
    const auto b_max2 = md::max(b, -1, md::logical_not(md::isnan(b)));
    const auto b_max3 = md::nanmax(b);

    std::cout << "b_max1: " << md::to_string(b_max1) << std::endl;
    std::cout << "b_max2: " << md::to_string(b_max2) << std::endl;
    std::cout << "b_max3: " << md::to_string(b_max3) << std::endl;

    ASSERT_TRUE(md::array_equal(
        b_max1, std::numeric_limits<value_t>::quiet_NaN(), true));
    ASSERT_EQ(b_max2, 4);
    ASSERT_EQ(b_max3, 4);
}

TEST(run_time, 4) {
    using value_t = int;
    using index_t = std::size_t;

    const auto a =
        md::tensor<value_t, md::dims<2>>{{-50, 10}, md::dims<2>{2, 1}};

    const auto a_max =
        md::max(a, std::integer_sequence<int, -1>{}, std::nullopt, 0);

    std::cout << "a_max: " << md::to_string(a_max) << std::endl;

    ASSERT_TRUE(md::array_equal(
        a_max, md::tensor<value_t, md::extents<index_t, 2>>{{0, 10}}));
}

TEST(run_time, 5) { ASSERT_EQ(md::max(5, 6), 6); }

TEST(compile_time, 1) {
    using index_t = std::size_t;

    constexpr auto a =
        md::reshape(md::arange<4>(), md::extents<index_t, 2, 2>{});
    constexpr auto a_max = md::max(a);

    std::cout << "a_max: " << a_max << std::endl;

    static_assert(a_max == 3);
}

TEST(compile_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a =
        md::reshape(md::arange<4>(), md::extents<index_t, 2, 2>{});

    constexpr auto a_max1 = md::max(a, std::index_sequence<0>{});
    constexpr auto a_max2 = md::max(a, std::index_sequence<1>{});
    constexpr auto a_max3 =
        md::max(a, std::index_sequence<0>{}, std::nullopt, -1,
                md::tensor<bool, md::extents<index_t, 2>>{{false, true}});

    std::cout << "a_max1: " << md::to_string(a_max1) << std::endl;
    std::cout << "a_max2: " << md::to_string(a_max2) << std::endl;
    std::cout << "a_max3: " << md::to_string(a_max3) << std::endl;

    static_assert(md::array_equal(
        a_max1, md::tensor<value_t, md::extents<index_t, 2>>{{2, 3}}));
    static_assert(md::array_equal(
        a_max2, md::tensor<value_t, md::extents<index_t, 2>>{{1, 3}}));
    static_assert(md::array_equal(
        a_max3, md::tensor<value_t, md::extents<index_t, 2>>{{-1, 3}}));
}

TEST(compile_time, 3) {
    using value_t = double;

    constexpr auto b = []() {
        auto temp = md::arange<5, value_t>();
        temp(2) = std::numeric_limits<value_t>::quiet_NaN();
        return temp;
    }();

    constexpr auto b_max1 = md::max(b);
    constexpr auto b_max2 = md::max(b, -1, md::logical_not(md::isnan(b)));
    constexpr auto b_max3 = md::nanmax(b);

    std::cout << "b_max1: " << md::to_string(b_max1) << std::endl;
    std::cout << "b_max2: " << md::to_string(b_max2) << std::endl;
    std::cout << "b_max3: " << md::to_string(b_max3) << std::endl;

    static_assert(md::array_equal(
        b_max1, std::numeric_limits<value_t>::quiet_NaN(), true));
    static_assert(b_max2 == 4);
    static_assert(b_max3 == 4);
}

TEST(compile_time, 4) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a =
        md::tensor<value_t, md::extents<index_t, 2, 1>>{{-50, 10}};

    constexpr auto a_max =
        md::max(a, std::integer_sequence<int, -1>{}, std::nullopt, 0);

    std::cout << "a_max: " << md::to_string(a_max) << std::endl;

    static_assert(md::array_equal(
        a_max, md::tensor<value_t, md::extents<index_t, 2>>{{0, 10}}));
}

TEST(compile_time, 5) { static_assert(md::max(5, 6) == 6); }
