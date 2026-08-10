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
    const auto a_min = md::min(a);

    std::cout << "a_min: " << a_min << std::endl;

    ASSERT_EQ(a_min, 0);
}

TEST(run_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    const auto a = md::reshape(md::arange(4), md::dims<2>{2, 2});

    const auto a_min1 = md::min<0>(a);
    const auto a_min2 = md::min<1>(a);
    const auto a_min3 = md::min<0>(
        a, std::nullopt, 10,
        md::container<bool, md::dims<1>>{{false, true}, md::dims<1>{2}});

    std::cout << "a_min1: " << md::to_string(a_min1) << std::endl;
    std::cout << "a_min2: " << md::to_string(a_min2) << std::endl;
    std::cout << "a_min3: " << md::to_string(a_min3) << std::endl;

    ASSERT_TRUE(md::array_equal(
        a_min1, md::container<value_t, md::extents<index_t, 2>>{{0, 1}}));
    ASSERT_TRUE(md::array_equal(
        a_min2, md::container<value_t, md::extents<index_t, 2>>{{0, 2}}));
    ASSERT_TRUE(md::array_equal(
        a_min3, md::container<value_t, md::extents<index_t, 2>>{{10, 1}}));
}

TEST(run_time, 3) {
    using value_t = double;

    auto b = md::arange<value_t>(5);
    b(2) = std::numeric_limits<value_t>::quiet_NaN();

    const auto b_min1 = md::min(b);
    const auto b_min2 =
        md::min(b, std::nullopt, 10, md::logical_not(md::isnan(b)));
    const auto b_min3 = md::nanmin(b);

    std::cout << "b_min1: " << md::to_string(b_min1) << std::endl;
    std::cout << "b_min2: " << md::to_string(b_min2) << std::endl;
    std::cout << "b_min3: " << md::to_string(b_min3) << std::endl;

    ASSERT_TRUE(md::array_equal(
        b_min1, std::numeric_limits<value_t>::quiet_NaN(), true));
    ASSERT_EQ(b_min2, 0);
    ASSERT_EQ(b_min3, 0);
}

TEST(run_time, 4) {
    using value_t = int;
    using index_t = std::size_t;

    const auto a =
        md::container<value_t, md::dims<2>>{{-50, 10}, md::dims<2>{2, 1}};

    const auto a_min = md::min<-1>(a, std::nullopt, 0);

    std::cout << "a_min: " << md::to_string(a_min) << std::endl;

    ASSERT_TRUE(md::array_equal(
        a_min, md::container<value_t, md::extents<index_t, 2>>{{-50, 0}}));
}

TEST(run_time, 5) { ASSERT_EQ(md::min(6, std::nullopt, 5), 5); }

TEST(compile_time, 1) {
    using index_t = std::size_t;

    constexpr auto a =
        md::reshape(md::arange<4>(), md::extents<index_t, 2, 2>{});
    constexpr auto a_min = md::min(a);

    std::cout << "a_min: " << a_min << std::endl;

    static_assert(a_min == 0);
}

TEST(compile_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a =
        md::reshape(md::arange<4>(), md::extents<index_t, 2, 2>{});

    constexpr auto a_min1 = md::min<0>(a);
    constexpr auto a_min2 = md::min<1>(a);
    constexpr auto a_min3 =
        md::min<0>(a, std::nullopt, 10,
                   md::container<bool, md::extents<index_t, 2>>{{false, true}});

    std::cout << "a_min1: " << md::to_string(a_min1) << std::endl;
    std::cout << "a_min2: " << md::to_string(a_min2) << std::endl;
    std::cout << "a_min3: " << md::to_string(a_min3) << std::endl;

    static_assert(md::array_equal(
        a_min1, md::container<value_t, md::extents<index_t, 2>>{{0, 1}}));
    static_assert(md::array_equal(
        a_min2, md::container<value_t, md::extents<index_t, 2>>{{0, 2}}));
    static_assert(md::array_equal(
        a_min3, md::container<value_t, md::extents<index_t, 2>>{{10, 1}}));
}

TEST(compile_time, 3) {
    using value_t = double;

    constexpr auto b = []() {
        auto temp = md::arange<5, value_t>();
        temp(2) = std::numeric_limits<value_t>::quiet_NaN();
        return temp;
    }();

    constexpr auto b_min1 = md::min(b);
    constexpr auto b_min2 =
        md::min(b, std::nullopt, 10, md::logical_not(md::isnan(b)));
    constexpr auto b_min3 = md::nanmin(b);

    std::cout << "b_min1: " << md::to_string(b_min1) << std::endl;
    std::cout << "b_min2: " << md::to_string(b_min2) << std::endl;
    std::cout << "b_min3: " << md::to_string(b_min3) << std::endl;

    static_assert(md::array_equal(
        b_min1, std::numeric_limits<value_t>::quiet_NaN(), true));
    static_assert(b_min2 == 0);
    static_assert(b_min3 == 0);
}

TEST(compile_time, 4) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a =
        md::container<value_t, md::extents<index_t, 2, 1>>{{-50, 10}};

    constexpr auto a_min = md::min<-1>(a, std::nullopt, 0);

    std::cout << "a_min: " << md::to_string(a_min) << std::endl;

    static_assert(md::array_equal(
        a_min, md::container<value_t, md::extents<index_t, 2>>{{-50, 0}}));
}

TEST(compile_time, 5) { static_assert(md::min(6, std::nullopt, 5) == 5); }
