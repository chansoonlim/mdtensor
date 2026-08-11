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

    const auto a = md::tensor<value_t, md::dims<1>>{{2, 3, 4}, md::dims<1>{3}};
    const auto b = md::tensor<value_t, md::dims<1>>{{1, 5, 2}, md::dims<1>{3}};

    const auto c = md::minimum(a, b);

    std::cout << "c: " << md::to_string(c) << std::endl;

    ASSERT_TRUE(md::array_equal(
        c, md::tensor<value_t, md::extents<index_t, 3>>{{1, 3, 2}}));
}

TEST(run_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    const auto a = md::eye(2);
    const auto b = md::tensor<value_t, md::dims<1>>{{0.5, 2}, md::dims<1>{2}};

    const auto c = md::minimum(a, b);

    std::cout << "c: " << md::to_string(c) << std::endl;

    ASSERT_TRUE(md::array_equal(
        c, md::tensor<value_t, md::extents<index_t, 2, 2>>{{0.5, 0, 0, 1}}));
}

TEST(run_time, 3) {
    using value_t = double;
    using index_t = std::size_t;

    const auto a = md::tensor<value_t, md::dims<1>>{
        {std::numeric_limits<value_t>::quiet_NaN(), 0,
         std::numeric_limits<value_t>::quiet_NaN()},
        md::dims<1>{3}};
    const auto b = md::tensor<value_t, md::dims<1>>{
        {0, std::numeric_limits<value_t>::quiet_NaN(),
         std::numeric_limits<value_t>::quiet_NaN()},
        md::dims<1>{3}};

    const auto c = md::minimum(a, b);

    std::cout << "c: " << md::to_string(c) << std::endl;

    ASSERT_TRUE(
        md::array_equal(c,
                        md::tensor<value_t, md::extents<index_t, 3>>{
                            {std::numeric_limits<value_t>::quiet_NaN(),
                             std::numeric_limits<value_t>::quiet_NaN(),
                             std::numeric_limits<value_t>::quiet_NaN()}},
                        true));
}

TEST(run_time, 4) {
    using value_t = double;

    ASSERT_EQ(md::minimum(-std::numeric_limits<value_t>::infinity(), 1),
              -std::numeric_limits<value_t>::infinity());
}

TEST(compile_time, 1) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a = md::tensor<value_t, md::extents<index_t, 3>>{{2, 3, 4}};
    constexpr auto b = md::tensor<value_t, md::extents<index_t, 3>>{{1, 5, 2}};

    constexpr auto c = md::minimum(a, b);

    std::cout << "c: " << md::to_string(c) << std::endl;

    static_assert(md::array_equal(
        c, md::tensor<value_t, md::extents<index_t, 3>>{{1, 3, 2}}));
}

TEST(compile_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a = md::eye<2>();
    constexpr auto b = md::tensor<value_t, md::extents<index_t, 2>>{{0.5, 2}};

    constexpr auto c = md::minimum(a, b);

    std::cout << "c: " << md::to_string(c) << std::endl;

    static_assert(md::array_equal(
        c, md::tensor<value_t, md::extents<index_t, 2, 2>>{{0.5, 0, 0, 1}}));
}

TEST(compile_time, 3) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a = md::tensor<value_t, md::extents<index_t, 3>>{
        {std::numeric_limits<value_t>::quiet_NaN(), 0,
         std::numeric_limits<value_t>::quiet_NaN()}};
    constexpr auto b = md::tensor<value_t, md::extents<index_t, 3>>{
        {0, std::numeric_limits<value_t>::quiet_NaN(),
         std::numeric_limits<value_t>::quiet_NaN()}};

    constexpr auto c = md::minimum(a, b);

    std::cout << "c: " << md::to_string(c) << std::endl;

    static_assert(
        md::array_equal(c,
                        md::tensor<value_t, md::extents<index_t, 3>>{
                            {std::numeric_limits<value_t>::quiet_NaN(),
                             std::numeric_limits<value_t>::quiet_NaN(),
                             std::numeric_limits<value_t>::quiet_NaN()}},
                        true));
}

TEST(compile_time, 4) {
    using value_t = double;

    static_assert(md::minimum(-std::numeric_limits<value_t>::infinity(), 1) ==
                  -std::numeric_limits<value_t>::infinity());
}
