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
    const auto x = md::arange<int>(6);

    EXPECT_TRUE(md::array_equal(
        md::full_like(x, 1),
        md::container<int, md::extents<std::size_t, 6>>{{1, 1, 1, 1, 1, 1}}));

    EXPECT_TRUE(md::array_equal(
        md::full_like(x, 0.1),
        md::container<int, md::extents<std::size_t, 6>>{{0, 0, 0, 0, 0, 0}}));

    EXPECT_TRUE(
        md::array_equal(md::full_like<double>(x, 0.1),
                        md::container<double, md::extents<std::size_t, 6>>{
                            {0.1, 0.1, 0.1, 0.1, 0.1, 0.1}}));

    EXPECT_TRUE(md::array_equal(
        md::full_like<double>(x, std::numeric_limits<double>::quiet_NaN()),
        md::container<double, md::extents<std::size_t, 6>>{
            {std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::quiet_NaN()}},
        true));
}

TEST(run_time, 2) {
    using value_t = double;

    const auto y = md::arange<value_t>(6);
    const auto z = md::full_like(y, 0.1);

    EXPECT_TRUE(
        md::array_equal(z, md::container<value_t, md::extents<std::size_t, 6>>{
                               {0.1, 0.1, 0.1, 0.1, 0.1, 0.1}}));
}

TEST(run_time, 3) {
    using value_t = int;
    using index_t = std::size_t;

    const auto y = md::zeros<value_t>(md::extents<index_t, 2, 2, 3>{});
    const auto z = md::full_like(
        y, md::container<value_t, md::extents<index_t, 3>>{{0, 0, 255}});

    EXPECT_TRUE(md::array_equal(
        z, md::container<value_t, md::extents<index_t, 2, 2, 3>>{
               {0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255}}));
}

TEST(compile_time, 1) {
    constexpr auto x = md::arange<6, int>();

    static_assert(md::array_equal(
        md::full_like(x, 1),
        md::container<int, md::extents<std::size_t, 6>>{{1, 1, 1, 1, 1, 1}}));

    static_assert(md::array_equal(
        md::full_like(x, 0.1),
        md::container<int, md::extents<std::size_t, 6>>{{0, 0, 0, 0, 0, 0}}));

    static_assert(
        md::array_equal(md::full_like<double>(x, 0.1),
                        md::container<double, md::extents<std::size_t, 6>>{
                            {0.1, 0.1, 0.1, 0.1, 0.1, 0.1}}));

    static_assert(md::array_equal(
        md::full_like<double>(x, std::numeric_limits<double>::quiet_NaN()),
        md::container<double, md::extents<std::size_t, 6>>{
            {std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::quiet_NaN(),
             std::numeric_limits<double>::quiet_NaN()}},
        true));
}

TEST(compile_time, 2) {
    using value_t = double;

    constexpr auto y = md::arange<6, value_t>();
    constexpr auto z = md::full_like(y, 0.1);

    static_assert(
        md::array_equal(z, md::container<value_t, md::extents<std::size_t, 6>>{
                               {0.1, 0.1, 0.1, 0.1, 0.1, 0.1}}));
}

TEST(compile_time, 3) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto y = md::zeros<value_t>(md::extents<index_t, 2, 2, 3>{});
    constexpr auto z = md::full_like(
        y, md::container<value_t, md::extents<index_t, 3>>{{0, 0, 255}});

    static_assert(md::array_equal(
        z, md::container<value_t, md::extents<index_t, 2, 2, 3>>{
               {0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0, 255}}));
}
