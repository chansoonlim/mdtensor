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

    ASSERT_EQ(md::sin(std::numbers::pi_v<value_t> / value_t{2}), 1);
}

TEST(run_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    const auto x = md::deg2rad(md::container<value_t, md::dims<1>>{
        {0, 30, 45, 60, 90}, md::dims<1>{5}});
    const auto x_sin = md::sin(x);

    std::cout << "x_sin: " << md::to_string(x_sin) << std::endl;

    ASSERT_TRUE(
        md::allclose(x_sin, md::container<value_t, md::extents<index_t, 5>>{
                                {0, 0.5, 0.70710678, 0.8660254, 1}}));
}

#ifdef REAL_GCC

TEST(compile_time, 1) {
    using value_t = double;

    static_assert(md::sin(std::numbers::pi_v<value_t> / value_t{2}) == 1);
}

TEST(compile_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto x = md::deg2rad(
        md::container<value_t, md::extents<index_t, 5>>{{0, 30, 45, 60, 90}});
    constexpr auto x_sin = md::sin(x);

    std::cout << "x_sin: " << md::to_string(x_sin) << std::endl;

    static_assert(
        md::allclose(x_sin, md::container<value_t, md::extents<index_t, 5>>{
                                {0, 0.5, 0.70710678, 0.8660254, 1}}));
}

#endif
