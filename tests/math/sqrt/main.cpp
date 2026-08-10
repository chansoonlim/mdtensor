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
        md::container<value_t, md::dims<1>>{{1, 4, 9}, md::dims<1>{3}};
    const auto x_sqrt = md::sqrt(x);

    std::cout << "x_sqrt: " << md::to_string(x_sqrt) << std::endl;

    ASSERT_TRUE(md::allclose(
        x_sqrt, md::container<value_t, md::extents<index_t, 3>>{{1, 2, 3}}));
}

TEST(run_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    const auto x = md::container<value_t, md::dims<1>>{
        {4, -1, std::numeric_limits<value_t>::infinity()}, md::dims<1>{3}};
    const auto x_sqrt = md::sqrt(x);

    std::cout << "x_sqrt: " << md::to_string(x_sqrt) << std::endl;

    ASSERT_TRUE(md::allclose(x_sqrt,
                             md::container<value_t, md::extents<index_t, 3>>{
                                 {2, std::numeric_limits<value_t>::quiet_NaN(),
                                  std::numeric_limits<value_t>::infinity()}},
                             1e-05, 1e-08, true));
}

TEST(compile_time, 1) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto x =
        md::container<value_t, md::extents<index_t, 3>>{{1, 4, 9}};
    constexpr auto x_sqrt = md::sqrt(x);

    std::cout << "x_sqrt: " << md::to_string(x_sqrt) << std::endl;

    static_assert(md::allclose(
        x_sqrt, md::container<value_t, md::extents<index_t, 3>>{{1, 2, 3}}));
}

TEST(compile_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto x = md::container<value_t, md::extents<index_t, 3>>{
        {4, -1, std::numeric_limits<value_t>::infinity()}};
    constexpr auto x_sqrt = md::sqrt(x);

    std::cout << "x_sqrt: " << md::to_string(x_sqrt) << std::endl;

    static_assert(
        md::allclose(x_sqrt,
                     md::container<value_t, md::extents<index_t, 3>>{
                         {2, std::numeric_limits<value_t>::quiet_NaN(),
                          std::numeric_limits<value_t>::infinity()}},
                     1e-05, 1e-08, true));
}
