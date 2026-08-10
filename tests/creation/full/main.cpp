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
    using value_t = float;
    using index_t = std::size_t;

    const auto x =
        md::full(md::dims<2>{2, 2}, std::numeric_limits<value_t>::infinity());

    std::cout << "x: " << md::to_string(x) << std::endl;

    EXPECT_TRUE(
        md::array_equal(x, md::container<value_t, md::extents<index_t, 2, 2>>{
                               {std::numeric_limits<value_t>::infinity(),
                                std::numeric_limits<value_t>::infinity(),
                                std::numeric_limits<value_t>::infinity(),
                                std::numeric_limits<value_t>::infinity()}}));
}

TEST(run_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    const auto x = md::full(md::dims<2>{2, 2}, 10);

    std::cout << "x: " << md::to_string(x) << std::endl;

    EXPECT_TRUE(md::array_equal(
        x,
        md::container<value_t, md::extents<index_t, 2, 2>>{{10, 10, 10, 10}}));
}

TEST(run_time, 3) {
    using value_t = int;
    using index_t = std::size_t;

    const auto x =
        md::full(md::dims<2>{2, 2},
                 md::container<value_t, md::dims<1>>{{1, 2}, md::dims<1>{2}});

    std::cout << "x: " << md::to_string(x) << std::endl;

    EXPECT_TRUE(md::array_equal(
        x, md::container<value_t, md::extents<index_t, 2, 2>>{{1, 2, 1, 2}}));
}

TEST(compile_time, 1) {
    using value_t = float;
    using index_t = std::size_t;

    constexpr auto x = md::full(md::extents<index_t, 2, 2>{},
                                std::numeric_limits<value_t>::infinity());

    std::cout << "x: " << md::to_string(x) << std::endl;

    static_assert(
        md::array_equal(x, md::container<value_t, md::extents<index_t, 2, 2>>{
                               {std::numeric_limits<value_t>::infinity(),
                                std::numeric_limits<value_t>::infinity(),
                                std::numeric_limits<value_t>::infinity(),
                                std::numeric_limits<value_t>::infinity()}}));
}

TEST(compile_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto x = md::full(md::extents<index_t, 2, 2>{}, 10);

    std::cout << "x: " << md::to_string(x) << std::endl;

    static_assert(md::array_equal(
        x,
        md::container<value_t, md::extents<index_t, 2, 2>>{{10, 10, 10, 10}}));
}

TEST(compile_time, 3) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto x =
        md::full(md::extents<index_t, 2, 2>{},
                 md::container<value_t, md::extents<index_t, 2>>{{1, 2}});

    std::cout << "x: " << md::to_string(x) << std::endl;

    static_assert(md::array_equal(
        x, md::container<value_t, md::extents<index_t, 2, 2>>{{1, 2, 1, 2}}));
}
