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
    using value_t = int;
    using index_t = std::size_t;

    const auto a =
        md::container<value_t, md::dims<2>>{{1, 2, 3, 4}, md::dims<2>{2, 2}};
    const auto b =
        md::container<value_t, md::dims<2>>{{5, 6}, md::dims<2>{1, 2}};

    EXPECT_TRUE(
        md::array_equal(md::concatenate<0>(a, b),
                        md::container<value_t, md::extents<index_t, 3, 2>>{
                            {1, 2, 3, 4, 5, 6}}));

    EXPECT_TRUE(
        md::array_equal(md::concatenate<1>(a, md::transpose(b)),
                        md::container<value_t, md::extents<index_t, 2, 3>>{
                            {1, 2, 5, 3, 4, 6}}));

    EXPECT_TRUE(md::array_equal(
        md::concatenate(md::flatten(a), md::flatten(b)),
        md::container<value_t, md::extents<index_t, 6>>{{1, 2, 3, 4, 5, 6}}));
}

TEST(compile_time, 1) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a =
        md::container<value_t, md::extents<index_t, 2, 2>>{{1, 2, 3, 4}};
    constexpr auto b =
        md::container<value_t, md::extents<index_t, 1, 2>>{{5, 6}};

    static_assert(
        md::array_equal(md::concatenate<0>(a, b),
                        md::container<value_t, md::extents<index_t, 3, 2>>{
                            {1, 2, 3, 4, 5, 6}}));

    static_assert(
        md::array_equal(md::concatenate<1>(a, md::transpose(b)),
                        md::container<value_t, md::extents<index_t, 2, 3>>{
                            {1, 2, 5, 3, 4, 6}}));

    static_assert(md::array_equal(
        md::concatenate(md::flatten(a), md::flatten(b)),
        md::container<value_t, md::extents<index_t, 6>>{{1, 2, 3, 4, 5, 6}}));
}
