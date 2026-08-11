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

    const auto out = md::eye<value_t>(2);

    EXPECT_TRUE(md::array_equal(
        out, md::tensor<value_t, md::extents<index_t, 2, 2>>{{1, 0, 0, 1}}));
}

TEST(run_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    const auto out = md::eye<value_t>(3, 1);

    EXPECT_TRUE(
        md::array_equal(out, md::tensor<value_t, md::extents<index_t, 3, 3>>{
                                 {0, 1, 0, 0, 0, 1, 0, 0, 0}}));
}

TEST(compile_time, 1) {
    using value_t = int;
    using index_t = std::size_t;

    const auto out = md::eye<2, value_t>();

    EXPECT_TRUE(md::array_equal(
        out, md::tensor<value_t, md::extents<index_t, 2, 2>>{{1, 0, 0, 1}}));
}

TEST(compile_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    const auto out = md::eye<3, value_t>(1);

    EXPECT_TRUE(
        md::array_equal(out, md::tensor<value_t, md::extents<index_t, 3, 3>>{
                                 {0, 1, 0, 0, 0, 1, 0, 0, 0}}));
}
