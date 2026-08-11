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
    using index_t = std::size_t;

    const auto a = md::tensor<value_t, md::extents<index_t, 4, 4>>{
        {6, 3, 4, 8, 3, 6, 5, 1, 4, 5, 10, 7, 8, 1, 7, 25}};

    const auto [l, valid] = md::linalg::cholesky(a, false);

    EXPECT_TRUE(md::allclose(
        l, md::tensor<value_t, md::extents<index_t, 4, 4>>{
               {2.449490, 0.000000, 0.000000, 0.000000, 1.224745, 2.121320,
                0.000000, 0.000000, 1.632993, 1.414214, 2.309401, 0.000000,
                3.265986, -1.414214, 1.587713, 3.132491}}));

    EXPECT_TRUE(valid);
}

TEST(compile_time, 1) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a = md::tensor<value_t, md::extents<index_t, 4, 4>>{
        {6, 3, 4, 8, 3, 6, 5, 1, 4, 5, 10, 7, 8, 1, 7, 25}};

    constexpr auto out = md::linalg::cholesky(a, false);
    constexpr auto l = std::get<0>(out);
    constexpr auto valid = std::get<1>(out);

    static_assert(md::allclose(
        l, md::tensor<value_t, md::extents<index_t, 4, 4>>{
               {2.449490, 0.000000, 0.000000, 0.000000, 1.224745, 2.121320,
                0.000000, 0.000000, 1.632993, 1.414214, 2.309401, 0.000000,
                3.265986, -1.414214, 1.587713, 3.132491}}));

    static_assert(valid);
}
