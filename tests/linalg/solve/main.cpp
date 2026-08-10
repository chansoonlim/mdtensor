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

    const auto a =
        md::container<value_t, md::extents<index_t, 2, 2>>{{1, 2, 3, 5}};
    const auto b = md::container<value_t, md::extents<index_t, 2>>{{1, 2}};
    const auto [x, valid] = md::linalg::solve(a, b);

    EXPECT_TRUE(md::allclose(
        x, md::container<value_t, md::extents<index_t, 2>>{{-1, 1}}));

    EXPECT_TRUE(md::allclose(md::linalg::matvec(a, x), b));
}

TEST(compile_time, 1) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a =
        md::container<value_t, md::extents<index_t, 2, 2>>{{1, 2, 3, 5}};
    constexpr auto b = md::container<value_t, md::extents<index_t, 2>>{{1, 2}};
    constexpr auto x = std::get<0>(md::linalg::solve(a, b));

    static_assert(md::allclose(
        x, md::container<value_t, md::extents<index_t, 2>>{{-1, 1}}));

    static_assert(md::allclose(md::linalg::matvec(a, x), b));
}
