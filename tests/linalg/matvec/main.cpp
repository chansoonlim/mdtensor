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

    const auto a = md::container<value_t, md::extents<index_t, 3, 3>>{
        {0, 1, 0, -1, 0, 0, 0, 0, 1}};
    const auto v = md::container<value_t, md::extents<index_t, 4, 3>>{
        {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 6, 8}};

    const auto c = md::linalg::matvec(a, v);

    EXPECT_TRUE(
        md::allclose(c, md::container<value_t, md::extents<index_t, 4, 3>>{
                            {0, -1, 0, 1, 0, 0, 0, 0, 1, 6, 0, 8}}));
}

TEST(compile_time, 1) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a = md::container<value_t, md::extents<index_t, 3, 3>>{
        {0, 1, 0, -1, 0, 0, 0, 0, 1}};
    constexpr auto v = md::container<value_t, md::extents<index_t, 4, 3>>{
        {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 6, 8}};

    constexpr auto c = md::linalg::matvec(a, v);

    static_assert(
        md::allclose(c, md::container<value_t, md::extents<index_t, 4, 3>>{
                            {0, -1, 0, 1, 0, 0, 0, 0, 1, 6, 0, 8}}));
}
