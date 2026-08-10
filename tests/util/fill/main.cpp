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

    auto a = md::container<value_t, md::dims<1>>{{1, 2}, md::dims<1>{2}};

    md::fill(a, 0);

    EXPECT_TRUE(md::array_equal(
        a, md::container<value_t, md::extents<index_t, 2>>{{0, 0}}));
}

TEST(run_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    auto a = md::empty<value_t>(2);

    md::fill(a, 1);

    EXPECT_TRUE(md::array_equal(
        a, md::container<value_t, md::extents<index_t, 2>>{{1, 1}}));
}

TEST(compile_time, 1) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a = [] {
        auto a = md::container<value_t, md::extents<index_t, 2>>{{1, 2}};
        md::fill(a, 0);
        return a;
    }();
}

TEST(compile_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a = [] {
        auto a = md::empty<value_t>(md::extents<index_t, 2>{});
        md::fill(a, 1);
        return a;
    }();
}
