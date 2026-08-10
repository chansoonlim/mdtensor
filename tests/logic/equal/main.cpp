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

    const auto a =
        md::container<value_t, md::dims<1>>{{0, 1, 3}, md::dims<1>{3}};
    const auto b = md::arange<value_t>(3);
    const auto c = md::equal(a, b);

    EXPECT_TRUE(md::array_equal(
        c, md::container<bool, md::extents<index_t, 3>>{{true, true, false}}));
}

TEST(run_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    const auto a = 1;
    const auto b = md::ones<value_t>(1);
    const auto c = md::equal(a, b);

    EXPECT_TRUE(md::array_equal(
        c, md::container<bool, md::extents<index_t, 1>>{{true}}));
}

TEST(compile_time, 1) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a =
        md::container<value_t, md::extents<index_t, 3>>{{0, 1, 3}};
    constexpr auto b = md::arange<3, value_t>();
    constexpr auto c = md::equal(a, b);

    static_assert(md::array_equal(
        c, md::container<bool, md::extents<index_t, 3>>{{true, true, false}}));
}

TEST(compile_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a = 1;
    constexpr auto b = md::ones<value_t>(md::extents<index_t, 1>{});
    constexpr auto c = md::equal(a, b);

    static_assert(md::array_equal(
        c, md::container<bool, md::extents<index_t, 1>>{{true}}));
}
