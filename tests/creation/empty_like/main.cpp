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

    const auto a = md::container<value_t, md::dims<2>>{{1, 2, 3, 4, 5, 6},
                                                       md::dims<2>{2, 3}};
    const auto b = md::empty_like(a);

    static_assert(std::is_same_v<decltype(a), decltype(b)>);

    EXPECT_TRUE(md::is_same_extents(a.extents(), b.extents()));
}

TEST(run_time, 2) {
    using value_t = double;

    const auto a = md::container<value_t, md::dims<2>>{{1, 2, 3, 4, 5, 6},
                                                       md::dims<2>{2, 3}};
    const auto b = md::empty_like(a);

    static_assert(std::is_same_v<decltype(a), decltype(b)>);

    EXPECT_TRUE(md::is_same_extents(a.extents(), b.extents()));
}

TEST(compile_time, 1) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a =
        md::container<value_t, md::extents<index_t, 2, 3>>{{1, 2, 3, 4, 5, 6}};
    constexpr auto b = md::empty_like(a);

    static_assert(std::is_same_v<decltype(a), decltype(b)>);

    static_assert(md::is_same_extents(a.extents(), b.extents()));
}

TEST(compile_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a =
        md::container<value_t, md::extents<index_t, 2, 3>>{{1, 2, 3, 4, 5, 6}};
    constexpr auto b = md::empty_like(a);

    static_assert(std::is_same_v<decltype(a), decltype(b)>);

    static_assert(md::is_same_extents(a.extents(), b.extents()));
}
