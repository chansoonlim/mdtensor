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

    const auto a = md::container<value_t, md::dims<1>>{{1, 2}, md::dims<1>{2}};
    const auto b = md::container<value_t, md::dims<1>>{{1, 2}, md::dims<1>{2}};

    EXPECT_TRUE(md::array_equiv(a, b));
}

TEST(run_time, 2) {
    using value_t = int;

    const auto a = md::container<value_t, md::dims<1>>{{1, 2}, md::dims<1>{2}};
    const auto b = md::container<value_t, md::dims<1>>{{1, 3}, md::dims<1>{2}};

    EXPECT_FALSE(md::array_equiv(a, b));
}

TEST(run_time, 3) {
    using value_t = int;

    const auto a = md::container<value_t, md::dims<1>>{{1, 2}, md::dims<1>{2}};
    const auto b =
        md::container<value_t, md::dims<2>>{{1, 2, 1, 2}, md::dims<2>{2, 2}};

    EXPECT_TRUE(md::array_equiv(a, b));
}

TEST(run_time, 4) {
    using value_t = int;

    const auto a = md::container<value_t, md::dims<1>>{{1, 2}, md::dims<1>{2}};
    const auto b = md::container<value_t, md::dims<2>>{{1, 2, 1, 2, 1, 2, 1, 2},
                                                       md::dims<2>{2, 4}};

    EXPECT_FALSE(md::array_equiv(a, b));
}

TEST(run_time, 5) {
    using value_t = int;

    const auto a = md::container<value_t, md::dims<1>>{{1, 2}, md::dims<1>{2}};
    const auto b =
        md::container<value_t, md::dims<2>>{{1, 2, 1, 3}, md::dims<2>{2, 2}};

    EXPECT_FALSE(md::array_equiv(a, b));
}

TEST(compile_time, 1) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a = md::container<value_t, md::extents<index_t, 2>>{{1, 2}};
    constexpr auto b = md::container<value_t, md::extents<index_t, 2>>{{1, 2}};

    static_assert(md::array_equiv(a, b));
}

TEST(compile_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a = md::container<value_t, md::extents<index_t, 2>>{{1, 2}};
    constexpr auto b = md::container<value_t, md::extents<index_t, 2>>{{1, 3}};

    static_assert(!md::array_equiv(a, b));
}

TEST(compile_time, 3) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a = md::container<value_t, md::extents<index_t, 2>>{{1, 2}};
    constexpr auto b =
        md::container<value_t, md::extents<index_t, 2, 2>>{{1, 2, 1, 2}};

    static_assert(md::array_equiv(a, b));
}

TEST(compile_time, 4) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a = md::container<value_t, md::extents<index_t, 2>>{{1, 2}};
    constexpr auto b = md::container<value_t, md::extents<index_t, 2, 4>>{
        {1, 2, 1, 2, 1, 2, 1, 2}};

    // static_assert(!md::array_equiv(a, b)); // compile-time fails occur.
}

TEST(compile_time, 5) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a = md::container<value_t, md::extents<index_t, 2>>{{1, 2}};
    constexpr auto b =
        md::container<value_t, md::extents<index_t, 2, 2>>{{1, 2, 1, 3}};

    static_assert(!md::array_equiv(a, b));
}
