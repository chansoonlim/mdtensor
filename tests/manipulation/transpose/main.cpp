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
        md::tensor<value_t, md::dims<2>>{{1, 2, 3, 4}, md::dims<2>{2, 2}};

    const auto a_t = md::transpose(a);

    EXPECT_TRUE(md::array_equal(
        a_t, md::tensor<value_t, md::extents<index_t, 2, 2>>{{1, 3, 2, 4}}));
}

TEST(run_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    const auto a =
        md::tensor<value_t, md::dims<1>>{{1, 2, 3, 4}, md::dims<1>{4}};
    const auto a_t = md::transpose(a);

    EXPECT_TRUE(md::array_equal(
        a_t, md::tensor<value_t, md::extents<index_t, 4>>{{1, 2, 3, 4}}));
}

TEST(run_time, 3) {
    using index_t = std::size_t;

    const auto a = md::ones(md::dims<3>{1, 2, 3});
    const auto a_t = md::transpose<1, 0, 2>(a);

    EXPECT_TRUE(
        md::is_same_extents(a_t.extents(), md::extents<index_t, 2, 1, 3>{}));
}

TEST(run_time, 4) {
    using index_t = std::size_t;

    const auto a = md::ones(md::dims<4>{2, 3, 4, 5});
    const auto a_t = md::transpose(a);

    EXPECT_TRUE(
        md::is_same_extents(a_t.extents(), md::extents<index_t, 5, 4, 3, 2>{}));
}

TEST(run_time, 5) {
    using index_t = std::size_t;

    const auto a = md::reshape(md::arange(3 * 4 * 5), md::dims<3>{3, 4, 5});
    const auto a_t = md::transpose<-1, 0, -2>(a);

    EXPECT_TRUE(
        md::is_same_extents(a_t.extents(), md::extents<index_t, 5, 3, 4>{}));
}

TEST(compile_time, 1) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a =
        md::tensor<value_t, md::extents<index_t, 2, 2>>{{1, 2, 3, 4}};

    static_assert(md::array_equal(
        md::transpose(a),
        md::tensor<value_t, md::extents<index_t, 2, 2>>{{1, 3, 2, 4}}));
}

TEST(compile_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a =
        md::tensor<value_t, md::extents<index_t, 4>>{{1, 2, 3, 4}};

    static_assert(md::array_equal(
        md::transpose(a),
        md::tensor<value_t, md::extents<index_t, 4>>{{1, 2, 3, 4}}));
}

TEST(compile_time, 3) {
    using index_t = std::size_t;

    constexpr auto a = md::ones(md::extents<index_t, 1, 2, 3>{});

    static_assert(md::is_same_extents(md::transpose<1, 0, 2>(a).extents(),
                                      md::extents<index_t, 2, 1, 3>{}));
}

TEST(compile_time, 4) {
    using index_t = std::size_t;

    constexpr auto a = md::ones(md::extents<index_t, 2, 3, 4, 5>{});

    static_assert(md::is_same_extents(md::transpose(a).extents(),
                                      md::extents<index_t, 5, 4, 3, 2>{}));
}

TEST(compile_time, 5) {
    using index_t = std::size_t;

    constexpr auto a =
        md::reshape(md::arange<3 * 4 * 5>(), md::extents<index_t, 3, 4, 5>{});

    static_assert(md::is_same_extents(md::transpose<-1, 0, -2>(a).extents(),
                                      md::extents<index_t, 5, 3, 4>{}));
}
