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

    const auto out = md::zeros(5);

    static_assert(std::is_same_v<typename decltype(out)::value_type, value_t>);

    EXPECT_TRUE(md::array_equal(
        out, md::tensor<value_t, md::extents<index_t, 5>>{{0, 0, 0, 0, 0}}));
}

TEST(run_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    const auto out = md::zeros<value_t>(5);

    static_assert(std::is_same_v<typename decltype(out)::value_type, value_t>);

    EXPECT_TRUE(md::array_equal(
        out, md::tensor<value_t, md::extents<index_t, 5>>{{0, 0, 0, 0, 0}}));
}

TEST(run_time, 3) {
    using value_t = double;
    using index_t = std::size_t;

    const auto out = md::zeros(md::dims<2>{2, 1});

    EXPECT_TRUE(md::array_equal(
        out, md::tensor<value_t, md::extents<index_t, 2, 1>>{{0, 0}}));
}

TEST(run_time, 4) {
    using value_t = double;
    using index_t = std::size_t;

    const auto out = md::zeros(md::dims<2>{2, 2});

    EXPECT_TRUE(md::array_equal(
        out, md::tensor<value_t, md::extents<index_t, 2, 2>>{{0, 0, 0, 0}}));
}

TEST(compile_time, 1) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto out = md::zeros(md::extents<index_t, 5>{});

    static_assert(std::is_same_v<typename decltype(out)::value_type, value_t>);

    static_assert(md::array_equal(
        out, md::tensor<value_t, md::extents<index_t, 5>>{{0, 0, 0, 0, 0}}));
}

TEST(compile_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto out = md::zeros<value_t>(md::extents<index_t, 5>{});

    static_assert(std::is_same_v<typename decltype(out)::value_type, value_t>);

    static_assert(md::array_equal(
        out, md::tensor<value_t, md::extents<index_t, 5>>{{0, 0, 0, 0, 0}}));
}

TEST(compile_time, 3) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto out = md::zeros(md::extents<index_t, 2, 1>{});

    static_assert(md::array_equal(
        out, md::tensor<value_t, md::extents<index_t, 2, 1>>{{0, 0}}));
}

TEST(compile_time, 4) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto out = md::zeros(md::extents<index_t, 2, 2>{2, 2});

    static_assert(md::array_equal(
        out, md::tensor<value_t, md::extents<index_t, 2, 2>>{{0, 0, 0, 0}}));
}
