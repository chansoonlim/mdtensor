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

    const auto x = md::reshape(md::arange(6), md::dims<2>{2, 3});
    const auto z = md::ones_like(x);

    static_assert(std::is_same_v<typename decltype(x)::value_type,
                                 typename decltype(z)::value_type>);

    EXPECT_TRUE(md::array_equal(
        z,
        md::tensor<value_t, md::extents<index_t, 2, 3>>{{1, 1, 1, 1, 1, 1}}));
}

TEST(run_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    const auto x = md::arange<value_t>(3);
    const auto z = md::ones_like(x);

    static_assert(std::is_same_v<typename decltype(x)::value_type,
                                 typename decltype(z)::value_type>);

    EXPECT_TRUE(md::array_equal(
        z, md::tensor<value_t, md::extents<index_t, 3>>{{1, 1, 1}}));
}

TEST(compile_time, 1) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto x =
        md::reshape(md::arange<6>(), md::extents<index_t, 2, 3>{});
    constexpr auto z = md::ones_like(x);

    static_assert(std::is_same_v<typename decltype(x)::value_type,
                                 typename decltype(z)::value_type>);

    static_assert(md::array_equal(
        z,
        md::tensor<value_t, md::extents<index_t, 2, 3>>{{1, 1, 1, 1, 1, 1}}));
}

TEST(compile_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto x = md::arange<3, value_t>();
    constexpr auto z = md::ones_like(x);

    static_assert(std::is_same_v<typename decltype(x)::value_type,
                                 typename decltype(z)::value_type>);

    static_assert(md::array_equal(
        z, md::tensor<value_t, md::extents<index_t, 3>>{{1, 1, 1}}));
}
