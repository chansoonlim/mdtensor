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

TEST(stack, norm) {
    using T = double;

    constexpr auto x = md::tensor<T, md::extents<std::size_t, 2>>{{3, 4}};
    constexpr auto x_norm = md::linalg::norm(x);

    constexpr auto x_norm_expect = 5;

    constexpr bool allclose = md::allclose(x_norm, x_norm_expect);

    EXPECT_TRUE(allclose);
}

TEST(heap, norm) {
    using T = double;

    const auto x = md::tensor<T, md::dims<1>>{{3, 4}, md::dims<1>{2}};
    const auto x_norm = md::linalg::norm(x);

    const auto x_norm_expect = 5;

    const bool allclose = md::allclose(x_norm, x_norm_expect);

    EXPECT_TRUE(allclose);
}

TEST(stack, zero) {
    using T = double;

    constexpr auto x = md::tensor<T, md::extents<std::size_t, 2>>{{0, 0}};
    constexpr auto x_norm = md::linalg::norm(x);

    constexpr auto x_norm_expect = 0;

    constexpr bool allclose = md::allclose(x_norm, x_norm_expect);

    EXPECT_TRUE(allclose);
}

TEST(heap, zero) {
    using T = double;

    const auto x = md::tensor<T, md::dims<1>>{{0, 0}, md::dims<1>{2}};
    const auto x_norm = md::linalg::norm(x);

    const auto x_norm_expect = 0;

    const bool allclose = md::allclose(x_norm, x_norm_expect);

    EXPECT_TRUE(allclose);
}
