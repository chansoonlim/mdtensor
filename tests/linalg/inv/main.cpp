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
        md::tensor<value_t, md::extents<index_t, 2, 2>>{{1, 2, 3, 4}};
    const auto [a_inv, valid] = md::linalg::inv(a);

    EXPECT_TRUE(md::allclose(
        a_inv,
        md::tensor<value_t, md::extents<index_t, 2, 2>>{{-2, 1, 1.5, -0.5}}));

    EXPECT_TRUE(
        md::allclose(md::linalg::matmul(a, a_inv), md::eye<value_t>(2)));

    EXPECT_TRUE(
        md::allclose(md::linalg::matmul(a_inv, a), md::eye<value_t>(2)));
}

TEST(run_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    const auto a = md::tensor<value_t, md::extents<index_t, 2, 2, 2>>{
        {1, 2, 3, 4, 1, 3, 3, 5}};
    const auto [a_inv, valid] = md::linalg::inv(a);

    EXPECT_TRUE(
        md::allclose(a_inv, md::tensor<value_t, md::extents<index_t, 2, 2, 2>>{
                                {-2, 1, 1.5, -0.5, -1.25, 0.75, 0.75, -0.25}}));
}

TEST(run_time, 3) {
    using value_t = double;
    using index_t = std::size_t;

    const auto a = md::tensor<value_t, md::extents<index_t, 3, 3>>{
        {2, 4, 6, 2, 0, 2, 6, 8, 14}};
    const auto [a_inv, valid] = md::linalg::inv(a);

    // NOTE: if a matrix is close to singular, computed inverse may not
    // satisfy a * a_inv = I even if the valid flag is true.
    EXPECT_TRUE(valid);

    EXPECT_TRUE(
        !md::allclose(md::linalg::matmul(a, a_inv), md::eye<value_t>(3)));

    // EXPECT_TRUE(cond(a) > 1e10); // TODO: implement cond()
}

TEST(compile_time, 1) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a =
        md::tensor<value_t, md::extents<index_t, 2, 2>>{{1, 2, 3, 4}};
    constexpr auto a_inv = std::get<0>(md::linalg::inv(a));

    static_assert(md::allclose(
        a_inv,
        md::tensor<value_t, md::extents<index_t, 2, 2>>{{-2, 1, 1.5, -0.5}}));

    static_assert(
        md::allclose(md::linalg::matmul(a, a_inv), md::eye<2, value_t>()));

    static_assert(
        md::allclose(md::linalg::matmul(a_inv, a), md::eye<2, value_t>()));
}

TEST(compile_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a = md::tensor<value_t, md::extents<index_t, 2, 2, 2>>{
        {1, 2, 3, 4, 1, 3, 3, 5}};
    constexpr auto a_inv = std::get<0>(md::linalg::inv(a));

    static_assert(
        md::allclose(a_inv, md::tensor<value_t, md::extents<index_t, 2, 2, 2>>{
                                {-2, 1, 1.5, -0.5, -1.25, 0.75, 0.75, -0.25}}));
}

TEST(compile_time, 3) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a = md::tensor<value_t, md::extents<index_t, 3, 3>>{
        {2, 4, 6, 2, 0, 2, 6, 8, 14}};
    constexpr auto out = md::linalg::inv(a);
    constexpr auto a_inv = std::get<0>(out);
    constexpr auto valid = std::get<1>(out);

    // NOTE: if a matrix is close to singular, computed inverse may not
    // satisfy a * a_inv = I even if the valid flag is true.
    static_assert(valid);

    static_assert(
        !md::allclose(md::linalg::matmul(a, a_inv), md::eye<3, value_t>()));

    // static_assert(cond(a) > 1e10); // TODO: implement cond()
}
