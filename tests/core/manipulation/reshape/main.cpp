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

    const auto a = md::arange(6);

    EXPECT_TRUE(
        md::array_equal(md::core::reshape(a, md::extents<index_t, 3, 2>{}),
                        md::core::tensor<value_t, md::extents<index_t, 3, 2>>{
                            {0, 1, 2, 3, 4, 5}}));
}

TEST(run_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    const auto a = md::arange(6);

    EXPECT_TRUE(
        md::array_equal(md::core::reshape(a, md::extents<index_t, 2, 3>{}),
                        md::core::tensor<value_t, md::extents<index_t, 2, 3>>{
                            {0, 1, 2, 3, 4, 5}}));
}

TEST(run_time, 3) {
    using value_t = int;
    using index_t = std::size_t;

    const auto a = md::core::tensor<value_t, md::extents<index_t, 2, 3>>{
        {1, 2, 3, 4, 5, 6}};

    EXPECT_TRUE(
        md::array_equal(md::core::reshape(a, 6),
                        md::core::tensor<value_t, md::extents<index_t, 6>>{
                            {1, 2, 3, 4, 5, 6}}));
}

TEST(compile_time, 1) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a = md::arange<6>();

    static_assert(
        md::array_equal(md::core::reshape(a, md::extents<index_t, 3, 2>{}),
                        md::core::tensor<value_t, md::extents<index_t, 3, 2>>{
                            {0, 1, 2, 3, 4, 5}}));
}

TEST(compile_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a = md::arange<6>();

    static_assert(
        md::array_equal(md::core::reshape(a, md::extents<index_t, 2, 3>{}),
                        md::core::tensor<value_t, md::extents<index_t, 2, 3>>{
                            {0, 1, 2, 3, 4, 5}}));
}

TEST(compile_time, 3) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a = md::core::tensor<value_t, md::extents<index_t, 2, 3>>{
        {1, 2, 3, 4, 5, 6}};

    static_assert(
        md::array_equal(md::core::reshape(a, 6),
                        md::core::tensor<value_t, md::extents<index_t, 6>>{
                            {1, 2, 3, 4, 5, 6}}));
}
