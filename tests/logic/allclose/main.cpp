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

    const auto a =
        md::tensor<value_t, md::dims<1>>{{1e10, 1e-7}, md::dims<1>{2}};
    const auto b =
        md::tensor<value_t, md::dims<1>>{{1.00001e10, 1e-8}, md::dims<1>{2}};

    EXPECT_FALSE(md::allclose(a, b));
}

TEST(run_time, 2) {
    using value_t = double;

    const auto a =
        md::tensor<value_t, md::dims<1>>{{1e10, 1e-8}, md::dims<1>{2}};
    const auto b =
        md::tensor<value_t, md::dims<1>>{{1.00001e10, 1e-9}, md::dims<1>{2}};

    EXPECT_TRUE(md::allclose(a, b));
}

TEST(run_time, 3) {
    using value_t = double;

    const auto a =
        md::tensor<value_t, md::dims<1>>{{1e10, 1e-8}, md::dims<1>{2}};
    const auto b =
        md::tensor<value_t, md::dims<1>>{{1.0001e10, 1e-9}, md::dims<1>{2}};

    EXPECT_FALSE(md::allclose(a, b));
}

TEST(run_time, 4) {
    using value_t = double;

    const auto a = md::tensor<value_t, md::dims<1>>{
        {1, std::numeric_limits<value_t>::quiet_NaN()}, md::dims<1>{2}};
    const auto b = md::tensor<value_t, md::dims<1>>{
        {1, std::numeric_limits<value_t>::quiet_NaN()}, md::dims<1>{2}};

    EXPECT_FALSE(md::allclose(a, b));
}

TEST(run_time, 5) {
    using value_t = double;

    const auto a = md::tensor<value_t, md::dims<1>>{
        {1, std::numeric_limits<value_t>::quiet_NaN()}, md::dims<1>{2}};
    const auto b = md::tensor<value_t, md::dims<1>>{
        {1, std::numeric_limits<value_t>::quiet_NaN()}, md::dims<1>{2}};

    EXPECT_TRUE(md::allclose(a, b, 1e-05, 1e-08, true));
}

TEST(compile_time, 1) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a =
        md::tensor<value_t, md::extents<index_t, 2>>{{1e10, 1e-7}};
    constexpr auto b =
        md::tensor<value_t, md::extents<index_t, 2>>{{1.00001e10, 1e-8}};

    static_assert(!md::allclose(a, b));
}

TEST(compile_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a =
        md::tensor<value_t, md::extents<index_t, 2>>{{1e10, 1e-8}};
    constexpr auto b =
        md::tensor<value_t, md::extents<index_t, 2>>{{1.00001e10, 1e-9}};

    static_assert(md::allclose(a, b));
}

TEST(compile_time, 3) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a =
        md::tensor<value_t, md::extents<index_t, 2>>{{1e10, 1e-8}};
    constexpr auto b =
        md::tensor<value_t, md::extents<index_t, 2>>{{1.0001e10, 1e-9}};

    static_assert(!md::allclose(a, b));
}

TEST(compile_time, 4) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a = md::tensor<value_t, md::extents<index_t, 2>>{
        {1, std::numeric_limits<value_t>::quiet_NaN()}};
    constexpr auto b = md::tensor<value_t, md::extents<index_t, 2>>{
        {1, std::numeric_limits<value_t>::quiet_NaN()}};

    static_assert(!md::allclose(a, b));
}

TEST(compile_time, 5) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a = md::tensor<value_t, md::extents<index_t, 2>>{
        {1, std::numeric_limits<value_t>::quiet_NaN()}};
    constexpr auto b = md::tensor<value_t, md::extents<index_t, 2>>{
        {1, std::numeric_limits<value_t>::quiet_NaN()}};

    static_assert(md::allclose(a, b, 1e-05, 1e-08, true));
}
