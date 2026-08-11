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
    using index_t = std::size_t;

    const auto x = md::tensor<int, md::dims<1>>{{-1, 1, 1, -1}, md::dims<1>{4}};
    const auto y = md::tensor<int, md::dims<1>>{{-1, -1, 1, 1}, md::dims<1>{4}};
    const auto out = md::rad2deg(md::atan2(y, x));

    std::cout << "out: " << md::to_string(out) << std::endl;

    ASSERT_TRUE(md::allclose(out, md::tensor<double, md::extents<index_t, 4>>{
                                      {-135., -45., 45., 135.}}));
}

TEST(run_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    const auto y = md::tensor<value_t, md::dims<1>>{{1., -1.}, md::dims<1>{2}};
    const auto x = md::tensor<value_t, md::dims<1>>{{0., 0.}, md::dims<1>{2}};
    const auto out = md::atan2(y, x);

    std::cout << "out: " << md::to_string(out) << std::endl;

    ASSERT_TRUE(md::allclose(out, md::tensor<value_t, md::extents<index_t, 2>>{
                                      {1.57079633, -1.57079633}}));
}

TEST(run_time, 3) {
    using value_t = double;
    using index_t = std::size_t;

    const auto y = md::tensor<value_t, md::dims<1>>{
        {0., 0., std::numeric_limits<value_t>::infinity()}, md::dims<1>{3}};
    const auto x = md::tensor<value_t, md::dims<1>>{
        {0., -0., std::numeric_limits<value_t>::infinity()}, md::dims<1>{3}};
    const auto out = md::atan2(y, x);

    std::cout << "out: " << md::to_string(out) << std::endl;

    ASSERT_TRUE(md::allclose(out, md::tensor<value_t, md::extents<index_t, 3>>{
                                      {0., 3.14159265, 0.78539816}}));
}

#ifdef REAL_GCC // NOTE: std::cos is not constexpr in clang 16.

TEST(compile_time, 1) {
    using index_t = std::size_t;

    constexpr auto x = md::tensor<int, md::extents<index_t, 4>>{{-1, 1, 1, -1}};
    constexpr auto y = md::tensor<int, md::extents<index_t, 4>>{{-1, -1, 1, 1}};
    constexpr auto out = md::rad2deg(md::atan2(y, x));

    std::cout << "out: " << md::to_string(out) << std::endl;

    static_assert(md::allclose(out, md::tensor<double, md::extents<index_t, 4>>{
                                        {-135., -45., 45., 135.}}));
}

TEST(compile_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto y = md::tensor<value_t, md::extents<index_t, 2>>{{1., -1.}};
    constexpr auto x = md::tensor<value_t, md::extents<index_t, 2>>{{0., 0.}};
    constexpr auto out = md::atan2(y, x);

    std::cout << "out: " << md::to_string(out) << std::endl;

    static_assert(
        md::allclose(out, md::tensor<value_t, md::extents<index_t, 2>>{
                              {1.57079633, -1.57079633}}));
}

TEST(compile_time, 3) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto y = md::tensor<value_t, md::extents<index_t, 3>>{
        {0., 0., std::numeric_limits<value_t>::infinity()}};
    constexpr auto x = md::tensor<value_t, md::extents<index_t, 3>>{
        {0., -0., std::numeric_limits<value_t>::infinity()}};

    // NOTE: atan2(+Inf, +Inf) is not a constant expression.
    const auto out = md::atan2(y, x);

    std::cout << "out: " << md::to_string(out) << std::endl;

    ASSERT_TRUE(md::allclose(out, md::tensor<value_t, md::extents<index_t, 3>>{
                                      {0., 3.14159265, 0.78539816}}));
}

#endif
