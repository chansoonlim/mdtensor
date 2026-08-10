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
    const auto out = md::divide(2.0, 4.0);

    std::cout << "out: " << out << std::endl;

    ASSERT_EQ(out, 0.5);
}

TEST(run_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    const auto x1 = md::reshape(md::arange(9.0), md::dims<2>{3, 3});
    const auto x2 = md::arange(3.0);
    const auto out = md::divide(x1, x2);

    std::cout << "out: " << md::to_string(out) << std::endl;

    ASSERT_TRUE(
        md::allclose(out,
                     md::container<value_t, md::extents<index_t, 3, 3>>{
                         {std::numeric_limits<double>::quiet_NaN(), 1., 1.,
                          std::numeric_limits<double>::infinity(), 4., 2.5,
                          std::numeric_limits<double>::infinity(), 7., 4.}},
                     1e-05, 1e-08, true));
}

TEST(compile_time, 1) {
    constexpr auto out = md::divide(2.0, 4.0);

    std::cout << "out: " << out << std::endl;

    static_assert(out == 0.5);
}

TEST(compile_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto x1 =
        md::reshape(md::arange<9, value_t>(), md::extents<index_t, 3, 3>{});
    constexpr auto x2 = md::arange<3, value_t>();

    // NOTE: (0.0 / 0.0) is not a constant expression
    const auto out = md::divide(x1, x2);

    std::cout << "out: " << md::to_string(out) << std::endl;

    ASSERT_TRUE(
        md::allclose(out,
                     md::container<value_t, md::extents<index_t, 3, 3>>{
                         {std::numeric_limits<double>::quiet_NaN(), 1., 1.,
                          std::numeric_limits<double>::infinity(), 4., 2.5,
                          std::numeric_limits<double>::infinity(), 7., 4.}},
                     1e-05, 1e-08, true));
}
