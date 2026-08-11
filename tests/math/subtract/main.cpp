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
    const auto out = md::subtract(1.0, 4.0);

    std::cout << "out: " << out << std::endl;

    ASSERT_EQ(out, -3.0);
}

TEST(run_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    const auto x1 = md::reshape(md::arange(9.0), md::dims<2>{3, 3});
    const auto x2 = md::arange(3.0);
    const auto out = md::subtract(x1, x2);

    std::cout << "out: " << md::to_string(out) << std::endl;

    ASSERT_TRUE(
        md::allclose(out, md::tensor<value_t, md::extents<index_t, 3, 3>>{
                              {0, 0, 0, 3, 3, 3, 6, 6, 6}}));
}

TEST(compile_time, 1) {
    constexpr auto out = md::subtract(1.0, 4.0);

    std::cout << "out: " << out << std::endl;

    static_assert(out == -3.0);
}

TEST(compile_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto x1 =
        md::reshape(md::arange<9, value_t>(), md::extents<index_t, 3, 3>{});
    constexpr auto x2 = md::arange<3, value_t>();
    constexpr auto out = md::subtract(x1, x2);

    std::cout << "out: " << md::to_string(out) << std::endl;

    static_assert(
        md::allclose(out, md::tensor<value_t, md::extents<index_t, 3, 3>>{
                              {0, 0, 0, 3, 3, 3, 6, 6, 6}}));
}
