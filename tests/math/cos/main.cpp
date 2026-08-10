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

    const auto x = md::container<value_t, md::dims<1>>{
        {0, std::numbers::pi_v<value_t> / 2., std::numbers::pi_v<value_t>},
        md::dims<1>{3}};
    const auto x_cos = md::cos(x);

    std::cout << "x_cos: " << md::to_string(x_cos) << std::endl;

    ASSERT_TRUE(
        md::allclose(x_cos, md::container<value_t, md::extents<index_t, 3>>{
                                {1, 6.12303177e-17, -1}}));
}

#ifdef REAL_GCC // NOTE: std::cos is not constexpr in clang 16.

TEST(compile_time, 1) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto x = md::container<value_t, md::extents<index_t, 3>>{
        {0, std::numbers::pi_v<value_t> / 2., std::numbers::pi_v<value_t>}};
    constexpr auto x_cos = md::cos(x);

    std::cout << "x_cos: " << md::to_string(x_cos) << std::endl;

    static_assert(
        md::allclose(x_cos, md::container<value_t, md::extents<index_t, 3>>{
                                {1, 6.12303177e-17, -1}}));
}

#endif
