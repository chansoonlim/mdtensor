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
    using value_t = double;
    using index_t = std::size_t;

    const auto x = md::container<value_t, md::dims<1>>{
        {-std::numbers::pi_v<value_t>, std::numbers::pi_v<value_t> / 2.,
         std::numbers::pi_v<value_t>},
        md::dims<1>{3}};
    const auto x_tan = md::tan(x);

    std::cout << "x_tan: " << md::to_string(x_tan) << std::endl;

    ASSERT_TRUE(
        md::allclose(x_tan,
                     md::container<value_t, md::extents<index_t, 3>>{
                         {1.22460635e-16, 1.63317787e+16, -1.22460635e-16}},
                     1e-04));
}

#ifdef REAL_GCC

TEST(compile_time, 1) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto x = md::container<value_t, md::extents<index_t, 3>>{
        {-std::numbers::pi_v<value_t>, std::numbers::pi_v<value_t> / 2.,
         std::numbers::pi_v<value_t>}};
    constexpr auto x_tan = md::tan(x);

    std::cout << "x_tan: " << md::to_string(x_tan) << std::endl;

    static_assert(
        md::allclose(x_tan,
                     md::container<value_t, md::extents<index_t, 3>>{
                         {1.22460635e-16, 1.63317787e+16, -1.22460635e-16}},
                     1e-04));
}

#endif
