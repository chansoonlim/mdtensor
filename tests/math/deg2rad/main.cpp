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
    using value_t = float;

    const auto x_rad = md::deg2rad(180);

    std::cout << "x_rad: " << x_rad << std::endl;

    ASSERT_TRUE(md::isclose(x_rad, std::numbers::pi_v<value_t>));
}

TEST(compile_time, 1) {
    using value_t = float;

    constexpr auto x_rad = md::deg2rad(180);

    std::cout << "x_rad: " << x_rad << std::endl;

    static_assert(md::isclose(x_rad, std::numbers::pi_v<value_t>));
}
