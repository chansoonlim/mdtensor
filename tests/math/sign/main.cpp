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
    using index_t = std::size_t;

    const auto a =
        md::container<value_t, md::dims<1>>{{-5, 4.5}, md::dims<1>{2}};

    const auto a_sign = md::sign(a);

    std::cout << "a_sign: " << md::to_string(a_sign) << std::endl;

    ASSERT_TRUE(md::array_equal(
        a_sign, md::container<int, md::extents<index_t, 2>>{{-1, 1}}));
}

TEST(run_time, 2) { ASSERT_EQ(md::sign(0), 0); }

TEST(compile_time, 1) {
    using value_t = float;
    using index_t = std::size_t;

    constexpr auto a =
        md::container<value_t, md::extents<index_t, 2>>{{-5, 4.5}};

    constexpr auto a_sign = md::sign(a);

    std::cout << "a_sign: " << md::to_string(a_sign) << std::endl;

    static_assert(md::array_equal(
        a_sign, md::container<int, md::extents<index_t, 2>>{{-1, 1}}));
}

TEST(compile_time, 2) { static_assert(md::sign(0) == 0); }
