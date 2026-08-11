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

    const auto x = md::tensor<value_t, md::dims<1>>{{1, -1}, md::dims<1>{2}};
    const auto out = md::negative(x);

    std::cout << "out: " << md::to_string(out) << std::endl;

    ASSERT_TRUE(md::array_equal(
        out, md::tensor<value_t, md::extents<index_t, 2>>{{-1, 1}}));
}

TEST(run_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto x = md::tensor<value_t, md::extents<index_t, 2>>{{1, -1}};
    constexpr auto out = md::negative(x);

    std::cout << "out: " << md::to_string(out) << std::endl;

    static_assert(md::array_equal(
        out, md::tensor<value_t, md::extents<index_t, 2>>{{-1, 1}}));
}
