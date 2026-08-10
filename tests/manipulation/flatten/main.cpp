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

    const auto a =
        md::container<value_t, md::dims<2>>{{1, 2, 3, 4}, md::dims<2>{2, 2}};

    EXPECT_TRUE(md::array_equal(
        md::flatten(a),
        md::container<value_t, md::extents<index_t, 4>>{{1, 2, 3, 4}}));
}

TEST(compile_time, 1) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a =
        md::container<value_t, md::extents<index_t, 2, 2>>{{1, 2, 3, 4}};

    static_assert(md::array_equal(
        md::flatten(a),
        md::container<value_t, md::extents<index_t, 4>>{{1, 2, 3, 4}}));
}
