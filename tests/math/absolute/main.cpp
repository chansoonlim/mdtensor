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

    const auto x =
        md::tensor<value_t, md::dims<1>>{{-1.2, 1.2}, md::dims<1>{2}};

    const auto x_abs = md::absolute(x);

    std::cout << "x_abs: " << md::to_string(x_abs) << std::endl;

    ASSERT_TRUE(md::allclose(
        x_abs, md::tensor<value_t, md::extents<index_t, 2>>{{1.2, 1.2}}));
}

TEST(compile_time, 1) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto x =
        md::tensor<value_t, md::extents<index_t, 2>>{{-1.2, 1.2}};

    constexpr auto x_abs = md::absolute(x);

    std::cout << "x_abs: " << md::to_string(x_abs) << std::endl;

    static_assert(md::allclose(
        x_abs, md::tensor<value_t, md::extents<index_t, 2>>{{1.2, 1.2}}));
}
