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

TEST(test, 1) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a =
        md::core::mdarray<value_t, md::core::extents<index_t, 2, 2>,
                          md::core::stdex::layout_right,
                          std::array<value_t, 9>>{{1, 2, 3, 4, 5, 6, 7, 8, 9}};

    auto b = md::core::submdspan(a, std::pair{0, 1}, 1);

    std::cout << "a: " << md::to_string(a) << std::endl;
    std::cout << "b: " << md::to_string(b) << std::endl;
}
