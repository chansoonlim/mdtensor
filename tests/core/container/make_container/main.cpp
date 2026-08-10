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

TEST(make_container, 1) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a1 =
        md::core::make_container<value_t>(md::core::extents<index_t, 2, 2>{});

    constexpr auto a2 =
        md::core::make_container(md::core::extents<index_t, 2, 2>{});

    std::cout << "a1: " << md::core::to_string(a1) << std::endl;
    std::cout << "a2: " << md::core::to_string(a2) << std::endl;
}
