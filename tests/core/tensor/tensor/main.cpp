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

TEST(tensor, 1) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a1 =
        md::core::tensor<value_t, md::core::extents<index_t, 2, 2>>{};

    std::cout << "a1: " << md::to_string(a1) << std::endl;

    auto a2 = md::core::tensor<value_t, md::core::extents<index_t, 2, 2>>{
        {1, 2, 3, 4}};

    std::cout << "a2: " << md::to_string(a2) << std::endl;

    const auto b1 = md::core::tensor<value_t, md::core::dextents<index_t, 2>>{
        md::core::dextents<index_t, 2>{2, 2}};

    std::cout << "b1: " << md::to_string(b1) << std::endl;

    const auto b2 = md::core::tensor<value_t, md::core::dextents<index_t, 2>>{
        {1, 2, 3, 4}, md::core::dextents<index_t, 2>{2, 2}};

    std::cout << "b2: " << md::to_string(b2) << std::endl;

    constexpr auto c = md::core::tensor<value_t, md::core::extents<index_t>>{};

    std::cout << "c: " << md::to_string(c) << std::endl;
}
