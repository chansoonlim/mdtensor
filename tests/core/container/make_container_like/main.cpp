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

TEST(make_container_like, 1) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a1 =
        md::core::container<value_t, md::core::extents<index_t, 2, 2>>{};
    constexpr auto a1_like = md::core::make_container_like(a1);

    std::cout << "a1: " << md::core::to_string(a1) << std::endl;
    std::cout << "a1_like: " << md::core::to_string(a1_like) << std::endl;

    constexpr auto a2 =
        md::core::container<value_t, md::core::extents<index_t, 2, 2>>{
            {1, 2, 3, 4}};
    constexpr auto a2_like = md::core::make_container_like(a2);

    std::cout << "a2: " << md::core::to_string(a2) << std::endl;
    std::cout << "a2_like: " << md::core::to_string(a2_like) << std::endl;

    const auto b1 =
        md::core::container<value_t, md::core::dextents<index_t, 2>>{
            md::core::dextents<index_t, 2>{2, 2}};
    const auto b1_like = md::core::make_container_like(b1);

    std::cout << "b1: " << md::core::to_string(b1) << std::endl;
    std::cout << "b1_like: " << md::core::to_string(b1_like) << std::endl;

    const auto b2 =
        md::core::container<value_t, md::core::dextents<index_t, 2>>{
            {1, 2, 3, 4}, md::core::dextents<index_t, 2>{2, 2}};
    const auto b2_like = md::core::make_container_like(b2);

    std::cout << "b2: " << md::core::to_string(b2) << std::endl;
    std::cout << "b2_like: " << md::core::to_string(b2_like) << std::endl;

    constexpr auto c =
        md::core::container<value_t, md::core::extents<index_t>>{};
    constexpr auto c_like = md::core::make_container_like(c);

    std::cout << "c: " << md::core::to_string(c) << std::endl;
    std::cout << "c_like: " << md::core::to_string(c_like) << std::endl;
}
