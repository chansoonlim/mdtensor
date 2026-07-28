/**
 * @file
 * @brief Tests
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#include <gtest/gtest.h>

#include "mdtensor/core/container.hpp"

namespace stdex = std::experimental;
namespace md = mdtensor;

TEST(mdarray, 1) {
    using value_t = int;
    using index_t = size_t;

    // this does not work
    // [[maybe_unused]] constexpr auto a1 =
    //     stdex::mdarray<value_t, stdex::extents<index_t, 2, 2>>{};

    // this does not work
    // [[maybe_unused]] constexpr auto a2 =
    //     stdex::mdarray<value_t, stdex::extents<index_t, 2, 2>>{{1, 2, 3, 4}};

    [[maybe_unused]] constexpr auto a3 =
        stdex::mdarray<value_t, stdex::extents<index_t, 2, 2>,
                       stdex::layout_right, std::array<value_t, 4>>{};

    [[maybe_unused]] constexpr auto a4 =
        stdex::mdarray<value_t, stdex::extents<index_t, 2, 2>,
                       stdex::layout_right, std::array<value_t, 4>>{
            {1, 2, 3, 4}};

    [[maybe_unused]] const auto b1 =
        stdex::mdarray<value_t, stdex::dextents<index_t, 2>>{
            stdex::dextents<index_t, 2>{2, 2}};

    [[maybe_unused]] const auto b2 =
        stdex::mdarray<value_t, stdex::dextents<index_t, 2>>{
            {1, 2, 3, 4}, stdex::dextents<index_t, 2>{2, 2}};

    // this does not work
    // [[maybe_unused]] constexpr auto c =
    //     stdex::mdarray<value_t, stdex::extents<index_t>>{};
}

TEST(container, 1) {
    using value_t = int;
    using index_t = size_t;

    [[maybe_unused]] constexpr auto a1 =
        md::core::container<value_t, stdex::extents<index_t, 2, 2>>{};

    [[maybe_unused]] auto a2 =
        md::core::container<value_t, stdex::extents<index_t, 2, 2>>{
            {1, 2, 3, 4}};

    [[maybe_unused]] const auto b1 =
        md::core::container<value_t, stdex::dextents<index_t, 2>>{
            stdex::dextents<index_t, 2>{2, 2}};

    [[maybe_unused]] const auto b2 =
        md::core::container<value_t, stdex::dextents<index_t, 2>>{
            {1, 2, 3, 4}, stdex::dextents<index_t, 2>{2, 2}};

    [[maybe_unused]] constexpr auto c =
        md::core::container<value_t, stdex::extents<index_t>>{};
}

TEST(make_container_like, 1) {
    using value_t = int;
    using index_t = size_t;

    [[maybe_unused]] constexpr auto a1 =
        md::core::container<value_t, stdex::extents<index_t, 2, 2>>{};
    [[maybe_unused]] constexpr auto a1_like = md::core::make_container_like(a1);

    [[maybe_unused]] constexpr auto a2 =
        md::core::container<value_t, stdex::extents<index_t, 2, 2>>{
            {1, 2, 3, 4}};
    [[maybe_unused]] constexpr auto a2_like = md::core::make_container_like(a2);

    [[maybe_unused]] const auto b1 =
        md::core::container<value_t, stdex::dextents<index_t, 2>>{
            stdex::dextents<index_t, 2>{2, 2}};
    [[maybe_unused]] const auto b1_like = md::core::make_container_like(b1);

    [[maybe_unused]] const auto b2 =
        md::core::container<value_t, stdex::dextents<index_t, 2>>{
            {1, 2, 3, 4}, stdex::dextents<index_t, 2>{2, 2}};
    [[maybe_unused]] const auto b2_like = md::core::make_container_like(b2);

    [[maybe_unused]] constexpr auto c =
        md::core::container<value_t, stdex::extents<index_t>>{};
    [[maybe_unused]] constexpr auto c_like = md::core::make_container_like(c);
}
