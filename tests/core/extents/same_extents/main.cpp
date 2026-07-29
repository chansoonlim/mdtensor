/**
 * @file
 * @brief Tests
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#include <gtest/gtest.h>

#include "mdtensor/core/extents.hpp"

namespace stdex = std::experimental;
namespace md = mdtensor;

TEST(base, 1) {
    static_assert(stdex::extents<size_t, 2, 3, 4>{} ==
                  stdex::extents<size_t, 2, 3, 4>{});

    static_assert(stdex::extents<size_t, 2, 3, 4>{} ==
                  stdex::extents<uint8_t, 2, 3, 4>{});

    static_assert(stdex::extents<size_t, 3, 3, 4>{} !=
                  stdex::extents<size_t, 2, 3>{});

    static_assert(stdex::dextents<size_t, 3>{3, 3, 4} !=
                  stdex::dextents<size_t, 3>{2, 3, 4});

    static_assert(
        stdex::extents<size_t, 3, stdex::dynamic_extent, stdex::dynamic_extent>{
            3, 3, 4} == stdex::extents<uint8_t, stdex::dynamic_extent,
                                       stdex::dynamic_extent, 4>{3, 3, 4});

    ASSERT_EQ((stdex::dextents<size_t, 3>{2, 3, 4}),
              (stdex::dextents<size_t, 3>{2, 3, 4}));

    ASSERT_EQ((stdex::dextents<size_t, 3>{2, 3, 4}),
              (stdex::dextents<uint8_t, 3>{2, 3, 4}));

    ASSERT_NE((stdex::dextents<size_t, 3>{2, 3, 4}),
              (stdex::dextents<size_t, 3>{2, 3, 5}));

    ASSERT_NE((stdex::dextents<size_t, 3>{2, 3, 4}),
              (stdex::dextents<size_t, 3>{2, 3, 5}));

    ASSERT_EQ((stdex::extents<size_t, stdex::dynamic_extent, 3,
                              stdex::dynamic_extent>{2, 3, 4}),
              (stdex::extents<size_t, stdex::dynamic_extent,
                              stdex::dynamic_extent, 4>{2, 3, 4}));
}

TEST(base, 2) {
    // using index_t = size_t;

    // Direct comparison of extents cannot capture the case where the ranks are
    // different.

    // static_assert(stdex::extents<size_t, 2, 3>{} !=
    //               stdex::extents<size_t, 2, 3, 4>{});

    // ASSERT_NE((stdex::dextents<size_t, 2>{2, 3}),
    //           (stdex::dextents<size_t, 3>{2, 3, 4}));
}

TEST(same_extents, 1) {
    static_assert(md::core::same_extents(
        stdex::extents<size_t, 3, stdex::dynamic_extent, stdex::dynamic_extent>{
            3, 3, 4}));

    static_assert(md::core::same_extents(
        stdex::extents<size_t, 3, stdex::dynamic_extent, stdex::dynamic_extent>{
            3, 3, 4},
        stdex::extents<uint8_t, stdex::dynamic_extent, stdex::dynamic_extent,
                       4>{3, 3, 4}));

    static_assert(md::core::same_extents(
        stdex::extents<size_t, 3, stdex::dynamic_extent, stdex::dynamic_extent>{
            3, 3, 4},
        stdex::extents<size_t, 3, 3, stdex::dynamic_extent>{3, 3, 4},
        stdex::extents<uint8_t, stdex::dynamic_extent, stdex::dynamic_extent,
                       4>{3, 3, 4}));

    ASSERT_TRUE(md::core::same_extents(
        stdex::extents<size_t, 3, stdex::dynamic_extent, stdex::dynamic_extent>{
            3, 3, 4}));

    ASSERT_FALSE(md::core::same_extents(
        stdex::extents<size_t, 3, stdex::dynamic_extent, stdex::dynamic_extent>{
            3, 3, 4},
        stdex::extents<size_t, stdex::dynamic_extent, stdex::dynamic_extent, 4>{
            2, 3, 4}));

    ASSERT_TRUE(md::core::same_extents(
        stdex::extents<size_t, stdex::dynamic_extent, 3, stdex::dynamic_extent>{
            2, 3, 4},
        stdex::extents<size_t, stdex::dynamic_extent, stdex::dynamic_extent, 4>{
            2, 3, 4},
        stdex::extents<size_t, 2, stdex::dynamic_extent, 4>{2, 3, 4}));
}

TEST(same_extents, 2) {
    static_assert(!md::core::same_extents(stdex::extents<size_t, 2, 3>{},
                                          stdex::extents<size_t, 2, 3, 4>{}));

    ASSERT_FALSE(md::core::same_extents(stdex::dextents<size_t, 2>{2, 3},
                                        stdex::dextents<size_t, 3>{2, 3, 4}));
}
