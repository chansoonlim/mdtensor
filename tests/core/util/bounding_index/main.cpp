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
    using index_t = std::int8_t;

    const std::size_t bound = 2;

    EXPECT_EQ(md::core::bounding_index<index_t>(0, bound), 0);
    EXPECT_EQ(md::core::bounding_index<index_t>(1, bound), 1);
    EXPECT_EQ(md::core::bounding_index<index_t>(2, bound), 2);
    EXPECT_THROW(static_cast<void>(md::core::bounding_index<index_t>(3, bound)),
                 std::out_of_range);
    EXPECT_THROW(static_cast<void>(md::core::bounding_index<index_t>(4, bound)),
                 std::out_of_range);

    EXPECT_EQ(md::core::bounding_index<index_t>(-1, bound), 2);
    EXPECT_EQ(md::core::bounding_index<index_t>(-2, bound), 1);
    EXPECT_EQ(md::core::bounding_index<index_t>(-3, bound), 0);
    EXPECT_THROW(
        static_cast<void>(md::core::bounding_index<index_t>(-4, bound)),
        std::out_of_range);
    EXPECT_THROW(
        static_cast<void>(md::core::bounding_index<index_t>(-5, bound)),
        std::out_of_range);
}

TEST(compile_time, 1) {
    using index_t = std::int8_t;

    constexpr std::size_t bound = 2;

    static_assert(md::core::bounding_index<index_t>(0, bound) == 0);
    static_assert(md::core::bounding_index<index_t>(1, bound) == 1);
    static_assert(md::core::bounding_index<index_t>(2, bound) == 2);
    EXPECT_THROW(static_cast<void>(md::core::bounding_index<index_t>(3, bound)),
                 std::out_of_range);
    EXPECT_THROW(static_cast<void>(md::core::bounding_index<index_t>(4, bound)),
                 std::out_of_range);

    static_assert(md::core::bounding_index<index_t>(-1, bound) == 2);
    static_assert(md::core::bounding_index<index_t>(-2, bound) == 1);
    static_assert(md::core::bounding_index<index_t>(-3, bound) == 0);
    EXPECT_THROW(
        static_cast<void>(md::core::bounding_index<index_t>(-4, bound)),
        std::out_of_range);
    EXPECT_THROW(
        static_cast<void>(md::core::bounding_index<index_t>(-5, bound)),
        std::out_of_range);
}
