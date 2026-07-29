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

TEST(base, 1) {
    static_assert(md::core::extents<std::size_t, 2, 3, 4>{} ==
                  md::core::extents<std::size_t, 2, 3, 4>{});

    static_assert(md::core::extents<std::size_t, 2, 3, 4>{} ==
                  md::core::extents<std::uint8_t, 2, 3, 4>{});

    static_assert(md::core::extents<std::size_t, 3, 3, 4>{} !=
                  md::core::extents<std::size_t, 2, 3>{});

    static_assert(md::core::dextents<std::size_t, 3>{3, 3, 4} !=
                  md::core::dextents<std::size_t, 3>{2, 3, 4});

    static_assert(
        md::core::extents<std::size_t, 3, md::core::dyn, md::core::dyn>{3, 3,
                                                                        4} ==
        md::core::extents<std::uint8_t, md::core::dyn, md::core::dyn, 4>{3, 3,
                                                                         4});

    EXPECT_EQ((md::core::dextents<std::size_t, 3>{2, 3, 4}),
              (md::core::dextents<std::size_t, 3>{2, 3, 4}));

    EXPECT_EQ((md::core::dextents<std::size_t, 3>{2, 3, 4}),
              (md::core::dextents<std::uint8_t, 3>{2, 3, 4}));

    EXPECT_NE((md::core::dextents<std::size_t, 3>{2, 3, 4}),
              (md::core::dextents<std::size_t, 3>{2, 3, 5}));

    EXPECT_NE((md::core::dextents<std::size_t, 3>{2, 3, 4}),
              (md::core::dextents<std::size_t, 3>{2, 3, 5}));

    EXPECT_EQ((md::core::extents<std::size_t, md::core::dyn, 3, md::core::dyn>{
                  2, 3, 4}),
              (md::core::extents<std::size_t, md::core::dyn, md::core::dyn, 4>{
                  2, 3, 4}));
}

TEST(base, 2) {
    // using index_t = std::size_t;

    // Direct comparison of extents cannot capture the case where the ranks are
    // different.

    // static_assert(md::core::extents<std::size_t, 2, 3>{} !=
    //               md::core::extents<std::size_t, 2, 3, 4>{});

    // EXPECT_NE((md::core::dextents<std::size_t, 2>{2, 3}),
    //           (md::core::dextents<std::size_t, 3>{2, 3, 4}));
}

TEST(is_same_extents, 1) {
    static_assert(md::core::is_same_extents(
        md::core::extents<std::size_t, 3, md::core::dyn, md::core::dyn>{3, 3,
                                                                        4}));

    static_assert(md::core::is_same_extents(
        md::core::extents<std::size_t, 3, md::core::dyn, md::core::dyn>{3, 3,
                                                                        4},
        md::core::extents<std::uint8_t, md::core::dyn, md::core::dyn, 4>{3, 3,
                                                                         4}));

    static_assert(md::core::is_same_extents(
        md::core::extents<std::size_t, 3, md::core::dyn, md::core::dyn>{3, 3,
                                                                        4},
        md::core::extents<std::size_t, 3, 3, md::core::dyn>{3, 3, 4},
        md::core::extents<std::uint8_t, md::core::dyn, md::core::dyn, 4>{3, 3,
                                                                         4}));

    EXPECT_TRUE(md::core::is_same_extents(
        md::core::extents<std::size_t, 3, md::core::dyn, md::core::dyn>{3, 3,
                                                                        4}));

    EXPECT_FALSE(md::core::is_same_extents(
        md::core::extents<std::size_t, 3, md::core::dyn, md::core::dyn>{3, 3,
                                                                        4},
        md::core::extents<std::size_t, md::core::dyn, md::core::dyn, 4>{2, 3,
                                                                        4}));

    EXPECT_TRUE(md::core::is_same_extents(
        md::core::extents<std::size_t, md::core::dyn, 3, md::core::dyn>{2, 3,
                                                                        4},
        md::core::extents<std::size_t, md::core::dyn, md::core::dyn, 4>{2, 3,
                                                                        4},
        md::core::extents<std::size_t, 2, md::core::dyn, 4>{2, 3, 4}));
}

TEST(is_same_extents, 2) {
    static_assert(
        !md::core::is_same_extents(md::core::extents<std::size_t, 2, 3>{},
                                   md::core::extents<std::size_t, 2, 3, 4>{}));

    EXPECT_FALSE(
        md::core::is_same_extents(md::core::dextents<std::size_t, 2>{2, 3},
                                  md::core::dextents<std::size_t, 3>{2, 3, 4}));
}
