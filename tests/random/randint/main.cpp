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
    using index_t = std::size_t;

    const auto out1 = md::random::randint(10, 2);
    const auto out2 = md::random::randint(10, 1);

    EXPECT_TRUE(
        md::is_same_extents(out1.extents(), md::extents<index_t, 10>{}));
    EXPECT_TRUE(
        md::is_same_extents(out2.extents(), md::extents<index_t, 10>{}));

    std::cout << "out1: " << md::to_string(out1) << std::endl; // random
    std::cout << "out2: " << md::to_string(out2) << std::endl; // bounded to 0

    EXPECT_TRUE(md::array_equiv(out2, 0));
}

TEST(run_time, 2) {
    using index_t = std::size_t;

    const auto out = md::random::randint(md::dims<2>{2, 4}, 5);

    EXPECT_TRUE(
        md::is_same_extents(out.extents(), md::extents<index_t, 2, 4>{}));

    std::cout << "out: " << md::to_string(out) << std::endl; // random
}

TEST(run_time, 3) {
    using value_t = int;
    using index_t = std::size_t;

    const auto out = md::random::randint(
        md::extents<index_t>{}, 1,
        md::container<value_t, md::dims<1>>{{3, 5, 10}, md::dims<1>{3}});

    EXPECT_TRUE(md::is_same_extents(out.extents(), md::extents<index_t, 3>{}));

    std::cout << "out: " << md::to_string(out) << std::endl; // random
}

TEST(run_time, 4) {
    using value_t = int;
    using index_t = std::size_t;

    const auto out = md::random::randint(
        md::extents<index_t>{},
        md::container<value_t, md::dims<1>>{{1, 5, 7}, md::dims<1>{3}}, 10);

    EXPECT_TRUE(md::is_same_extents(out.extents(), md::extents<index_t, 3>{}));

    std::cout << "out: " << md::to_string(out) << std::endl; // random
}

TEST(run_time, 5) {
    using value_t = uint8_t;
    using index_t = std::size_t;

    const auto out = md::random::randint<value_t>(
        md::extents<index_t>{},
        md::container<value_t, md::dims<1>>{{1, 3, 5, 7}, md::dims<1>{4}},
        md::container<value_t, md::dims<2>>{{10, 20}, md::dims<2>{2, 1}});

    EXPECT_TRUE(
        md::is_same_extents(out.extents(), md::extents<index_t, 2, 4>{}));

    std::cout << "out: " << md::to_string(out) << std::endl; // random
}

TEST(compile_time, 1) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto out1 =
        md::random::randint<value_t, md::random::generator::SplitMix64>(
            md::extents<index_t, 10>{}, 2, std::nullopt, std::nullopt,
            md::random::seed_t{0});
    constexpr auto out2 =
        md::random::randint<value_t, md::random::generator::SplitMix64>(
            md::extents<index_t, 10>{}, 1, std::nullopt, std::nullopt,
            md::random::seed_t{0});

    static_assert(
        md::is_same_extents(out1.extents(), md::extents<index_t, 10>{}));
    static_assert(
        md::is_same_extents(out2.extents(), md::extents<index_t, 10>{}));

    std::cout << "out1: " << md::to_string(out1) << std::endl; // random
    std::cout << "out2: " << md::to_string(out2) << std::endl; // bounded to 0

    static_assert(md::array_equiv(out2, 0));
}

TEST(compile_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto out =
        md::random::randint<value_t, md::random::generator::SplitMix64>(
            md::extents<index_t, 2, 4>{}, 5, std::nullopt, std::nullopt,
            md::random::seed_t{0});

    static_assert(
        md::is_same_extents(out.extents(), md::extents<index_t, 2, 4>{}));

    std::cout << "out: " << md::to_string(out) << std::endl; // random
}

TEST(compile_time, 3) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto out =
        md::random::randint<value_t, md::random::generator::SplitMix64>(
            md::extents<index_t>{}, 1,
            md::container<value_t, md::extents<index_t, 3>>{{3, 5, 10}},
            std::nullopt, md::random::seed_t{0});

    static_assert(
        md::is_same_extents(out.extents(), md::extents<index_t, 3>{}));

    std::cout << "out: " << md::to_string(out) << std::endl; // random
}

TEST(compile_time, 4) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto out =
        md::random::randint<value_t, md::random::generator::SplitMix64>(
            md::extents<index_t>{},
            md::container<value_t, md::extents<index_t, 3>>{{1, 5, 7}}, 10,
            std::nullopt, md::random::seed_t{0});

    static_assert(
        md::is_same_extents(out.extents(), md::extents<index_t, 3>{}));

    std::cout << "out: " << md::to_string(out) << std::endl; // random
}

TEST(compile_time, 5) {
    using value_t = uint8_t;
    using index_t = std::size_t;

    constexpr auto out =
        md::random::randint<value_t, md::random::generator::SplitMix64>(
            md::extents<index_t>{},
            md::container<value_t, md::extents<index_t, 4>>{{1, 3, 5, 7}},
            md::container<value_t, md::extents<index_t, 2, 1>>{{10, 20}},
            std::nullopt, md::random::seed_t{0});

    static_assert(
        md::is_same_extents(out.extents(), md::extents<index_t, 2, 4>{}));

    std::cout << "out: " << md::to_string(out) << std::endl; // random
}
