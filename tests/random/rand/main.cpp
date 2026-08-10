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

    const auto out = md::random::rand<value_t>(md::dims<2>{3, 2});

    EXPECT_TRUE(
        md::is_same_extents(out.extents(), md::extents<index_t, 3, 2>{}));
    EXPECT_TRUE(!md::array_equiv(out, out(0, 0)));

    // NOTE: out changes every runtime.
    std::cout << "out: " << md::to_string(out) << std::endl;
}

TEST(run_time, 2) {
    using value_t = double;

    const auto out1 = md::random::rand<value_t>(md::dims<2>{3, 2});
    const auto out2 = md::random::rand<value_t>(md::dims<2>{3, 2});

    std::cout << "out1: " << md::to_string(out1) << std::endl;
    std::cout << "out2: " << md::to_string(out2) << std::endl;

    EXPECT_TRUE(!md::array_equal(out1, out2));
}

TEST(run_time, 3) {
    const auto out = md::random::rand();

    static_assert(std::is_floating_point_v<std::remove_cvref_t<decltype(out)>>);

    // NOTE: out changes every runtime.
    std::cout << md::to_string(out) << std::endl;
}

TEST(compile_time, 1) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto out =
        md::random::rand<value_t, md::random::generator::SplitMix64>(
            md::extents<index_t, 3, 2>{}, std::nullopt, md::random::seed_t{0});

    static_assert(
        md::is_same_extents(out.extents(), md::extents<index_t, 3, 2>{}));
    static_assert(!md::array_equiv(out, out(0, 0)));

    // NOTE: out looks like random, but is deterministic due to the fixed seed.
    std::cout << "out: " << md::to_string(out) << std::endl;
}

TEST(compile_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto out1 =
        md::random::rand<value_t, md::random::generator::SplitMix64>(
            md::extents<index_t, 3, 2>{}, std::nullopt, md::random::seed_t{0});
    constexpr auto out2 =
        md::random::rand<value_t, md::random::generator::SplitMix64>(
            md::extents<index_t, 3, 2>{}, std::nullopt, md::random::seed_t{0});
    constexpr auto out3 =
        md::random::rand<value_t, md::random::generator::SplitMix64>(
            md::extents<index_t, 3, 2>{}, std::nullopt, md::random::seed_t{1});

    std::cout << "out1: " << md::to_string(out1) << std::endl;
    std::cout << "out2: " << md::to_string(out2) << std::endl;
    std::cout << "out3: " << md::to_string(out3) << std::endl;

    static_assert(md::array_equal(out1, out2));
    static_assert(!md::array_equal(out1, out3));
}

TEST(compile_time, 3) {
    using value_t = double;

    const auto out =
        md::random::rand<value_t, md::random::generator::SplitMix64>(
            md::extents<std::uint8_t>{}, std::nullopt, md::random::seed_t{0});

    static_assert(std::is_floating_point_v<std::remove_cvref_t<decltype(out)>>);

    // NOTE: out looks like random, but is deterministic due to the fixed seed.
    std::cout << md::to_string(out) << std::endl;
}
