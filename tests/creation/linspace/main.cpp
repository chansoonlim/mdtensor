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
    const auto out = md::linspace(5, 2.0, 3.0, true);

    EXPECT_TRUE(
        md::allclose(out, md::tensor<float, md::extents<std::size_t, 5>>{
                              {2, 2.25, 2.5, 2.75, 3}}));

    std::cout << "out: " << md::to_string(out) << std::endl;
}

TEST(run_time, 2) {
    const auto out = md::linspace(5, 2.0, 3.0, false);

    EXPECT_TRUE(
        md::allclose(out, md::tensor<float, md::extents<std::size_t, 5>>{
                              {2, 2.2, 2.4, 2.6, 2.8}}));

    std::cout << "out: " << md::to_string(out) << std::endl;
}

TEST(compile_time, 1) {
    constexpr auto out = md::linspace<5>(2.0, 3.0, true);

    static_assert(
        md::allclose(out, md::tensor<float, md::extents<std::size_t, 5>>{
                              {2, 2.25, 2.5, 2.75, 3}}));

    std::cout << "out: " << md::to_string(out) << std::endl;
}

TEST(compile_time, 2) {
    constexpr auto out = md::linspace<5>(2.0, 3.0, false);

    static_assert(
        md::allclose(out, md::tensor<float, md::extents<std::size_t, 5>>{
                              {2, 2.2, 2.4, 2.6, 2.8}}));

    std::cout << "out: " << md::to_string(out) << std::endl;
}

TEST(compile_time, 3) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr index_t ext1 = 2;
    constexpr index_t ext2 = 3;
    constexpr index_t num = 10;

    constexpr auto start =
        md::random::uniform<value_t, md::random::generator::SplitMix64>(
            md::extents<index_t, ext1, ext2>{}, 0, 1, std::nullopt,
            md::random::seed_t{0});
    constexpr auto stop =
        md::random::uniform<value_t, md::random::generator::SplitMix64>(
            md::extents<index_t, ext1, ext2>{}, 2, 3, std::nullopt,
            md::random::seed_t{1});

    std::cout << "start: " << md::to_string(start) << std::endl;
    std::cout << "stop: " << md::to_string(stop) << std::endl;

    constexpr auto out1 = md::linspace<num, 0>(start, stop, true);
    constexpr auto out2 = md::linspace<num, 1>(start, stop, true);
    constexpr auto out3 = md::linspace<num, 2>(start, stop, true);
    constexpr auto out4 = md::linspace<num, -1>(start, stop, true);
    constexpr auto out5 = md::linspace<num, -2>(start, stop, true);
    constexpr auto out6 = md::linspace<num, -3>(start, stop, true);

    static_assert(md::is_same_extents(out1.extents(),
                                      md::extents<index_t, num, ext1, ext2>{}));
    static_assert(md::array_equal(
        md::submdspan(out1, 0, md::full_extent, md::full_extent), start));
    static_assert(md::array_equal(
        md::submdspan(out1, num - 1, md::full_extent, md::full_extent), stop));

    static_assert(md::is_same_extents(out2.extents(),
                                      md::extents<index_t, ext1, num, ext2>{}));
    static_assert(md::array_equal(
        md::submdspan(out2, md::full_extent, 0, md::full_extent), start));
    static_assert(md::array_equal(
        md::submdspan(out2, md::full_extent, num - 1, md::full_extent), stop));

    static_assert(md::is_same_extents(out3.extents(),
                                      md::extents<index_t, ext1, ext2, num>{}));
    static_assert(md::array_equal(
        md::submdspan(out3, md::full_extent, md::full_extent, 0), start));
    static_assert(md::array_equal(
        md::submdspan(out3, md::full_extent, md::full_extent, num - 1), stop));

    static_assert(md::array_equal(out3, out4));
    static_assert(md::array_equal(out2, out5));
    static_assert(md::array_equal(out1, out6));
}
