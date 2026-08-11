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
    const auto out = md::arange<int>(0, 5, 0.5);

    EXPECT_TRUE(
        md::array_equal(out, md::tensor<int, md::extents<std::size_t, 10>>{
                                 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}));

    std::cout << "out: " << md::to_string(out) << std::endl;
}

TEST(run_time, 2) {
    const auto out = md::arange<int>(-3, 3, 0.5);

    EXPECT_TRUE(
        md::array_equal(out, md::tensor<int, md::extents<std::size_t, 12>>{
                                 {-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8}}));

    std::cout << "out: " << md::to_string(out) << std::endl;
}

TEST(run_time, 3) {
    const auto out = md::arange(3);

    static_assert(std::integral<typename decltype(out)::value_type>);

    EXPECT_TRUE(md::array_equal(
        out, md::tensor<int, md::extents<std::uint8_t, 3>>{{0, 1, 2}}));

    std::cout << "out: " << md::to_string(out) << std::endl;
}

TEST(run_time, 4) {
    const auto out = md::arange(3.0);

    static_assert(std::floating_point<typename decltype(out)::value_type>);

    EXPECT_TRUE(md::array_equal(
        out, md::tensor<float, md::extents<std::uint8_t, 3>>{{0, 1, 2}}));

    std::cout << "out: " << md::to_string(out) << std::endl;
}

TEST(run_time, 5) {
    const auto out = md::arange(3, 7);

    EXPECT_TRUE(md::array_equal(
        out, md::tensor<int, md::extents<std::uint8_t, 4>>{{3, 4, 5, 6}}));

    std::cout << "out: " << md::to_string(out) << std::endl;
}

TEST(run_time, 6) {
    const auto out = md::arange(3, 7, 2);

    EXPECT_TRUE(md::array_equal(
        out, md::tensor<int, md::extents<std::uint8_t, 2>>{{3, 5}}));

    std::cout << "out: " << md::to_string(out) << std::endl;
}

TEST(compile_time, 1) {
    constexpr auto out = md::arange<10, int>(0, 0.5);

    static_assert(
        md::array_equal(out, md::tensor<int, md::extents<std::size_t, 10>>{
                                 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}));

    std::cout << "out: " << md::to_string(out) << std::endl;
}

TEST(compile_time, 2) {
    constexpr auto out = md::arange<12, int>(-3, 0.5);

    static_assert(
        md::array_equal(out, md::tensor<int, md::extents<std::size_t, 12>>{
                                 {-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8}}));

    std::cout << "out: " << md::to_string(out) << std::endl;
}

TEST(compile_time, 3) {
    constexpr auto out = md::arange<3>();

    static_assert(std::integral<typename decltype(out)::value_type>);

    static_assert(md::array_equal(
        out, md::tensor<int, md::extents<std::uint8_t, 3>>{{0, 1, 2}}));

    std::cout << "out: " << md::to_string(out) << std::endl;
}

TEST(compile_time, 4) {
    constexpr auto out = md::arange<3>(0.0);

    static_assert(std::floating_point<typename decltype(out)::value_type>);

    static_assert(md::array_equal(
        out, md::tensor<float, md::extents<std::uint8_t, 3>>{{0, 1, 2}}));

    std::cout << "out: " << md::to_string(out) << std::endl;
}

TEST(compile_time, 5) {
    constexpr auto out = md::arange<4>(3);

    static_assert(md::array_equal(
        out, md::tensor<int, md::extents<std::uint8_t, 4>>{{3, 4, 5, 6}}));

    std::cout << "out: " << md::to_string(out) << std::endl;
}

TEST(compile_time, 6) {
    constexpr auto out = md::arange<2>(3, 2);

    static_assert(md::array_equal(
        out, md::tensor<int, md::extents<std::uint8_t, 2>>{{3, 5}}));

    std::cout << "out: " << md::to_string(out) << std::endl;
}
