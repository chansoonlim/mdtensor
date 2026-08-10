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
    using value_t = int;
    using index_t = std::size_t;

    const auto a = md::arange(10);
    const auto out1 = md::clip(a, 1, 8);
    const auto out2 = md::clip(a, 8, 1);

    std::cout << "out1: " << md::to_string(out1) << std::endl;
    std::cout << "out2: " << md::to_string(out2) << std::endl;

    ASSERT_TRUE(
        md::array_equal(out1, md::container<value_t, md::extents<index_t, 10>>{
                                  {1, 1, 2, 3, 4, 5, 6, 7, 8, 8}}));
    ASSERT_TRUE(
        md::array_equal(out2, md::container<value_t, md::extents<index_t, 10>>{
                                  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}));
}

TEST(run_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    auto a = md::arange(10);

    static_cast<void>(md::clip(a, 3, 6, a));

    std::cout << "a: " << md::to_string(a) << std::endl;

    ASSERT_TRUE(
        md::array_equal(a, md::container<value_t, md::extents<index_t, 10>>{
                               {3, 3, 3, 3, 4, 5, 6, 6, 6, 6}}));
}

TEST(run_time, 3) {
    using value_t = int;
    using index_t = std::size_t;

    const auto a = md::arange(10);
    const auto out = md::clip(a,
                              md::container<value_t, md::extents<index_t, 10>>{
                                  {3, 4, 1, 1, 1, 4, 4, 4, 4, 4}},
                              8);

    std::cout << "out: " << md::to_string(out) << std::endl;

    ASSERT_TRUE(
        md::array_equal(out, md::container<value_t, md::extents<index_t, 10>>{
                                 {3, 4, 2, 3, 4, 5, 6, 7, 8, 8}}));
}

TEST(compile_time, 1) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a = md::arange<10>();
    constexpr auto out1 = md::clip(a, 1, 8);
    constexpr auto out2 = md::clip(a, 8, 1);

    std::cout << "out1: " << md::to_string(out1) << std::endl;
    std::cout << "out2: " << md::to_string(out2) << std::endl;

    static_assert(
        md::array_equal(out1, md::container<value_t, md::extents<index_t, 10>>{
                                  {1, 1, 2, 3, 4, 5, 6, 7, 8, 8}}));
    static_assert(
        md::array_equal(out2, md::container<value_t, md::extents<index_t, 10>>{
                                  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}));
}

TEST(compile_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    static_assert([&]() {
        auto a = md::arange<10>();

        static_cast<void>(md::clip(a, 3, 6, a));

        return md::array_equal(a,
                               md::container<value_t, md::extents<index_t, 10>>{
                                   {3, 3, 3, 3, 4, 5, 6, 6, 6, 6}});
    }());
}

TEST(compile_time, 3) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto a = md::arange<10>();
    constexpr auto out =
        md::clip(a,
                 md::container<value_t, md::extents<index_t, 10>>{
                     {3, 4, 1, 1, 1, 4, 4, 4, 4, 4}},
                 8);

    std::cout << "out: " << md::to_string(out) << std::endl;

    static_assert(
        md::array_equal(out, md::container<value_t, md::extents<index_t, 10>>{
                                 {3, 4, 2, 3, 4, 5, 6, 7, 8, 8}}));
}
