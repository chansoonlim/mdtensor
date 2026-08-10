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

TEST(compile_time, 1) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a = md::container<value_t, md::extents<index_t, 4, 4>>{
        {2, 5, 8, 7, 5, 2, 2, 8, 7, 5, 6, 6, 5, 4, 4, 8}};

    constexpr auto out1 = md::linalg::lu(a);
    constexpr auto p1 = std::get<0>(out1);
    constexpr auto l1 = std::get<1>(out1);
    constexpr auto u1 = std::get<2>(out1);

    static_assert(md::array_equal(
        p1, md::container<index_t, md::extents<index_t, 4, 4>>{
                {0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0}}));

    constexpr auto out2 = md::linalg::lu<false, true>(a);
    constexpr auto p2 = std::get<0>(out2);

    static_assert(md::array_equal(
        p2, md::container<index_t, md::extents<index_t, 4>>{{1, 3, 0, 2}}));

    constexpr auto pl = [&] {
        auto pl = md::container<value_t, md::extents<index_t, 4, 4>>{};
        for (index_t i = 0; i < pl.extent(0); i++) {
            for (index_t j = 0; j < pl.extent(1); j++) {
                pl(i, j) = l1(p2(i), j);
            }
        }
        return pl;
    }();

    static_assert(md::allclose(a, md::linalg::matmul(pl, u1)));
}

TEST(compile_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a =
        md::random::uniform<value_t, md::random::generator::SplitMix64>(
            md::extents<index_t, 3, 2, 4, 8>{}, -4, 4, std::nullopt,
            md::random::seed_t{0});

    constexpr auto out1 = md::linalg::lu(a);
    constexpr auto p1 = std::get<0>(out1);
    constexpr auto l1 = std::get<1>(out1);
    constexpr auto u1 = std::get<2>(out1);

    static_assert(
        md::is_same_extents(p1.extents(), md::extents<index_t, 3, 2, 4, 4>{}));
    static_assert(
        md::is_same_extents(l1.extents(), md::extents<index_t, 3, 2, 4, 4>{}));
    static_assert(
        md::is_same_extents(u1.extents(), md::extents<index_t, 3, 2, 4, 8>{}));

    static_assert(md::allclose(
        a, md::linalg::matmul(p1, md::linalg::matmul(l1, u1)), 0, 1e-5));

    constexpr auto out2 = md::linalg::lu<true>(a);
    constexpr auto pl2 = std::get<0>(out2);
    constexpr auto u2 = std::get<1>(out2);

    static_assert(md::allclose(a, md::linalg::matmul(pl2, u2), 0, 1e-5));
}
