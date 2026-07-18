#include <gtest/gtest.h>

#include "mdtensor/mdtensor.hpp"

namespace md = mdtensor;

TEST(test, 1) {
    using value_t = double;
    using index_t = size_t;

    constexpr auto a =
        md::mdarray<value_t, md::extents<index_t, 2, 2>>{{1, 2, 3, 5}};
    constexpr auto b = md::mdarray<value_t, md::extents<index_t, 2>>{{1, 2}};
    constexpr auto x = std::get<0>(md::linalg::solve(a, b));

    static_assert(md::allclose(
        x, md::mdarray<value_t, md::extents<index_t, 2>>{{-1, 1}}));

    static_assert(md::allclose(md::matvec(a, x), b));
}

TEST(test, 2) {
    using value_t = double;
    using index_t = size_t;

    constexpr auto a =
        md::mdarray<value_t, md::extents<index_t, 2, 2>>{{1, 2, 3, 5}};
    constexpr auto b = md::mdarray<value_t, md::extents<index_t, 2, 1>>{{1, 2}};
    constexpr auto x = std::get<0>(md::linalg::solve(a, b));

    static_assert(md::allclose(
        x, md::mdarray<value_t, md::extents<index_t, 2, 1>>{{-1, 1}}));

    static_assert(md::allclose(md::matmul(a, x), b));
}
