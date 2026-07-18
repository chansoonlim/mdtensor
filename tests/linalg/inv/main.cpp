#include <gtest/gtest.h>

#include "mdtensor/mdtensor.hpp"

namespace md = mdtensor;

TEST(test, 1) {
    using value_t = double;
    using index_t = size_t;

    constexpr auto a =
        md::mdarray<value_t, md::extents<index_t, 2, 2>>{{1, 2, 3, 4}};
    constexpr auto a_inv = std::get<0>(md::linalg::inv(a));

    static_assert(md::allclose(
        a_inv,
        md::mdarray<value_t, md::extents<index_t, 2, 2>>{{-2, 1, 1.5, -0.5}}));

    static_assert(md::allclose(md::linalg::matmul(a, a_inv),
                               md::eye<value_t>(md::extents<index_t, 2, 2>{})));

    static_assert(md::allclose(md::linalg::matmul(a_inv, a),
                               md::eye<value_t>(md::extents<index_t, 2, 2>{})));
}

TEST(test, 2) {
    using value_t = double;
    using index_t = size_t;

    constexpr auto a = md::mdarray<value_t, md::extents<index_t, 2, 2, 2>>{
        {1, 2, 3, 4, 1, 3, 3, 5}};
    constexpr auto a_inv = std::get<0>(md::linalg::inv(a));

    static_assert(
        md::allclose(a_inv, md::mdarray<value_t, md::extents<index_t, 2, 2, 2>>{
                                {-2, 1, 1.5, -0.5, -1.25, 0.75, 0.75, -0.25}}));
}

TEST(test, 3) {
    using value_t = double;
    using index_t = size_t;

    constexpr auto a = md::mdarray<value_t, md::extents<index_t, 3, 3>>{
        {2, 4, 6, 2, 0, 2, 6, 8, 14}};
    constexpr auto out = md::linalg::inv(a);

    constexpr auto a_inv = std::get<0>(out);
    constexpr auto valid = std::get<1>(out);

    // NOTE: if a matrix is close to singular, computed inverse may not satisfy
    // a * a_inv = I even if the valid flag is true.
    static_assert(valid == true);

    // static_assert(cond(a) > 1e10); // TODO: implement cond()
}
