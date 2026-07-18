#include <gtest/gtest.h>

#include "mdtensor/linalg/lu.hpp"
#include "mdtensor/linalg/matmul.hpp"
#include "mdtensor/logic/allclose.hpp"
#include "mdtensor/logic/array_equal.hpp"
#include "mdtensor/random/uniform.hpp"

namespace md = mdtensor;

TEST(test, 1) {
    using value_t = double;
    using index_t = uint8_t;

    constexpr auto a = md::mdarray<value_t, md::extents<index_t, 4, 4>>{
        {2, 5, 8, 7, 5, 2, 2, 8, 7, 5, 6, 6, 5, 4, 4, 8}};

    constexpr auto out1 = md::linalg::lu(a);
    constexpr auto p1 = std::get<0>(out1);
    constexpr auto l1 = std::get<1>(out1);
    constexpr auto u1 = std::get<2>(out1);

    static_assert(md::array_equal(
        p1, md::mdarray<index_t, md::extents<index_t, 4, 4>>{
                {0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0}}));

    constexpr auto out2 = md::linalg::lu<false, true>(a);
    constexpr auto p2 = std::get<0>(out2);

    static_assert(md::array_equal(
        p2, md::mdarray<index_t, md::extents<index_t, 4>>{{1, 3, 0, 2}}));

    constexpr auto pl = [&] {
        auto pl = md::mdarray<value_t, md::extents<index_t, 4, 4>>{};
        for (index_t i = 0; i < pl.extent(0); i++) {
            for (index_t j = 0; j < pl.extent(1); j++) {
                pl(i, j) = l1(p2(i), j);
            }
        }
        return pl;
    }();

    static_assert(md::allclose(a, md::linalg::matmul(pl, u1)));
}

TEST(test, 2) {
    using value_t = double;
    using index_t = uint8_t;

    constexpr auto a =
        md::random::uniform<value_t>(md::extents<index_t, 3, 2, 4, 8>{}, -4, 4);

    constexpr auto out1 = md::linalg::lu(a);
    constexpr auto p1 = std::get<0>(out1);
    constexpr auto l1 = std::get<1>(out1);
    constexpr auto u1 = std::get<2>(out1);

    static_assert(
        md::core::same(p1.extents(), md::extents<index_t, 3, 2, 4, 4>{}));
    static_assert(
        md::core::same(l1.extents(), md::extents<index_t, 3, 2, 4, 4>{}));
    static_assert(
        md::core::same(u1.extents(), md::extents<index_t, 3, 2, 4, 8>{}));

    static_assert(md::allclose(
        a, md::linalg::matmul(p1, md::linalg::matmul(l1, u1)), 0, 1e-5));

    constexpr auto out2 = md::linalg::lu<true>(a);
    constexpr auto pl2 = std::get<0>(out2);
    constexpr auto u2 = std::get<1>(out2);

    static_assert(md::allclose(a, md::linalg::matmul(pl2, u2), 0, 1e-5));
}
