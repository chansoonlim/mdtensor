#include <gtest/gtest.h>

#include "mdtensor/linalg/norm.hpp"
#include "mdtensor/logic/allclose.hpp"

namespace md = mdtensor;

TEST(stack, norm) {
    using T = double;

    constexpr auto x =
        md::mdarray<T, md::extents<size_t, 2>>{std::array<T, 2>{3, 4}};
    constexpr auto x_norm = md::linalg::norm(x);

    constexpr auto x_norm_expect = 5;

    constexpr bool allclose = md::allclose(x_norm, x_norm_expect);

    ASSERT_TRUE(allclose);
}

TEST(heap, norm) {
    using T = double;

    const auto x =
        md::mdarray<T, md::dims<1>>{std::vector<T>{3, 4}, md::dims<1>{2}};
    const auto x_norm = md::linalg::norm(x);

    const auto x_norm_expect = 5;

    const bool allclose = md::allclose(x_norm, x_norm_expect);

    ASSERT_TRUE(allclose);
}

TEST(stack, zero) {
    using T = double;

    constexpr auto x =
        md::mdarray<T, md::extents<size_t, 2>>{std::array<T, 2>{0, 0}};
    constexpr auto x_norm = md::linalg::norm(x);

    constexpr auto x_norm_expect = 0;

    constexpr bool allclose = md::allclose(x_norm, x_norm_expect);

    ASSERT_TRUE(allclose);
}

TEST(heap, zero) {
    using T = double;

    const auto x =
        md::mdarray<T, md::dims<1>>{std::vector<T>{0, 0}, md::dims<1>{2}};
    const auto x_norm = md::linalg::norm(x);

    const auto x_norm_expect = 0;

    const bool allclose = md::allclose(x_norm, x_norm_expect);

    ASSERT_TRUE(allclose);
}
