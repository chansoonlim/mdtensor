#include <numbers>

#include <gtest/gtest.h>

#include "mdtensor/logic/allclose.hpp"
#include "mdtensor/math/deg2rad.hpp"

namespace md = mdtensor;

TEST(single, deg2rad) {
    using T = double;

    constexpr T a = 90;
    constexpr T b = md::deg2rad(a);

    constexpr T b_expect = std::numbers::pi_v<T> / 2.;

    constexpr bool allclose = md::allclose(b, b_expect);

    ASSERT_TRUE(allclose);
}

TEST(stack, deg2rad) {
    using T = double;

    constexpr auto a = md::mdarray<T, md::extents<size_t, 3>>{{0, 90, 180}};
    constexpr auto b = md::deg2rad(a);

    constexpr auto b_expect = md::mdarray<T, md::extents<size_t, 3>>{
        {0, std::numbers::pi_v<T> / 2., std::numbers::pi_v<T>}};

    constexpr bool allclose = md::allclose(b, b_expect);

    ASSERT_TRUE(allclose);
}

TEST(heap, deg2rad) {
    using T = double;

    const auto a = md::mdarray<T, md::dims<1>>{{0, 90, 180}, md::dims<1>{3}};
    const auto b = md::deg2rad(a);

    const auto b_expect = md::mdarray<T, md::dims<1>>{
        {0, std::numbers::pi_v<T> / 2., std::numbers::pi_v<T>}, md::dims<1>{3}};

    const bool allclose = md::allclose(b, b_expect);

    ASSERT_TRUE(allclose);
}
