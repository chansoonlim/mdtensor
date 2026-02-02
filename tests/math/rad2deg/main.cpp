#include <numbers>

#include <gtest/gtest.h>

#include "mdtensor/logic/allclose.hpp"
#include "mdtensor/math/rad2deg.hpp"

namespace md = mdtensor;

TEST(single, deg2rad) {
    using T = double;

    constexpr T a = std::numbers::pi_v<T> / 2.;
    constexpr T b = md::rad2deg(a);

    constexpr T b_expect = 90;

    constexpr bool allclose = md::allclose(b, b_expect);

    ASSERT_TRUE(allclose);
}

TEST(stack, rad2deg) {
    using T = double;

    constexpr auto a = md::mdarray<T, md::extents<size_t, 3>>{
        std::array<T, 3>{0, std::numbers::pi_v<T> / 2., std::numbers::pi_v<T>}};
    constexpr auto b = md::rad2deg(a);

    constexpr auto b_expect =
        md::mdarray<T, md::extents<size_t, 3>>{std::array<T, 3>{0, 90, 180}};

    constexpr bool allclose = md::allclose(b, b_expect);

    ASSERT_TRUE(allclose);
}

TEST(heap, rad2deg) {
    using T = double;

    const auto a = md::mdarray<T, md::dims<1>>{
        std::vector<T>{0, std::numbers::pi_v<T> / 2., std::numbers::pi_v<T>},
        md::dims<1>{3}};
    const auto b = md::rad2deg(a);

    const auto b_expect =
        md::mdarray<T, md::dims<1>>{std::vector<T>{0, 90, 180}, md::dims<1>{3}};

    const bool allclose = md::allclose(b, b_expect);

    ASSERT_TRUE(allclose);
}
