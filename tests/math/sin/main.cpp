#include <numbers>

#include <gtest/gtest.h>

#include "mdtensor/logic/allclose.hpp"
#include "mdtensor/math/sin.hpp"

namespace md = mdtensor;

#ifdef REAL_GCC

TEST(stack, negative) {
    using T = float;

    constexpr auto a = md::mdarray<T, md::extents<size_t, 4>>{
        {0, std::numbers::pi_v<T> / 4., std::numbers::pi_v<T> / 2.,
         std::numbers::pi_v<T>}};
    constexpr auto b = md::sin(a);

    constexpr bool is_allclose =
        md::allclose(b,
                     md::mdarray<T, md::extents<size_t, 4>>{
                         {0.000000, 0.707107, 1.000000, -0.000000}},
                     0, 1e-5);

    ASSERT_TRUE(is_allclose);
}

#endif

TEST(heap, negative) {
    using T = double;

    const auto a = md::mdarray<T, md::dims<1>>{{0, std::numbers::pi_v<T> / 4.,
                                                std::numbers::pi_v<T> / 2.,
                                                std::numbers::pi_v<T>},
                                               md::dims<1>{4}};
    const auto b = md::sin(a);

    const bool is_allclose =
        md::allclose(b,
                     md::mdarray<T, md::extents<size_t, 4>>{
                         {0.000000, 0.707107, 1.000000, -0.000000}},
                     0, 1e-5);

    ASSERT_TRUE(is_allclose);
}
