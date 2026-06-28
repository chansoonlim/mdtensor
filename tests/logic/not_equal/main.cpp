#include <gtest/gtest.h>

#include "mdtensor/logic/array_equal.hpp"
#include "mdtensor/logic/not_equal.hpp"

namespace md = mdtensor;

TEST(stack, not_equal) {
    using T = double;

    constexpr auto a =
        md::mdarray<T, md::extents<size_t, 2, 1, 2>>{{1, 2, 3, 4}};
    constexpr auto b = md::mdarray<T, md::extents<size_t, 2, 1>>{{2, 3}};
    constexpr auto c = md::not_equal(a, b);

    constexpr auto c_expect =
        md::mdarray<uint8_t, md::extents<size_t, 2, 2, 2>>{
            {1, 0, 1, 1, 1, 1, 0, 1}};

    constexpr auto is_array_equal = md::array_equal(c, c_expect);

    ASSERT_TRUE(is_array_equal);
}

TEST(heap, not_equal) {
    using T = double;

    const auto a =
        md::mdarray<T, md::dims<3>>{{1, 2, 3, 4}, md::dims<3>{2, 1, 2}};
    const auto b = md::mdarray<T, md::dims<2>>{{2, 3}, md::dims<2>{2, 1}};
    const auto c = md::not_equal(a, b);

    const auto c_expect = md::mdarray<uint8_t, md::dims<3>>{
        {1, 0, 1, 1, 1, 1, 0, 1}, md::dims<3>{2, 2, 2}};

    const auto is_array_equal = md::array_equal(c, c_expect);

    ASSERT_TRUE(is_array_equal);
}
