#include <gtest/gtest.h>

#include "mdtensor/logic/array_equal.hpp"
#include "mdtensor/manipulation/reshape.hpp"

namespace md = mdtensor;

TEST(stack, 1) {
    using T = double;

    constexpr auto in =
        md::mdarray<T, md::extents<size_t, 2, 3>>{{1, 2, 3, 4, 5, 6}};

    constexpr bool array_equal = md::array_equal(
        md::reshape(in, md::extents<size_t, 6>{}),
        md::mdarray<T, md::extents<size_t, 6>>{{1, 2, 3, 4, 5, 6}});

    ASSERT_TRUE(array_equal);
}

TEST(stack, 2) {
    using T = double;

    const auto in =
        md::mdarray<T, md::dims<2>>{{1, 2, 3, 4, 5, 6}, md::dims<2>{2, 3}};

    const bool array_equal = md::array_equal(
        md::reshape(in, md::extents<size_t, 6>{}),
        md::mdarray<T, md::dims<1>>{{1, 2, 3, 4, 5, 6}, md::dims<1>{6}});

    ASSERT_TRUE(array_equal);
}
