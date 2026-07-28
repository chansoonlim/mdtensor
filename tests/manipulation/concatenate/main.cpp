#include <gtest/gtest.h>

#include "mdtensor/logic/array_equal.hpp"
#include "mdtensor/manipulation/concatenate.hpp"

namespace md = mdtensor;

TEST(stack, 1) {
    using T = double;

    constexpr auto a =
        md::container<T, md::extents<size_t, 2, 2>>{{1, 2, 3, 4}};
    constexpr auto b = md::container<T, md::extents<size_t, 1, 2>>{{5, 6}};

    constexpr auto out = md::concatenate<0>(a, b);

    constexpr bool is_array_equal = md::array_equal(
        out, md::container<T, md::extents<size_t, 3, 2>>{{1, 2, 3, 4, 5, 6}});

    ASSERT_TRUE(is_array_equal);
}

TEST(stack, 2) {
    using T = double;

    const auto a =
        md::container<T, md::dims<2>>{{1, 2, 3, 4}, md::dims<2>{2, 2}};
    const auto b = md::container<T, md::dims<2>>{{5, 6}, md::dims<2>{1, 2}};

    const auto out = md::concatenate<0>(a, b);

    const bool is_array_equal = md::array_equal(
        out, md::container<T, md::extents<size_t, 3, 2>>{{1, 2, 3, 4, 5, 6}});

    ASSERT_TRUE(is_array_equal);
}
