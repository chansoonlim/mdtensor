#include <gtest/gtest.h>

#include "mdtensor/creation/full.hpp"
#include "mdtensor/logic/array_equal.hpp"

namespace md = mdtensor;

TEST(stack, full) {
    using T = double;

    constexpr auto a = md::full<T>(1, md::extents<size_t, 2, 1, 2>{});
    constexpr auto a_expect =
        md::mdarray<T, md::extents<size_t, 2, 1, 2>>{{1, 1, 1, 1}};

    constexpr bool array_equal = md::array_equal(a, a_expect);

    ASSERT_TRUE(array_equal);
}

TEST(heap, full) {
    using T = double;

    const auto a = md::full<T>(1, md::dims<3>{2, 1, 2});
    const auto a_expect =
        md::mdarray<T, md::dims<3>>{{1, 1, 1, 1}, md::dims<3>{2, 1, 2}};

    const bool array_equal = md::array_equal(a, a_expect);

    ASSERT_TRUE(array_equal);
}
