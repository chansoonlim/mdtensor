#include <gtest/gtest.h>

#include "mdtensor/logic/array_equiv.hpp"

namespace md = mdtensor;

TEST(stack, array_equiv) {
    using T = double;

    constexpr auto a = md::container<T, md::extents<size_t, 2>>{{1, 2}};
    constexpr auto b =
        md::container<T, md::extents<size_t, 2, 2>>{{1, 2, 1, 2}};
    constexpr auto is_array_equiv = md::array_equiv(a, b);

    ASSERT_TRUE(is_array_equiv);
}

TEST(heap, array_equiv) {
    using T = double;

    const auto a = md::container<T, md::dims<1>>{{1, 2}, md::dims<1>{2}};
    const auto b =
        md::container<T, md::dims<2>>{{1, 2, 1, 2}, md::dims<2>{2, 2}};
    const auto is_array_equiv = md::array_equiv(a, b);

    ASSERT_TRUE(is_array_equiv);
}
