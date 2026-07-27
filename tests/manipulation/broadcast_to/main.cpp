#include <gtest/gtest.h>

#include "mdtensor/logic/array_equal.hpp"
#include "mdtensor/manipulation/broadcast_to.hpp"

namespace md = mdtensor;

TEST(test, 1) {
    using T = double;

    constexpr auto a = md::mdarray<T, md::extents<size_t, 3>>{{1, 2, 3}};

    static_assert(
        md::array_equal(md::broadcast_to(a, md::extents<size_t, 3, 3>{}),
                        md::mdarray<T, md::extents<size_t, 3, 3>>{
                            {1, 2, 3, 1, 2, 3, 1, 2, 3}}));
}
