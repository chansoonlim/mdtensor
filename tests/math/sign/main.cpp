#include <gtest/gtest.h>

#include "mdtensor/logic/array_equal.hpp"
#include "mdtensor/math/sign.hpp"

namespace md = mdtensor;

TEST(test, 1) {
    using T = double;

    static_assert(md::array_equal(
        md::sign(md::mdarray<T, md::extents<size_t, 2>>{{-5, 4.5}}),
        md::mdarray<int8_t, md::extents<size_t, 2>>{{-1, 1}}));

    static_assert(md::array_equal(md::sign(0), 0));

    static_assert(md::array_equal(md::sign(2), 1));
    static_assert(md::array_equal(md::sign(-3), -1));
}
