#include <gtest/gtest.h>

#include "mdtensor/logic/allclose.hpp"
#include "mdtensor/math/tan.hpp"

namespace md = mdtensor;

TEST(test, 1) {
    using T = double;

    static_assert(md::allclose(
        md::tan(
            md::mdarray<T, md::extents<size_t, 3>>{{-M_PI, M_PI / 2., M_PI}}),
        md::mdarray<T, md::extents<size_t, 3>>{{0, 16331239353195370, 0}}));
}
