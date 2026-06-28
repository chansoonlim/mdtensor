#include <gtest/gtest.h>

#include "mdtensor/logic/allclose.hpp"
#include "mdtensor/math/sqrt.hpp"

namespace md = mdtensor;

TEST(test, 1) {
    using T = double;

    static_assert(md::allclose(
        md::sqrt(md::mdarray<T, md::extents<size_t, 3>>{{1, 4, 9}}),
        md::mdarray<T, md::extents<size_t, 3>>{{1, 2, 3}}));
}
