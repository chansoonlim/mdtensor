#include <gtest/gtest.h>

#include "mdtensor/logic/allclose.hpp"
#include "mdtensor/math/absolute.hpp"

namespace md = mdtensor;

TEST(test, 1) {
    using T = double;

    constexpr auto x = md::mdarray<T, md::extents<size_t, 2>>{{-1.2, 1.2}};

    static_assert(md::allclose(
        md::absolute(x), md::mdarray<T, md::extents<size_t, 2>>{{1.2, 1.2}}));
}
