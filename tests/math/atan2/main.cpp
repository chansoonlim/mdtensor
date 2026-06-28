#include <gtest/gtest.h>

#include "mdtensor/logic/allclose.hpp"
#include "mdtensor/math/atan2.hpp"
#include "mdtensor/math/rad2deg.hpp"

namespace md = mdtensor;

TEST(test, 1) {
    using T = double;

    constexpr auto x = md::mdarray<T, md::extents<size_t, 4>>{{-1, 1, 1, -1}};
    constexpr auto y = md::mdarray<T, md::extents<size_t, 4>>{{-1, -1, 1, 1}};

    static_assert(md::allclose(
        md::rad2deg(md::atan2(y, x)),
        md::mdarray<T, md::extents<size_t, 4>>{{-135, -45, 45, 135}}));
}
