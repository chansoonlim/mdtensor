#include <gtest/gtest.h>

#include "mdtensor/creation/arange.hpp"
#include "mdtensor/creation/ones_like.hpp"
#include "mdtensor/logic/array_equal.hpp"
#include "mdtensor/manipulation/reshape.hpp"

namespace md = mdtensor;

TEST(test, 1) {
    const auto x = md::arange<int>(6);
    auto x_rs = md::reshape(x, md::extents<size_t, 2, 3>{2, 3});

    ASSERT_TRUE(md::array_equal(
        x_rs, md::mdarray<int, md::extents<size_t, 2, 3>>{{0, 1, 2, 3, 4, 5}}));

    ASSERT_TRUE(md::array_equal(
        md::ones_like(x_rs),
        md::mdarray<int, md::extents<size_t, 2, 3>>{{1, 1, 1, 1, 1, 1}}));
}
