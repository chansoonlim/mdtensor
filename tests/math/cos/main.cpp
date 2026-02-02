#include <gtest/gtest.h>

#include "mdtensor/logic/allclose.hpp"
#include "mdtensor/math/cos.hpp"

namespace md = mdtensor;

TEST(test, 1) {
    using T = double;

    static_assert(md::allclose(md::cos(md::mdarray<T, md::extents<size_t, 3>>{
                                   std::array<T, 3>{0, M_PI / 2., M_PI}}),
                               md::mdarray<T, md::extents<size_t, 3>>{
                                   std::array<T, 3>{1, 6.12303177e-17, -1}}));
}
