#include <gtest/gtest.h>

#include "mdtensor/logic/array_equal.hpp"
#include "mdtensor/logic/greater.hpp"

namespace md = mdtensor;

TEST(test, 1) {
    using T = double;

    static_assert(md::array_equal(
        md::greater(md::mdarray<T, md::extents<size_t, 2>>{{4, 2}},
                    md::mdarray<T, md::extents<size_t, 2>>{{2, 2}}),
        md::mdarray<uint8_t, md::extents<size_t, 2>>{{true, false}}));
}
