#include <gtest/gtest.h>

#include "mdtensor/logic/array_equal.hpp"
#include "mdtensor/logic/greater_equal.hpp"

namespace md = mdtensor;

TEST(test, 1) {
    using T = double;

    static_assert(md::array_equal(
        md::greater_equal(md::container<T, md::extents<size_t, 3>>{{4, 2, 1}},
                          md::container<T, md::extents<size_t, 3>>{{2, 2, 2}}),
        md::container<uint8_t, md::extents<size_t, 3>>{{true, true, false}}));
}
