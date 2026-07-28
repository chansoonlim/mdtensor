#include <gtest/gtest.h>

#include "mdtensor/creation/eye.hpp"
#include "mdtensor/logic/array_equal.hpp"
#include "mdtensor/math/maximum.hpp"

namespace md = mdtensor;

TEST(test, 1) {
    using T = double;

    static_assert(md::array_equal(
        md::maximum(md::container<T, md::extents<size_t, 3>>{{2, 3, 4}},
                    md::container<T, md::extents<size_t, 3>>{{1, 5, 2}}),
        md::container<T, md::extents<size_t, 3>>{{2, 5, 4}}));

    static_assert(md::array_equal(
        md::maximum(md::eye<T>(md::extents<size_t, 2, 2>{}),
                    md::container<T, md::extents<size_t, 2>>{{0.5, 2}}),
        md::container<T, md::extents<size_t, 2, 2>>{{1, 2, 0.5, 2}}));
}
