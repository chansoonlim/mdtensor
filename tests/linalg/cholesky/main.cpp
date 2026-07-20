#include <gtest/gtest.h>

#include "mdtensor/linalg/cholesky.hpp"
#include "mdtensor/logic/allclose.hpp"

namespace md = mdtensor;

TEST(test, 1) {
    using value_t = double;
    using index_t = uint8_t;

    constexpr auto a = md::mdarray<value_t, md::extents<index_t, 4, 4>>{
        {6, 3, 4, 8, 3, 6, 5, 1, 4, 5, 10, 7, 8, 1, 7, 25}};

    constexpr auto out = md::linalg::cholesky(a, false);
    constexpr auto l = std::get<0>(out);
    constexpr auto valid = std::get<1>(out);

    static_assert(md::allclose(
        l, md::mdarray<value_t, md::extents<index_t, 4, 4>>{
               {2.449490, 0.000000, 0.000000, 0.000000, 1.224745, 2.121320,
                0.000000, 0.000000, 1.632993, 1.414214, 2.309401, 0.000000,
                3.265986, -1.414214, 1.587713, 3.132491}}));
    static_assert(valid == true);
}
