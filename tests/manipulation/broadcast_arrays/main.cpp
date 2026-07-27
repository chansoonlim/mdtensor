#include <gtest/gtest.h>

#include "mdtensor/logic/array_equal.hpp"
#include "mdtensor/manipulation/broadcast_arrays.hpp"

namespace md = mdtensor;

TEST(test, 1) {
    using T = double;

    constexpr auto x = md::mdarray<T, md::extents<size_t, 1, 3>>{{1, 2, 3}};
    constexpr auto y = md::mdarray<T, md::extents<size_t, 2, 1>>{{4, 5}};

    static_assert([&]() {
        const auto out = md::broadcast_arrays(x, y);
        const auto x_bcast = std::get<0>(out);
        const auto y_bcast = std::get<1>(out);

        return md::array_equal(x_bcast,
                               md::mdarray<T, md::extents<size_t, 2, 3>>{
                                   {1, 2, 3, 1, 2, 3}}) &&
               md::array_equal(y_bcast,
                               md::mdarray<T, md::extents<size_t, 2, 3>>{
                                   {4, 4, 4, 5, 5, 5}});
    }());
}
