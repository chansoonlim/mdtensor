#include <gtest/gtest.h>

#include "mdtensor/logic/any.hpp"
#include "mdtensor/logic/array_equal.hpp"

namespace md = mdtensor;

TEST(test, 1) {
    static_assert(
        md::array_equal(md::any(md::mdarray<int8_t, md::extents<uint8_t, 2, 2>>{
                            {true, false, true, false}}),
                        true));

    static_assert(md::array_equal(
        md::any<0>(md::mdarray<int8_t, md::extents<uint8_t, 2, 3>>{
            {true, false, true, false, false, false}}),
        md::mdarray<int8_t, md::extents<uint8_t, 3>>{{true, false, true}}));

    static_assert(md::array_equal(
        md::any(md::mdarray<int8_t, md::extents<uint8_t, 3>>{{-1, 0, 5}}),
        true));
}

TEST(test, 2) {
    using T = int8_t;

    constexpr auto a =
        md::mdarray<T, md::extents<uint8_t, 3, 3>>{{1, 0, 0, 0, 0, 1, 0, 0, 0}};

    static_assert(md::array_equal(
        md::any<0>(a),
        md::mdarray<T, md::extents<uint8_t, 3>>{{true, false, true}}));

    static_assert(md::array_equal(
        md::any<1>(a),
        md::mdarray<T, md::extents<uint8_t, 3>>{{true, true, false}}));
}
