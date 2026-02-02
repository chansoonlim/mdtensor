#include <gtest/gtest.h>

#include "mdtensor/creation/ones.hpp"
#include "mdtensor/logic/array_equal.hpp"

namespace md = mdtensor;

TEST(test, 1) {
    using T = int;

    ASSERT_TRUE(
        md::array_equal(md::ones<T>(5), md::mdarray<T, md::extents<size_t, 5>>{
                                            std::array<T, 5>{1, 1, 1, 1, 1}}));
}

TEST(test, 2) {
    using T = int;

    static_assert(md::array_equal(md::ones<T>(md::extents<size_t, 5>{5}),
                                  md::mdarray<T, md::extents<size_t, 5>>{
                                      std::array<T, 5>{1, 1, 1, 1, 1}}));
}

TEST(test, 3) {
    using T = int;

    static_assert(md::array_equal(
        md::ones<T>(md::extents<size_t, 2, 1>{2, 1}),
        md::mdarray<T, md::extents<size_t, 2, 1>>{std::array<T, 2>{1, 1}}));
}

TEST(test, 4) {
    using T = int;

    static_assert(md::array_equal(md::ones<T>(md::extents<size_t, 2, 2>{2, 2}),
                                  md::mdarray<T, md::extents<size_t, 2, 2>>{
                                      std::array<T, 4>{1, 1, 1, 1}}));
}
