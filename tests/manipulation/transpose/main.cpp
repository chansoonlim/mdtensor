#include <gtest/gtest.h>

#include "mdtensor/creation/arange.hpp"
#include "mdtensor/creation/ones.hpp"
#include "mdtensor/logic/array_equal.hpp"
#include "mdtensor/manipulation/reshape.hpp"
#include "mdtensor/manipulation/transpose.hpp"

namespace md = mdtensor;

TEST(test, 1) {
    using T = double;

    constexpr auto a = md::mdarray<T, md::extents<size_t, 2, 2>>{
        {1, 2, 3, 4}, md::extents<size_t, 2, 2>{}};

    static_assert(md::array_equal(
        md::transpose(a), md::mdarray<T, md::extents<size_t, 2, 2>>{
                              {1, 3, 2, 4}, md::extents<size_t, 2, 2>{}}));
}

TEST(test, 2) {
    using T = double;

    constexpr auto a = md::mdarray<T, md::extents<size_t, 4>>{
        {1, 2, 3, 4}, md::extents<size_t, 4>{}};

    static_assert(md::array_equal(md::transpose(a), a));
}

TEST(test, 3) {
    using T = double;

    constexpr auto a = md::ones<T>(md::extents<size_t, 1, 2, 3>{});

    static_assert(md::core::same(
        md::transpose(a, std::integer_sequence<size_t, 1, 0, 2>{}).extents(),
        md::extents<size_t, 2, 1, 3>{}));
}

TEST(test, 4) {
    using T = double;

    constexpr auto a = md::ones<T>(md::extents<size_t, 2, 3, 4, 5>{});

    static_assert(md::core::same(md::transpose(a).extents(),
                                 md::extents<size_t, 5, 4, 3, 2>{}));
}

TEST(test, 5) {
    using T = double;

    const auto a = md::arange<T>(60);

    ASSERT_TRUE(md::core::same(
        md::transpose(md::reshape(a, md::extents<size_t, 3, 4, 5>{}),
                      std::integer_sequence<int, -1, 0, -2>{})
            .extents(),
        md::extents<size_t, 5, 3, 4>{}));
}
