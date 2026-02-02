#include <gtest/gtest.h>

#include "mdtensor/core/broadcast.hpp"

namespace md = mdtensor;

TEST(broadcast, 1) {
    auto exts1 = md::extents<size_t, 8, 6, 2, 3, 1, 4>{};

    auto bexts = md::core::broadcast(exts1);

    static_assert(md::core::same(bexts, exts1));
}

TEST(broadcast, 2) {
    auto exts1 = md::extents<size_t, 8, 6, 2, 3, 1, 4>{};
    auto exts2 = md::extents<size_t, 1, 3, 5, 4>{};

    auto bexts = md::core::broadcast(exts1, exts2);

    static_assert(
        md::core::same(bexts, md::extents<size_t, 8, 6, 2, 3, 5, 4>{}));
}

TEST(broadcast, 3) {
    auto exts1 = md::extents<size_t, 8, 6, 2, 3, 1, 4>{};
    auto exts2 = md::extents<size_t, 1, 3, 1, 4>{};
    auto exts3 = md::extents<size_t, 2, 1, 5, 1>{};

    auto bexts = md::core::broadcast(exts1, exts2, exts3);

    static_assert(
        md::core::same(bexts, md::extents<size_t, 8, 6, 2, 3, 5, 4>{}));
}

TEST(broadcast_to, 1) {
    auto mda = md::mdarray<int, md::extents<size_t, 1, 3, 2>>{};
    auto bexts = md::extents<size_t, 4, 5, 3>{};

    auto bcast_mda = md::core::broadcast_to<0, 3>(mda.to_mdspan(), bexts);

    static_assert(md::core::same(bcast_mda.extents(), bexts));
}

TEST(broadcast_to, 2) {
    auto mda = md::mdarray<int, md::extents<size_t, 4, 1, 3, 2>>{};
    auto bexts = md::extents<size_t, 4, 5, 3>{};

    auto bcast_mda = md::core::broadcast_to<0, 3>(mda.to_mdspan(), bexts);

    static_assert(
        md::core::same(bcast_mda.extents(), md::extents<size_t, 4, 5, 3, 2>{}));
}

TEST(broadcast_to, 3) {
    auto mda = md::mdarray<int, md::extents<size_t, 7, 4, 1, 3, 2>>{};
    auto bexts = md::extents<size_t, 4, 5, 3>{};

    auto bcast_mda = md::core::broadcast_to<1, 3>(mda.to_mdspan(), bexts);

    static_assert(md::core::same(bcast_mda.extents(),
                                 md::extents<size_t, 7, 4, 5, 3, 2>{}));
}
