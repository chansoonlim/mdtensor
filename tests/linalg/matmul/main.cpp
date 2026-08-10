/**
 * @file
 * @brief test
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#include <gtest/gtest.h>

#ifdef MDTENSOR_USE_SINGLE_HEADER // for single header include
#include "mdtensor.hpp"
#else
#include "mdtensor/mdtensor.hpp"
#endif

namespace md = mdtensor;

TEST(run_time, 1) {
    using index_t = std::size_t;

    const auto a = md::ones(md::extents<index_t, 9, 5, 7, 4>{});
    const auto b = md::ones(md::extents<index_t, 9, 5, 4, 3>{});

    const auto c = md::linalg::matmul(a, b);

    EXPECT_TRUE(
        md::is_same_extents(c.extents(), md::extents<index_t, 9, 5, 7, 3>{}));
}

TEST(run_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    const auto a =
        md::container<value_t, md::extents<index_t, 2, 2>>{{1, 0, 0, 1}};
    const auto b =
        md::container<value_t, md::extents<index_t, 2, 2>>{{4, 1, 2, 2}};

    EXPECT_TRUE(md::allclose(
        md::linalg::matmul(a, b),
        md::container<value_t, md::extents<index_t, 2, 2>>{{4, 1, 2, 2}}));
}

TEST(run_time, 3) {
    using value_t = double;
    using index_t = std::size_t;

    const auto a =
        md::container<value_t, md::extents<index_t, 2, 2>>{{1, 0, 0, 1}};
    const auto b = md::container<value_t, md::extents<index_t, 2>>{{1, 2}};

    EXPECT_TRUE(
        md::allclose(md::linalg::matmul(a, b),
                     md::container<value_t, md::extents<index_t, 2>>{{1, 2}}));

    EXPECT_TRUE(
        md::allclose(md::linalg::matmul(b, a),
                     md::container<value_t, md::extents<index_t, 2>>{{1, 2}}));
}

TEST(run_time, 4) {
    using index_t = std::size_t;

    const auto data = md::arange(2 * 2 * 4);
    const auto a = md::reshape(data, md::extents<index_t, 2, 2, 4>{});
    const auto b = md::reshape(data, md::extents<index_t, 2, 4, 2>{});
    const auto c = md::linalg::matmul(a, b);

    EXPECT_TRUE(
        md::is_same_extents(c.extents(), md::extents<index_t, 2, 2, 2>{}));

    EXPECT_EQ(c(0, 1, 1), 98);
    EXPECT_EQ(md::sum(md::multiply(md::submdspan(a, 0, 1, md::full_extent),
                                   md::submdspan(b, 0, md::full_extent, 1))),
              98);
}

TEST(compile_time, 1) {
    using index_t = std::size_t;

    constexpr auto a = md::ones(md::extents<index_t, 9, 5, 7, 4>{});
    constexpr auto b = md::ones(md::extents<index_t, 9, 5, 4, 3>{});

    constexpr auto c = md::linalg::matmul(a, b);

    static_assert(
        md::is_same_extents(c.extents(), md::extents<index_t, 9, 5, 7, 3>{}));
}

TEST(compile_time, 2) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a =
        md::container<value_t, md::extents<index_t, 2, 2>>{{1, 0, 0, 1}};
    constexpr auto b =
        md::container<value_t, md::extents<index_t, 2, 2>>{{4, 1, 2, 2}};

    static_assert(md::allclose(
        md::linalg::matmul(a, b),
        md::container<value_t, md::extents<index_t, 2, 2>>{{4, 1, 2, 2}}));
}

TEST(compile_time, 3) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto a =
        md::container<value_t, md::extents<index_t, 2, 2>>{{1, 0, 0, 1}};
    constexpr auto b = md::container<value_t, md::extents<index_t, 2>>{{1, 2}};

    static_assert(
        md::allclose(md::linalg::matmul(a, b),
                     md::container<value_t, md::extents<index_t, 2>>{{1, 2}}));

    static_assert(
        md::allclose(md::linalg::matmul(b, a),
                     md::container<value_t, md::extents<index_t, 2>>{{1, 2}}));
}

TEST(compile_time, 4) {
    using index_t = std::size_t;

    constexpr auto data = md::arange<2 * 2 * 4>();

    static_assert([&]() {
        const auto a = md::reshape(data, md::extents<index_t, 2, 2, 4>{});
        const auto b = md::reshape(data, md::extents<index_t, 2, 4, 2>{});
        const auto c = md::linalg::matmul(a, b);

        if (!md::is_same_extents(c.extents(),
                                 md::extents<index_t, 2, 2, 2>{})) {
            return false;
        }

        if (c(0, 1, 1) != 98) {
            return false;
        }

        if (md::sum(md::multiply(md::submdspan(a, 0, 1, md::full_extent),
                                 md::submdspan(b, 0, md::full_extent, 1))) !=
            98) {
            return false;
        }

        return true;
    }());
}
