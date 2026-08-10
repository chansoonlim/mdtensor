/**
 * @file
 * @brief test
 *
 * @copyright
 * SPDX-License-Identifier: Apache-2.0
 * See README and LICENSE files for full attribution details.
 */

#include <gtest/gtest.h>

#ifdef MDTENSOR_SINGLE_HEADER_INCLUDE_GUARD_ // for single header include
#include "mdtensor.hpp"
#else
#include "mdtensor/mdtensor.hpp"
#endif

namespace md = mdtensor;

template <typename... Ts> using test_t = typename md::core::mdarray<Ts...>;

template <typename... Ts>
constexpr bool assigned = requires { typename test_t<Ts...>; };

template <typename... Ts>
constexpr bool is_constexpr = std::is_trivially_copyable_v<test_t<Ts...>>;

TEST(test, constexpr) {
    using value_t = int;
    using index_t = std::size_t;

    static_assert(
        is_constexpr<value_t, md::core::extents<index_t, 2, 2>,
                     md::core::stdex::layout_right, std::array<value_t, 4>>);
    static_assert(
        is_constexpr<value_t, md::core::extents<index_t, 2, 2>,
                     md::core::stdex::layout_left, std::array<value_t, 4>>);

    static_assert(
        is_constexpr<value_t, md::core::dextents<index_t, 2>,
                     md::core::stdex::layout_right, std::array<value_t, 4>>);
    static_assert(
        is_constexpr<value_t, md::core::dextents<index_t, 2>,
                     md::core::stdex::layout_left, std::array<value_t, 4>>);

    static_assert(
        is_constexpr<value_t, md::core::dims<2>, md::core::stdex::layout_right,
                     std::array<value_t, 4>>);
    static_assert(
        is_constexpr<value_t, md::core::dims<2>, md::core::stdex::layout_left,
                     std::array<value_t, 4>>);

    static_assert(
        is_constexpr<value_t, md::core::extents<index_t, 2, 2>,
                     md::core::stdex::layout_right, std::span<value_t>>);

    static_assert(
        is_constexpr<value_t *, md::core::extents<index_t, 2, 2>,
                     md::core::stdex::layout_right, std::array<value_t *, 4>>);

    static_assert(
        is_constexpr<std::array<value_t, 2>, md::core::extents<index_t, 2, 2>,
                     md::core::stdex::layout_right,
                     std::array<std::array<value_t, 2>, 4>>);

    static_assert(
        is_constexpr<value_t, md::core::extents<index_t, 2, 2>,
                     md::core::stdex::layout_right, std::array<value_t, 6>>);
}

TEST(test, not_constexpr) {
    using value_t = int;
    using index_t = std::size_t;

    static_assert(
        !is_constexpr<value_t, md::core::extents<index_t, 2, 2>,
                      md::core::stdex::layout_right, std::vector<value_t>>);
    static_assert(
        !is_constexpr<value_t, md::core::extents<index_t, 2, 2>,
                      md::core::stdex::layout_left, std::vector<value_t>>);
    static_assert(!is_constexpr<value_t, md::core::extents<index_t>>);
    static_assert(!is_constexpr<value_t, md::core::extents<index_t, 2, 2>>);
    static_assert(!is_constexpr<value_t, md::core::dextents<index_t, 2>>);
}

TEST(test, assigned) {
    using value_t = int;
    using index_t = std::size_t;

    static_assert(assigned<value_t, md::core::extents<index_t>>);
    static_assert(assigned<value_t, md::core::extents<index_t, 2, 2>>);
    static_assert(
        assigned<value_t, md::core::extents<index_t, md::core::dyn, 2>>);
    static_assert(assigned<value_t, md::core::dextents<index_t, 2>>);
    static_assert(assigned<value_t, md::core::dims<3>>);
}
