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

template <typename... Ts> using test_t = typename md::core::mdspan<Ts...>;

template <typename... Ts>
constexpr bool assigned = requires { typename test_t<Ts...>; };

TEST(test, 1) {
    using value_t = int;
    using index_t = std::size_t;

    static_assert(assigned<value_t, md::core::extents<index_t, 2>>);
    static_assert(assigned<value_t *, md::core::extents<index_t, 2>>);
    static_assert(assigned<value_t &, md::core::extents<index_t, 2>>);

    static_assert(
        assigned<std::optional<value_t>, md::core::extents<index_t, 2>>);

    static_assert(assigned<void, md::core::extents<index_t, 2>>);
    static_assert(assigned<std::nullopt_t, md::core::extents<index_t, 2>>);
}
