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

TEST(run_time, 1) {
    using value_t = double;

    auto x = md::container<value_t, md::dims<1>>{{1, 2, 3}, md::dims<1>{3}};
    const auto y = md::to_mdspan(x);
    auto z = md::copy(x);

    x(0) = 10;

    EXPECT_EQ(x(0), y(0));
    EXPECT_NE(x(0), z(0));
}

TEST(run_time, 2) {
    using value_t = double;

    const auto x =
        md::container<value_t, md::dims<1>>{{1, 2, 3}, md::dims<1>{3}};
    auto z = md::copy(x);

    static_assert(std::is_const_v<decltype(x)>);
    static_assert(!std::is_const_v<decltype(z)>);
}

TEST(compile_time, 1) {
    using value_t = double;
    using index_t = std::size_t;

    static_assert([] {
        auto x = md::container<value_t, md::extents<index_t, 3>>{{1, 2, 3}};
        const auto y = md::to_mdspan(x);
        auto z = md::copy(x);

        x(0) = 10;

        if (x(0) != y(0)) {
            return false;
        }

        if (x(0) == z(0)) {
            return false;
        }

        return true;
    }());
}

TEST(run_time, 3) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto x =
        md::container<value_t, md::extents<index_t, 3>>{{1, 2, 3}};
    auto z = md::copy(x);

    static_assert(std::is_const_v<decltype(x)>);
    static_assert(!std::is_const_v<decltype(z)>);
}
