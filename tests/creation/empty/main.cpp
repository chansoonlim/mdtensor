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

    const auto x = md::empty(md::dims<2>{2, 2});

    static_assert(std::is_same_v<typename decltype(x)::value_type, value_t>);
}

TEST(run_time, 2) {
    using value_t = int;

    const auto x = md::empty<value_t>(md::dims<2>{2, 2});

    static_assert(std::is_same_v<typename decltype(x)::value_type, value_t>);
}

TEST(compile_time, 1) {
    using value_t = double;
    using index_t = std::size_t;

    constexpr auto x = md::empty(md::extents<index_t, 2, 2>{});

    static_assert(std::is_same_v<typename decltype(x)::value_type, value_t>);
}

TEST(compile_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    constexpr auto x = md::empty<value_t>(md::extents<index_t, 2, 2>{});

    static_assert(std::is_same_v<typename decltype(x)::value_type, value_t>);
}
