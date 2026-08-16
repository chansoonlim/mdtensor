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
    using value_t = int;

    const value_t a = 2;

    std::cout << "a : " << md::to_string(a) << std::endl;

    EXPECT_EQ(md::to_string(a), "2");
}

TEST(run_time, 2) {
    using value_t = int;
    using index_t = std::size_t;

    const value_t value = 2;

    const auto a = md::mdspan<const value_t, md::extents<index_t>>{&value};

    std::cout << "a : " << md::to_string(a) << std::endl;
    std::cout << "a extents : " << md::to_string(a.extents()) << std::endl;
    std::cout << "a rank : " << a.rank() << std::endl;

    EXPECT_EQ(md::to_string(a), "2");
    EXPECT_EQ(md::to_string(a.extents()), "()");
    EXPECT_EQ(a.rank(), 0);
}

TEST(run_time, 3) {
    using value_t = int;
    using index_t = std::size_t;

    const value_t value = 2;

    const auto a = md::mdspan<const value_t, md::extents<index_t, 1>>{&value};

    std::cout << "a : " << md::to_string(a) << std::endl;
    std::cout << "a extents : " << md::to_string(a.extents()) << std::endl;
    std::cout << "a rank : " << a.rank() << std::endl;

    EXPECT_EQ(md::to_string(a), "[2]");
    EXPECT_EQ(md::to_string(a.extents()), "(1,)");
    EXPECT_EQ(a.rank(), 1);
}

TEST(run_time, 4) {
    const auto a =
        md::tensor<int, md::extents<std::size_t, 2, 1, 2>>{{1, 2, 3, 4}};

    std::cout << "a : " << md::to_string(a) << std::endl;
    std::cout << "a extents : " << md::to_string(a.extents()) << std::endl;
    std::cout << "a rank : " << a.rank() << std::endl;

    EXPECT_EQ(md::to_string(a), "[[[1, 2]], [[3, 4]]]");
    EXPECT_EQ(md::to_string(a.extents()), "(2, 1, 2)");
    EXPECT_EQ(a.rank(), 3);
}
