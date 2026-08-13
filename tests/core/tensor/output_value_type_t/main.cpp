
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

template <typename dtype, typename... Ts>
using test_t = md::core::calc_type_t<dtype, Ts...>;

TEST(test, assign) {
    static_assert(std::same_as<test_t<void, std::uint8_t>, std::uint8_t>);
    static_assert(std::same_as<test_t<void, std::uint16_t>, std::uint16_t>);
    static_assert(std::same_as<test_t<void, std::uint32_t>, std::uint32_t>);
    static_assert(std::same_as<test_t<void, std::uint64_t>, std::uint64_t>);
    static_assert(std::same_as<test_t<void, std::int8_t>, std::int8_t>);
    static_assert(std::same_as<test_t<void, std::int16_t>, std::int16_t>);
    static_assert(std::same_as<test_t<void, std::int32_t>, std::int32_t>);
    static_assert(std::same_as<test_t<void, std::int64_t>, std::int64_t>);
    static_assert(std::same_as<test_t<void, float>, float>);
    static_assert(std::same_as<test_t<void, double>, double>);
    static_assert(std::same_as<test_t<void, long double>, long double>);
}

TEST(test, not_assign) {
    // static_assert(
    //     !requires { std::same_as<test_t<void, std::nullopt_t>, void>; });
    // static_assert(std::same_as<test_t<void, std::array<int, 2>>, void>);
    // static_assert(std::same_as<test_t<void, std::vector<int>>, void>);
    // static_assert(std::same_as<test_t<void, std::tuple<int, double>>, void>);
}

TEST(test, unwrap_optional) {
    static_assert(std::same_as<test_t<void, std::optional<int>>, int>);
    static_assert(std::same_as<test_t<void, std::optional<float>>, float>);

    // static_assert(
    //     std::same_as<test_t<void, std::optional<std::nullopt_t>>, void>);
}

TEST(test, unwrap_mdspan) {
    static_assert(
        std::same_as<
            test_t<void, md::core::mdspan<int, md::core::extents<size_t, 1>>>,
            int>);
    static_assert(
        std::same_as<
            test_t<void, md::core::mdspan<float, md::core::extents<size_t, 1>>>,
            float>);

    // static_assert(std::same_as<
    //               test_t<void, md::core::mdspan<std::nullopt_t,
    //                                             md::core::extents<size_t,
    //                                             1>>>,
    //               void>);
}

TEST(test, unwrap_mixed) {
    static_assert(
        std::same_as<test_t<void, std::optional<md::core::mdspan<
                                      int, md::core::extents<size_t, 1>>>>,
                     int>);
    static_assert(
        std::same_as<test_t<void, std::optional<md::core::mdspan<
                                      float, md::core::extents<size_t, 1>>>>,
                     float>);
    static_assert(std::same_as<
                  test_t<void, md::core::mdspan<std::optional<int>,
                                                md::core::extents<size_t, 1>>>,
                  int>);
    static_assert(std::same_as<
                  test_t<void, md::core::mdspan<std::optional<float>,
                                                md::core::extents<size_t, 1>>>,
                  float>);

    // static_assert(
    //     std::same_as<
    //         test_t<void, std::optional<md::core::mdspan<
    //                          std::nullopt_t, md::core::extents<size_t, 1>>>>,
    //         void>);
    // static_assert(std::same_as<
    //               test_t<void,
    //               md::core::mdspan<std::optional<std::nullopt_t>,
    //                                             md::core::extents<size_t,
    //                                             1>>>,
    //               void>);
}

TEST(test, define_dtype) {
    static_assert(std::same_as<test_t<int, std::uint8_t>, int>);
    static_assert(std::same_as<test_t<int, std::uint16_t>, int>);
    static_assert(std::same_as<test_t<int, std::uint32_t>, int>);
    static_assert(std::same_as<test_t<int, std::uint64_t>, int>);
    static_assert(std::same_as<test_t<int, std::int8_t>, int>);
    static_assert(std::same_as<test_t<int, std::int16_t>, int>);
    static_assert(std::same_as<test_t<int, std::int32_t>, int>);
    static_assert(std::same_as<test_t<int, std::int64_t>, int>);
    static_assert(std::same_as<test_t<int, float>, int>);
    static_assert(std::same_as<test_t<int, double>, int>);
    static_assert(std::same_as<test_t<int, long double>, int>);

    static_assert(std::same_as<test_t<int, std::nullopt_t>, int>);
    static_assert(std::same_as<test_t<int, std::optional<int>>, int>);
    static_assert(std::same_as<test_t<int, std::string>, int>);
    static_assert(std::same_as<test_t<int, std::vector<int>>, int>);
    static_assert(std::same_as<test_t<int, std::array<int, 3>>, int>);
    static_assert(std::same_as<test_t<int, std::tuple<int, double>>, int>);
    static_assert(std::same_as<test_t<int, std::pair<int, double>>, int>);
}

TEST(test, filter_nullopt) {
    static_assert(
        std::same_as<test_t<void, std::nullopt_t, std::uint8_t>, std::uint8_t>);
    static_assert(std::same_as<test_t<void, std::nullopt_t, std::uint16_t>,
                               std::uint16_t>);
    static_assert(std::same_as<test_t<void, std::nullopt_t, std::uint32_t>,
                               std::uint32_t>);
    static_assert(std::same_as<test_t<void, std::nullopt_t, std::uint64_t>,
                               std::uint64_t>);
    static_assert(
        std::same_as<test_t<void, std::nullopt_t, std::int8_t>, std::int8_t>);
    static_assert(
        std::same_as<test_t<void, std::nullopt_t, std::int16_t>, std::int16_t>);
    static_assert(
        std::same_as<test_t<void, std::nullopt_t, std::int32_t>, std::int32_t>);
    static_assert(
        std::same_as<test_t<void, std::nullopt_t, std::int64_t>, std::int64_t>);
    static_assert(std::same_as<test_t<void, std::nullopt_t, float>, float>);
    static_assert(std::same_as<test_t<void, std::nullopt_t, double>, double>);
    static_assert(
        std::same_as<test_t<void, std::nullopt_t, long double>, long double>);

    static_assert(
        std::same_as<test_t<void, std::uint8_t, std::nullopt_t>, std::uint8_t>);
    static_assert(std::same_as<test_t<void, std::uint16_t, std::nullopt_t>,
                               std::uint16_t>);
    static_assert(std::same_as<test_t<void, std::uint32_t, std::nullopt_t>,
                               std::uint32_t>);
    static_assert(std::same_as<test_t<void, std::uint64_t, std::nullopt_t>,
                               std::uint64_t>);
    static_assert(
        std::same_as<test_t<void, std::int8_t, std::nullopt_t>, std::int8_t>);
    static_assert(
        std::same_as<test_t<void, std::int16_t, std::nullopt_t>, std::int16_t>);
    static_assert(
        std::same_as<test_t<void, std::int32_t, std::nullopt_t>, std::int32_t>);
    static_assert(
        std::same_as<test_t<void, std::int64_t, std::nullopt_t>, std::int64_t>);
    static_assert(std::same_as<test_t<void, float, std::nullopt_t>, float>);
    static_assert(std::same_as<test_t<void, double, std::nullopt_t>, double>);
    static_assert(
        std::same_as<test_t<void, long double, std::nullopt_t>, long double>);
}
