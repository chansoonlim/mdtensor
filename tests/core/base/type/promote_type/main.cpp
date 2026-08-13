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

template <typename... Ts>
using test_t = typename md::core::promote_type_t<Ts...>;

template <typename... Ts>
constexpr bool assigned = requires { typename test_t<Ts...>; };

TEST(test, assign) {
    static_assert(assigned<bool>);

    static_assert(assigned<std::uint8_t>);
    static_assert(assigned<std::uint16_t>);
    static_assert(assigned<std::uint32_t>);
    static_assert(assigned<std::uint64_t>);

    static_assert(assigned<std::int8_t>);
    static_assert(assigned<std::int16_t>);
    static_assert(assigned<std::int32_t>);
    static_assert(assigned<std::int64_t>);

    static_assert(assigned<float>);
    static_assert(assigned<double>);
    static_assert(assigned<long double>);

    static_assert(!assigned<void>);
    static_assert(!assigned<std::nullopt_t>);
    static_assert(!assigned<std::optional<int>>);
    static_assert(!assigned<std::string>);
    static_assert(!assigned<std::vector<int>>);
    static_assert(!assigned<std::array<int, 3>>);
    static_assert(!assigned<std::tuple<int, double>>);
    static_assert(!assigned<std::pair<int, double>>);
}

TEST(test, sample) {
    static_assert(std::same_as<test_t<bool, bool>, bool>);
    static_assert(std::same_as<test_t<bool, std::int16_t>, std::int16_t>);
    static_assert(std::same_as<test_t<bool, std::uint32_t>, std::uint32_t>);
    static_assert(
        std::same_as<test_t<std::int8_t, std::int32_t>, std::int32_t>);
    static_assert(
        std::same_as<test_t<std::uint8_t, std::uint32_t>, std::uint32_t>);
    static_assert(
        std::same_as<test_t<std::int8_t, std::uint8_t>, std::int16_t>);
    static_assert(
        std::same_as<test_t<std::int16_t, std::uint16_t>, std::int32_t>);
    static_assert(
        std::same_as<test_t<std::int32_t, std::uint32_t>, std::int64_t>);
    static_assert(
        std::same_as<test_t<std::int64_t, std::uint32_t>, std::int64_t>);

    static_assert(std::same_as<test_t<std::int8_t, std::uint64_t>, double>);

    static_assert(std::same_as<test_t<std::int32_t, std::uint64_t>, double>);

    static_assert(std::same_as<test_t<std::int64_t, std::uint64_t>, double>);

    static_assert(std::same_as<test_t<std::int8_t, float>, float>);

    static_assert(std::same_as<test_t<std::uint16_t, float>, float>);

    static_assert(std::same_as<test_t<std::int32_t, float>, double>);

    static_assert(std::same_as<test_t<std::uint32_t, float>, double>);

    static_assert(std::same_as<test_t<std::int64_t, float>, double>);

    static_assert(std::same_as<test_t<std::uint64_t, float>, double>);
}
